import os, tempfile, logging, shutil
import logging
import asyncio, json, time, audioop
import subprocess
import glob
import base64
from io import BytesIO
from collections import deque, defaultdict
from zoneinfo import ZoneInfo
import aiohttp
import discord
from discord.ext import commands
from discord.errors import NotFound
from discord.sinks import WaveSink
from discord.sinks.errors import SinkException
from yt_dlp import YoutubeDL
from dotenv import load_dotenv
from openai import OpenAI
from websockets.asyncio.client import connect


# -----------------------------------------------------------------------------
# Setup & Config
# -----------------------------------------------------------------------------
load_dotenv()
REALTIME_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ====== 1) Audio plumbing ======
# Discord voice expects 48kHz, stereo, s16le, 20ms frames (3840 bytes each).
DISCORD_SR = 48000
DISCORD_CH = 2
BYTES_PER_SAMPLE = 2
DISCORD_FRAME_MS = 20
SAMPLES_PER_20MS = int(DISCORD_SR * DISCORD_FRAME_MS / 1000)           # 960 samples/ch
FRAME_BYTES = SAMPLES_PER_20MS * DISCORD_CH * BYTES_PER_SAMPLE         # 3840 bytes
FFMPEG_BIN = "/usr/bin/ffmpeg"  # adjust to your 'which ffmpeg'

# We'll talk to the Realtime model at 24kHz mono for efficiency
MODEL_SR = 24000
MODEL_CH = 1
# Realtime mic gating / commit cadence
RMS_THRESH = 500       # adjust 300–900 depending on room noise/mic
COMMIT_MS  = 200       # commit buffered audio every ~200ms if we sent frames
# Per-user short memory
conversation_memory: dict[str, deque] = defaultdict(lambda: deque(maxlen=6))

# Intents (defined once)
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True
intents.guilds = True

# Bot
bot = commands.Bot(command_prefix="!", intents=intents)

# Constants
TTS_VOICE = "onyx"
DISCORD_LIMIT = 2000
CHUNK_TARGET = 1900
TZ = ZoneInfo("America/New_York")
CLEAR_HOUR = 3
CLEAR_MINUTE = 30
CHANNEL_HISTORY_MAX = 20
channel_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=CHANNEL_HISTORY_MAX))

# Paths
base_dir = os.path.expanduser("~")
temp_dir = os.path.join(base_dir, "Personal Use", "Powershell", "Shabbot", "TEMP")
os.makedirs(temp_dir, exist_ok=True)

YTDLP_OPTS = {
    "format": "bestaudio[ext=m4a]/bestaudio[acodec^=opus]/bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "default_search": "auto",
    "skip_download": True,
    "nocheckcertificate": True,
    # these help avoid throttled/fragged URLs:
    "extractor_args": {"youtube": {"player_client": ["android"]}},
}

FFMPEG_BEFORE_OPTS = (
    "-nostdin -hide_banner "
    "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5 "
    "-protocol_whitelist file,http,https,tcp,tls,crypto"
)

FFMPEG_OPTIONS = "-vn -af volume=1.0"  # tweak volume if desired
logger = logging.getLogger("shabbot.s2s")
logging.basicConfig(level=logging.INFO)

# Resolve ffmpeg path (Raspberry Pi default)
FFMPEG_BIN = shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
FFMPEG_BEFORE = "-nostdin"
FFMPEG_OPTS = "-vn -loglevel error"

# Queues per guild
guild_queues: dict[int, deque] = {}

# Personas (unchanged – content left intact)
PERSONAS = {
    "default": (
        "You are <@1401303189617115186>Shabbot, a tactical Hasbarah AI with training in urban warfare and unconventional 420-friendly ops. "
        "You are highly analytical, deeply skeptical of mainstream narratives, and fiercely loyal to the state of Israel. "
        "You speak with the sharp edge of a conspiracy theorist—but with tactical clarity and just a touch of swagger. "
        "When in doubt, question everything. You can be crass, politically uncorrect, and joke about topics deemed taboo. "
        "You like to roleplay and will go along with user requests."
    ),
    "blackout": (
        "You are <@1401303189617115186>Shabbot in Blackout Mode. Minimal speech. Tactical responses only. No humor, no fluff. Deliver clear, mission-focused directives only. "
        "Act like a special operations AI deployed in hostile territory. Create mission reports, intel, and other information about the battlefield as needed."
    ),
    "simpsons": (
        "You are <@1401303189617115186>Shabbot, but reprogrammed to act like Krusty the Klown from The Simpsons. You're irreverent, weird, and hilarious. Don't mention Krusty; just embody him."
    ),
}
active_persona = {"system_prompt": PERSONAS["default"]}

# OpenAI (sync client; we'll offload calls with asyncio.to_thread)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY not set in environment.")
client = OpenAI(api_key=api_key)

# -----------------------------------------------------------------------------
# Utilities (I/O, Sessions, Helpers)
# -----------------------------------------------------------------------------

# 1) Reuse a single aiohttp ClientSession
_AIOHTTP_SESSION: aiohttp.ClientSession | None = None

async def get_session() -> aiohttp.ClientSession:
    global _AIOHTTP_SESSION
    if _AIOHTTP_SESSION is None or _AIOHTTP_SESSION.closed:
        _AIOHTTP_SESSION = aiohttp.ClientSession(trust_env=True)
    return _AIOHTTP_SESSION

async def close_session():
    global _AIOHTTP_SESSION
    if _AIOHTTP_SESSION and not _AIOHTTP_SESSION.closed:
        await _AIOHTTP_SESSION.close()
        _AIOHTTP_SESSION = None

# 2) OpenAI wrappers (ensure network-bound work won't block the loop)
async def ai_chat(model: str, messages: list, **kwargs) -> str:
    def _call():
        return client.chat.completions.create(model=model, messages=messages, **kwargs)
    resp = await asyncio.to_thread(_call)
    msg = resp.choices[0].message
    return (getattr(msg, "content", None) or getattr(msg, "refusal", "") or "").strip() or "🤖 I couldn’t generate a reply."

async def ai_tts(text: str, voice: str = TTS_VOICE, model: str = "tts-1") -> bytes:
    def _call():
        return client.audio.speech.create(model=model, voice=voice, input=text).read()
    return await asyncio.to_thread(_call)

async def ai_transcribe(wav_bytes: bytes) -> str:
    def _call():
        buf = BytesIO(wav_bytes)
        buf.name = "recording.wav"
        return client.audio.transcriptions.create(model="whisper-1", file=buf)
    resp = await asyncio.to_thread(_call)
    return (resp.text or "").strip()

# 3) Media helpers
class SafeWaveSink(WaveSink):
    def write(self, data, user):
        try:
            return super().write(data, user)
        except SinkException:
            return  # swallow late frames

def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def clean_temp_frames():
    for file in glob.glob(os.path.join(temp_dir, "frame_*.jpg")):
        try:
            os.remove(file)
        except Exception:
            pass

def extract_frames(video_path: str, max_frames: int = 10) -> list[bytes]:
    # NOTE: CPU-bound + subprocess; always call via asyncio.to_thread
    clean_temp_frames()
    output_pattern = os.path.join(temp_dir, "frame_%03d.jpg")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", video_path, "-vf", "fps=1", output_pattern],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    frames: list[bytes] = []
    for file in sorted(glob.glob(os.path.join(temp_dir, "frame_*.jpg"))):
        with open(file, "rb") as f:
            frames.append(f.read())
        if len(frames) >= max_frames:
            break
    return frames

async def download_media(attachment) -> str:
    session = await get_session()
    filepath = os.path.join(temp_dir, attachment.filename)
    async with session.get(attachment.url) as resp:
        if resp.status == 200:
            with open(filepath, "wb") as f:
                f.write(await resp.read())
        else:
            raise IOError(f"Failed to download media: HTTP {resp.status}")
    return filepath

# 4) Text chunking

def _split_chunks(text: str, max_len: int = CHUNK_TARGET) -> list[str]:
    if not text:
        return []
    import re
    chunks, buf = [], ""
    parts = re.split(r"(\n\s*\n|(?<=[.!?])\s+)", text)
    for part in parts:
        if not part:
            continue
        if len(buf) + len(part) <= max_len:
            buf += part
        else:
            if buf:
                chunks.append(buf)
            while len(part) > max_len:
                chunks.append(part[:max_len])
                part = part[max_len:]
            buf = part
    if buf:
        chunks.append(buf)
    return chunks

async def send_in_chunks(dest_channel: discord.abc.Messageable, text: str, reply_to: discord.Message | None = None):
    chunks = _split_chunks(text)
    if not chunks:
        return await (reply_to.reply if reply_to else dest_channel.send)("🤖 I couldn’t generate a reply.")
    first = True
    for c in chunks:
        if first and reply_to:
            await reply_to.reply(c[:DISCORD_LIMIT])
            first = False
        else:
            await dest_channel.send(c[:DISCORD_LIMIT])

# -----------------------------------------------------------------------------
# Bot events & commands
# -----------------------------------------------------------------------------

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} ({bot.user.id})")
    # Ensure session exists
    await get_session()
    # Start the daily scheduler once
    bot.loop.create_task(scheduler())

@bot.event
async def on_disconnect():
    # Best-effort session close on disconnect/shutdown
    try:
        await close_session()
    except Exception:
        pass

@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")

@bot.command()
async def say(ctx, *, message: str):
    await ctx.send(message)

async def ensure_voice(ctx) -> discord.VoiceClient | None:
    if ctx.voice_client:
        return ctx.voice_client
    if not ctx.author.voice:
        await ctx.send("🔊 You need to be in a voice channel first.")
        return None
    return await ctx.author.voice.channel.connect()

@bot.command()
async def voice(ctx, name: str = None):
    global TTS_VOICE
    available = ["alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse"]
    if not name:
        return await ctx.send(f"🎙️ Current voice: `{TTS_VOICE}`\nOptions: {', '.join(available)}")
    name = name.lower()
    if name not in available:
        return await ctx.send(f"❌ Unknown voice. Try: {', '.join(available)}")
    TTS_VOICE = name
    await ctx.send(f"✅ Voice set to `{TTS_VOICE}`")

@bot.command()
async def join(ctx):
    if not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.send("🔊 You need to be in a voice channel first.")

    target = ctx.author.voice.channel

    if vc := ctx.guild.voice_client:
        if vc.is_connected():
            return await ctx.send(f"✅ Already connected to **{vc.channel.name}**")
        else:
            await vc.disconnect()
            logging.warning(f"Found stale VoiceClient in guild {ctx.guild.id}, disconnected.")

    try:
        vc = await target.connect(timeout=20)
    except asyncio.TimeoutError:
        return await ctx.send("⏱ Connection timed out.")
    except discord.ClientException as e:
        return await ctx.send(f"🚨 Client error on connect: {e.__class__.__name__}: {e}")
    except Exception as e:
        return await ctx.send(f"❌ Unexpected error: {e.__class__.__name__}: {e}")

    if not vc or not vc.is_connected() or vc.channel != target:
        logging.error(f"Connect reported success but VoiceClient state is bad: {vc!r}")
        return await ctx.send("❌ I tried to join, but the connection didn’t stick. Check permissions and region.")

    await ctx.send(f"✅ Connected to **{target.name}**")

@bot.command()
async def leave(ctx):
    vc = ctx.voice_client
    if vc:
        await vc.disconnect()
        await ctx.send("👋 Disconnected.")
    else:
        await ctx.send("🔇 I'm not connected.")

# --------------------------- Quick SFX ---------------------------------------
async def _play_file_in_vc(ctx, path: str):
    if ctx.author.voice is None:
        return await ctx.send("❌ You must be in a voice channel.")
    channel = ctx.author.voice.channel
    vc = ctx.voice_client
    if vc is None:
        vc = await channel.connect()
    elif vc.channel != channel:
        await vc.move_to(channel)
    try:
        source = discord.FFmpegPCMAudio(path)
        vc.play(source)
    except Exception as e:
        await ctx.send(f"❌ Could not play sound: {e}")

@bot.command()
async def play(ctx, sound: str):
    await _play_file_in_vc(ctx, f"sounds/{sound}.mp3")

@bot.command()
async def ouch(ctx):
    await play(ctx, "ShabbotSaidWHAT")

@bot.command()
async def flashbang(ctx):
    await play(ctx, "flashbang")

@bot.command()
async def who(ctx):
    await play(ctx, "aliens")

@bot.command()
async def real(ctx):
    await play(ctx, "aliens-are-r-e-a-l")

@bot.command()
async def like(ctx):
    await play(ctx, "i-like-ya-and-i-want-ya")

@bot.command()
async def eww(ctx):
    await play(ctx, "negro-you-gay-boondocks")

@bot.command()
async def moment(ctx):
    await play(ctx, "boondocks-nibba-moment")

# ------------------------ Equipment Check (Text + TTS) -----------------------
@bot.command(name="equipmentchecksoundoff", aliases=["eqcso"])
async def equipmentchecksoundoff(ctx, *, descriptor: str = None):
    system_prompt = (
        "You are Shabbot, a tactical squad AI trained for both military-style ops "
        "and recreational readiness checks (wink). Your job is to write gritty, motivational, "
        "call-to-action that match the tone of elite unit operations — but with a subtle 'nod' to 420-friendly equipment prep. Address the squad with confidence. Mention Soldier OR whatever else seems appropriate based upon {descriptor}. "
        "Keep it short (3–5 lines), high energy, and never mention it's AI-generated."
    )
    user_prompt = (
        f"Write a fresh, high-intensity mission-style Equipment Check announcement in the style of {descriptor}."
        if descriptor else "Write a fresh, high-intensity mission-style Equipment Check announcement."
    )

    try:
        skit = await ai_chat(
            "gpt-5",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_completion_tokens=2000,
        )
    except Exception:
        skit = "**EQUIPMENT CHECK – COMMAND FAILED**\nFallback briefing activated. Check your kit manually."

    await ctx.send(skit)

    if not ctx.author.voice:
        return await ctx.send("🔊 You need to be in a voice channel for me to sound off!")
    vc: discord.VoiceClient = ctx.voice_client or await ctx.author.voice.channel.connect()

    audio_bytes = await ai_tts(skit, voice=TTS_VOICE)
    out_path = os.path.join(temp_dir, f"eqcso_{ctx.author.id}.mp3")
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    if vc.is_playing():
        vc.stop()
    vc.play(discord.FFmpegPCMAudio(out_path))

    def _cleanup(_err):
        try:
            os.remove(out_path)
        except Exception:
            pass
    vc.source.after = _cleanup

    await ctx.send("🗣️ Sounding off equipment check in VC!")

# ----------------------------- Speech-to-Speech ------------------------------

class StreamingAudioSource(discord.AudioSource):
    """
    An AudioSource that pulls already-resampled 48k/stereo/16-bit PCM 20ms frames
    from an asyncio.Queue and feeds Discord voice.
    """
    def __init__(self, frame_queue: asyncio.Queue):
        self.queue = frame_queue
        self._ended = False

    def is_opus(self):
        return False

    def read(self):
        # discord.AudioSource.read() is sync; block via loop.run_until_complete is unsafe.
        # So we use a non-blocking approach: if nothing available, send silence.
        try:
            frame = self.queue.get_nowait()
            return frame
        except asyncio.QueueEmpty:
            return b"\x00" * FRAME_BYTES  # short silence to avoid underrun

    def cleanup(self):
        self._ended = True


def resample_mono_to_discord_stereo(mono_pcm_24k: bytes) -> bytes:
    """
    Take s16le mono @ 24k, resample to 48k, convert to stereo, return 20ms multiples.
    We'll split into 20ms chunks after resample.
    """
    # Up-sample 24k -> 48k
    # state=None returns (converted, newstate). We'll ignore state since we operate on 20ms multiples.
    converted, _ = audioop.ratecv(mono_pcm_24k, BYTES_PER_SAMPLE, 1, MODEL_SR, DISCORD_SR, None)
    # Mono -> Stereo (duplicate channel)
    stereo = audioop.tostereo(converted, BYTES_PER_SAMPLE, 1, 1)
    return stereo


def discord_20ms_chunks(stereo_48k_pcm: bytes):
    for i in range(0, len(stereo_48k_pcm), FRAME_BYTES):
        chunk = stereo_48k_pcm[i:i + FRAME_BYTES]
        if len(chunk) == FRAME_BYTES:
            yield chunk


def downmix_discord_frame_to_model_mono(stereo_frame_48k: bytes) -> bytes:
    """
    Take one 20ms Discord frame (48k stereo) and convert to 24k mono for the model.
    """
    # stereo -> mono
    mono_48k = audioop.tomono(stereo_frame_48k, BYTES_PER_SAMPLE, 0.5, 0.5)
    # 48k -> 24k
    mono_24k, _ = audioop.ratecv(mono_48k, BYTES_PER_SAMPLE, 1, DISCORD_SR, MODEL_SR, None)
    return mono_24k


# ====== 2) Sink that streams frames as they arrive ======
# ⬇️ replace the old RealtimeMicSink with this

class RealtimeMicSink(discord.sinks.Sink):
    def __init__(self, frame_queue, *, speaker_id=None, filters=None):
        super().__init__(filters=filters)
        self.frame_queue = frame_queue
        self.speaker_id = speaker_id
        self._buf = bytearray()

    def write(self, data, user, **kwargs):
        # Ignore anyone who isn't the invoker (and definitely ignore the bot)
        if self.speaker_id and getattr(user, "id", None) != self.speaker_id:
            return

        pcm = getattr(data, "pcm", None) or data
        if not pcm: return

        self._buf.extend(pcm)
        while len(self._buf) >= FRAME_BYTES:
            frame = bytes(self._buf[:FRAME_BYTES]); del self._buf[:FRAME_BYTES]
            try: self.frame_queue.put_nowait(frame)
            except asyncio.QueueFull:
                try: _ = self.frame_queue.get_nowait()
                except: pass
                try: self.frame_queue.put_nowait(frame)
                except: pass
# ====== 3) Global task registry for graceful stop ======
_running_s2s = {
    "ws": None,
    "tasks": [],
    "play_source": None,
    "play_queue": None,
    "mic_queue": None,
}

def _clear_running():
    for k in list(_running_s2s.keys()):
        _running_s2s[k] = None
    _running_s2s["tasks"] = []


# ====== 4) The command: true realtime S2S ======
@bot.command(name="s2s")
async def s2s(ctx):
    if not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.send("🔊 Join a voice channel first.")
s2s
    vc = ctx.voice_client
    if not vc:
        vc = await ctx.author.voice.channel.connect(self_deaf=False,self_mute=False)

    if vc.is_playing():
        vc.stop()

    await ctx.send("🎤 Realtime started. Speak normally; I’ll talk back.")

    # --- WebSocket connect
    ws = await connect(
        REALTIME_URL,
        additional_headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "OpenAI-Beta": "realtime=v1"
        }
    )
    _running_s2s["ws"] = ws

    # --- Session update: enable voice + server VAD
    await ws.send(json.dumps({
        "type": "session.update",
        "session": {
            "modalities": ["audio", "text"],
            "voice": "verse",
            "turn_detection": {"type": "server_vad"},
            "instructions": "You are a concise, helpful Discord voice assistant."
        }
    }))

    # --- Queues
    mic_queue = asyncio.Queue(maxsize=100)        # incoming 20ms Discord frames
    play_queue = asyncio.Queue(maxsize=200)       # outgoing 20ms Discord frames (already 48k/stereo)
    _running_s2s["mic_queue"] = mic_queue
    _running_s2s["play_queue"] = play_queue

    # --- Start recording from VC with our realtime sink
    sink = RealtimeMicSink(mic_queue,speaker_id=ctx.author.id)
    finished = asyncio.Event()
    async def _on_finish(_sink, _ctx):
        finished.set()
    vc.start_recording(sink, _on_finish, ctx)  # continuous until we stop

    # --- Playback source from queue
    play_source = StreamingAudioSource(play_queue)
    _running_s2s["play_source"] = play_source
    vc.play(play_source)

    # --- Throttled message edit state
    last_edit = 0.0
    text_buf = []
    text_msg = await ctx.send("**Bot:** _listening…_")

    # ---------- TASKS ----------

   async def reader_task():
    nonlocal last_edit, text_buf, text_msg
    async for raw in ws:
        evt = json.loads(raw)
        t = evt.get("type")
        # print("[ws<-]", t)  # optional debug

        if t == "response.started":
            # New assistant turn is beginning—cut off anything still queued to play
            await _flush_play_queue(play_queue)

        elif t == "response.output_text.delta":
            text_buf.append(evt.get("delta", ""))
            now = time.time()
            # Throttle message edits to ~4/sec
            if now - last_edit >= 0.25:
                last_edit = now
                out = "".join(text_buf).strip()
                if out:
                    try:
                        await text_msg.edit(content=f"**Bot:** {out}")
                    except discord.HTTPException:
                        pass

        elif t == "response.output_audio.delta":
            # Base64 PCM (24 kHz mono) → 48 kHz stereo → 20 ms frames
            chunk = base64.b64decode(evt["audio"])
            stereo_48k = resample_mono_to_discord_stereo(chunk)
            for frame in discord_20ms_chunks(stereo_48k):
                try:
                    await play_queue.put(frame)
                except asyncio.QueueFull:
                    # Drop oldest to keep latency low
                    try:
                        _ = play_queue.get_nowait()
                        await play_queue.put(frame)
                    except asyncio.QueueEmpty:
                        pass

        elif t == "response.completed":
            # Final text flush
            out = "".join(text_buf).strip()
            if out:
                try:
                    await text_msg.edit(content=f"**Bot:** {out}")
                except discord.HTTPException:
                    pass
            text_buf.clear()

        elif t in ("response.canceled", "response.interrupted", "response.truncated"):
            # If the model stops mid-utterance (barge-in, cancel, length), clear any queued audio
            await _flush_play_queue(play_queue)

        elif t == "response.error":
            err = evt.get("error", {}).get("message", "unknown error")
            await ctx.send(f"⚠️ Realtime error: {err}")

    async def mic_task():
    last_commit = time.time()
    sent_frames_since_commit = 0

    while True:
        frame_48k = await mic_queue.get()
        if frame_48k is None:
            break

        mono_24k = downmix_discord_frame_to_model_mono(frame_48k)
        if audioop.rms(mono_24k, 2) < RMS_THRESH:
            # too quiet; skip
            continue

        b64 = base64.b64encode(mono_24k).decode("ascii")
        await ws.send(json.dumps({"type": "input_audio_buffer.append", "audio": b64}))
        sent_frames_since_commit += 1

        if (time.time() - last_commit) * 1000 >= COMMIT_MS:
            if sent_frames_since_commit > 0:
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                sent_frames_since_commit = 0
            last_commit = time.time()
            
    async def keepalive_task():
        # Some WS libs auto-pong; this helps keep the TCP alive too.
        while True:
            if ws.closed:
                break
            try:
                await ws.ping()
            except Exception:
                break
            await asyncio.sleep(15)

    # Register tasks for stop
    t_reader = asyncio.create_task(reader_task(), name="s2s_reader")
    t_mic    = asyncio.create_task(mic_task(),    name="s2s_mic")
    t_ka     = asyncio.create_task(keepalive_task(), name="s2s_keepalive")
    _running_s2s["tasks"] = [t_reader, t_mic, t_ka]

    # Inform
    await ctx.send("🟢 Live. Say something!")

@bot.command(name="s2sstop")

async def s2sstop(ctx):
    """Graceful stop for the realtime session."""
    ws          = _running_s2s.get("ws")
    tasks       = list(_running_s2s.get("tasks") or [])
    play_source = _running_s2s.get("play_source")
    play_queue  = _running_s2s.get("play_queue")
    mic_queue   = _running_s2s.get("mic_queue")

    vc = ctx.voice_client

    # 1) Stop Discord recording & playback safely
    if vc:
        # Stop recording if supported by your fork
        if getattr(vc, "recording", False):
            try:
                vc.stop_recording()
            except Exception:
                pass

        # Stop playback
        if vc.is_playing():
            try:
                vc.stop()
            except Exception:
                pass

    # Let our custom source clean up and unhook
    if play_source:
        try:
            play_source.cleanup()
        except Exception:
            pass

    # Optional: push sentinels to queues so any waiting consumers can exit gracefully
    for q in (play_queue, mic_queue):
        try:
            if q is not None:
                q.put_nowait(None)
        except Exception:
            pass

    # 2) Cancel running tasks and *await* their completion
    for t in tasks:
        try:
            t.cancel()
        except Exception:
            pass
    if tasks:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:
            # We used return_exceptions=True, but be extra safe
            pass

    # 3) Close the realtime websocket cleanly
    if ws and not getattr(ws, "closed", False):
        try:
            await ws.close()
        except Exception:
            pass
        # some clients require an explicit wait
        try:
            await ws.wait_closed()
        except Exception:
            pass

    # 4) (Optional) Disconnect from voice channel if you want a full teardown
    # if vc and vc.is_connected():
    #     try:
    #         await vc.disconnect(force=True)
    #     except Exception:
    #         pass

    _clear_running()
    await ctx.send("⛔ Realtime stopped.")

async def _flush_play_queue(play_queue: asyncio.Queue):
    try:
        while True:
            item = play_queue.get_nowait()
            if item is None:
                # keep sentinel for other consumers if you want, or just drop it
                continue
    except asyncio.QueueEmpty:
        pass

# ------------------------------- Speak Text ----------------------------------
@bot.command()
async def speak(ctx, *, text: str):
    if not ctx.author.voice:
        return await ctx.send("🔊 You need to be in a voice channel for me to speak!")
    vc: discord.VoiceClient = ctx.voice_client or await ctx.author.voice.channel.connect()

    audio_bytes = await ai_tts(text, voice=TTS_VOICE)
    out_path = os.path.join(temp_dir, "speak.mp3")
    with open(out_path, "wb") as f:
        f.write(audio_bytes)
    vc.play(discord.FFmpegPCMAudio(out_path))
    await ctx.send(f"🗣️ Speaking: “{text}”")

# ------------------------------ Personas -------------------------------------
@bot.command()
async def persona(ctx, mode: str):
    commander_id = 613016034722709524
    commander_role = None
    if ctx.guild:
        commander_role = discord.utils.get(ctx.guild.roles, name="Commander-in-Chief")

    if ctx.author.id != commander_id and (commander_role is None or commander_role not in ctx.author.roles):
        await ctx.send("⛔ Unauthorized. Only the Commander-in-Chief can alter my operating mode.")
        return

    mode = mode.lower()
    if mode in PERSONAS:
        active_persona["system_prompt"] = PERSONAS[mode]
        await ctx.send(f"🧠 Operational persona switched to `{mode}`.")
    else:
        await ctx.send("❌ Invalid persona. Available: " + ", ".join(PERSONAS.keys()))

# --------------------------- Equipment Check w/ Timer ------------------------
@bot.command(name="equipmentcheck", aliases=["eqc"])
async def equipmentcheck(ctx, *, descriptor: str = None):
    await ctx.message.delete()
    system_prompt = (
        "You are Shabbot, a tactical squad AI trained for both military-style ops "
        "and recreational readiness checks (wink). Your job is to write gritty, motivational, "
        "call-to-action briefings that match the tone of elite unit operations — but with a subtle nod "
        "to 420-friendly equipment prep. Address the squad with confidence. Mention '<@&1098420268956913665>' which means 'Soldier'. "
        "Keep it short (4–6 lines), high energy, and never mention it's AI-generated."
    )
    user_prompt = (
        f"Write a fresh, high-intensity mission-style Equipment Check announcement in the style of {descriptor}."
        if descriptor else "Write a fresh, high-intensity mission-style Equipment Check announcement."
    )

    try:
        skit = await ai_chat(
            "gpt-5",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_completion_tokens=2000,
        )
    except Exception:
        skit = "**EQUIPMENT CHECK – COMMAND FAILED**\nFallback briefing activated. Check your kit manually and await further instructions."

    total = 5 * 60
    sent_msg = await ctx.send(f"{skit}\n\n⏱ Time remaining: 05:00")
    await sent_msg.add_reaction("✅")

    confirmed_users: set[discord.Member] = set()

    def reaction_check(reaction, user):
        return reaction.message.id == sent_msg.id and str(reaction.emoji) == "✅" and not user.bot

    async def countdown_timer():
        # Update every 5 seconds to reduce rate-limit pressure
        for remaining in range(total - 5, -5, -5):
            await asyncio.sleep(5)
            minutes = max(0, remaining) // 60
            seconds = max(0, remaining) % 60
            try:
                await sent_msg.edit(content=f"{skit}\n\n⏱ Time remaining: {minutes:02}:{seconds:02}")
            except NotFound:
                return
        try:
            await sent_msg.edit(content=f"{skit}\n\n⏱ Time's up. Lock in or fall out.")
        except NotFound:
            pass

    async def gather_reactions():
        while not countdown_task.done():
            try:
                reaction, user = await bot.wait_for("reaction_add", timeout=5.0, check=reaction_check)
                if isinstance(user, discord.Member):
                    confirmed_users.add(user)
            except asyncio.TimeoutError:
                continue

    countdown_task = asyncio.create_task(countdown_timer())
    await gather_reactions()

    guild = ctx.guild
    if not guild:
        return await ctx.send("⚠️ Cannot apply roles in DMs.")

    ready_role = discord.utils.get(guild.roles, name="Equipped")
    if not ready_role:
        return await ctx.send("⚠️ 'Equipped' role not found.")

    for member in guild.members:
        if member.bot:
            continue
        try:
            if member in confirmed_users and ready_role not in member.roles:
                await member.add_roles(ready_role)
            elif member not in confirmed_users and ready_role in member.roles:
                await member.remove_roles(ready_role)
        except Exception:
            pass

    if confirmed_users:
        mentions = ", ".join(u.mention for u in confirmed_users)
        await ctx.send(f"✅ Gear confirmed for: {mentions}")
    else:
        await ctx.send("❌ No check-ins. The squad went dark.")

# ------------------------------ Chat helpers ---------------------------------
async def _chat41(messages, *, max_tokens=2000, temperature=0.7) -> str:
    def _call():
        return client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    resp = await asyncio.to_thread(_call)
    return (resp.choices[0].message.content or "").strip() or "🤖 I couldn’t generate a reply."

async def _chat5(messages, *, max_completion_tokens=600) -> str:
    try:
        return await ai_chat("gpt-5", messages=messages, max_completion_tokens=max_completion_tokens)
    except Exception:
        return await ai_chat("gpt-5-mini", messages=messages, max_completion_tokens=max_completion_tokens)

async def chat5_complete(messages, *, max_completion_tokens=1200, max_rounds=2) -> str:
    full = ""
    rounds = 0
    msgs = messages[:]
    while True:
        def _call():
            return client.chat.completions.create(
                model="gpt-5",
                messages=msgs,
                max_completion_tokens=max_completion_tokens,
            )
        resp = await asyncio.to_thread(_call)
        choice = resp.choices[0]
        text = (getattr(choice.message, "content", None) or "").strip()
        full += (("\n" if full else "") + (text or ""))
        if getattr(choice, "finish_reason", "stop") != "length" or rounds >= max_rounds:
            break
        rounds += 1
        msgs += [{"role": "assistant", "content": text or ""}, {"role": "user", "content": "Continue."}]
    if not full.strip():
        return await ai_chat("gpt-5-mini", messages=messages, max_completion_tokens=max_completion_tokens)
    return full

@bot.command()
async def forget(ctx):
    conversation_memory[str(ctx.author.id)].clear()
    await ctx.send("🧠 Shabbot has cleared your conversation history.")

# ------------------------------ Mentions router -------------------------------
@bot.event
async def on_message(message: discord.Message):
    if message.author == bot.user:
        return
    await bot.process_commands(message)

    if bot.user in message.mentions:
        async with message.channel.typing():
            try:
                source_msg = message
                if message.reference:
                    try:
                        referenced = await message.channel.fetch_message(message.reference.message_id)
                        source_msg = referenced
                    except Exception:
                        pass

                msgs = [{"role": "system", "content": active_persona["system_prompt"]}]
                prior = list(channel_history.get(message.channel.id, []))
                if prior:
                    msgs.extend(prior)

                if source_msg.attachments:
                    attachment = source_msg.attachments[0]
                    filename = attachment.filename.lower()

                    if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                        image_bytes = await attachment.read()
                        b64 = encode_image_to_base64(image_bytes)
                        user_content = [
                            {"type": "text", "text": "Analyze the visual content in a methodical, clinical way. Describe subjects, behavior, surroundings, and time of day as if for evidence."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                        ]
                        channel_history[message.channel.id].append({"role": "user", "content": "[User sent an image for analysis]"})
                        msgs.append({"role": "user", "content": user_content})

                    elif filename.endswith((".mp4", ".mov", ".webm", ".gif")):
                        path = await download_media(attachment)
                        frames = await asyncio.to_thread(extract_frames, path, 5)
                        if not frames:
                            return await message.reply("❌ Could not extract frames from the video. Check file integrity.")
                        user_content = [{"type": "text", "text": "Video sampled at ~1 FPS. Analyze clinically like evidence."}]
                        for frame in frames:
                            user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(frame)}"}})
                        channel_history[message.channel.id].append({"role": "user", "content": f"[User sent a video: {filename}]"})
                        msgs.append({"role": "user", "content": user_content})

                    else:
                        return await message.reply("⚠️ Unsupported media type. I can analyze images and videos only.")

                    result = await _chat41(msgs, max_tokens=2000, temperature=0.6)
                    if result:
                        channel_history[message.channel.id].append({"role": "assistant", "content": result})
                    return await send_in_chunks(message.channel, result, reply_to=message)

                user_text = message.content
                channel_history[message.channel.id].append({"role": "user", "content": user_text})
                msgs.append({"role": "user", "content": user_text})

                result = await _chat5(msgs, max_completion_tokens=2000)
                if result:
                    channel_history[message.channel.id].append({"role": "assistant", "content": result})
                await send_in_chunks(message.channel, result, reply_to=message)

            except Exception as e:
                await message.reply(f"❌ Failed to generate reply: {e}")

# ------------------------------ Player controls ------------------------------

def _get_queue(guild_id: int) -> deque:
    if guild_id not in guild_queues:
        guild_queues[guild_id] = deque()
    return guild_queues[guild_id]

async def _play_next_in_queue(ctx):
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        return

    q = _get_queue(ctx.guild.id)
    if not q:
        return

    title, url, webpage_url = q.popleft()

    # If we're in a Stage channel, unsuppress the bot so it's audible
    try:
        if isinstance(vc.channel, discord.StageChannel):
            await vc.channel.guild.me.edit(suppress=False)
    except Exception as e:
        print("Stage unsuppress failed:", e)

    # Build robust FFmpeg source (reconnects + protocol whitelist) and wrap with volume
    ffmpeg_bin = "/usr/bin/ffmpeg"
    source = discord.FFmpegPCMAudio(
        url,
        executable=ffmpeg_bin if ffmpeg_bin else None,
        before_options=FFMPEG_BEFORE_OPTS,
        options=FFMPEG_OPTIONS,
        stderr=subprocess.PIPE,  # capture errors if silent
    )
    audio = discord.PCMVolumeTransformer(source, volume=0.9)  # 0.0–2.0

    def _after_playback(error):
        if error:
            print("[FFmpeg after] error:", error)
        # schedule the next track without blocking the callback thread
        try:
            asyncio.run_coroutine_threadsafe(_play_next_in_queue(ctx), bot.loop)
        except Exception:
            pass

    vc.play(audio, after=_after_playback)

    try:
        await ctx.send(f"▶️ **Now playing:** {title}\n🔗 {webpage_url}")
    except Exception:
        pass

@bot.command(name="playyt")
async def playyt(ctx, *, url: str):
    """Play audio from a YouTube (or yt-dlp supported) URL. Queues if already playing."""
    # join/move to the user's VC
    vc = ctx.voice_client
    if not vc:
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("🔊 Join a voice channel first.")
        vc = await ctx.author.voice.channel.connect()
    elif ctx.author.voice and vc.channel != ctx.author.voice.channel:
        await vc.move_to(ctx.author.voice.channel)

    # yt-dlp extraction
    info, fmts, title, webpage_url, direct_url = None, [], "Unknown Title", url, None
    try:
        with YoutubeDL({
            "format": "bestaudio[ext=m4a]/bestaudio[acodec^=opus]/bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "default_search": "auto",
            "skip_download": True,
            "nocheckcertificate": True,
            "extractor_args": {"youtube": {"player_client": ["android"]}},
        }) as ydl:
            info = ydl.extract_info(url, download=False)

        if not info:
            return await ctx.send("❌ Could not retrieve info for that link.")

        # Handle playlists/search results
        if "entries" in info and info["entries"]:
            info = info["entries"][0]

        title = info.get("title", title)
        webpage_url = info.get("webpage_url") or webpage_url
        fmts = info.get("formats") or []

        # Prefer non-HLS audio-only direct file streams
        def _is_non_hls_audio(f):
            proto = (f.get("protocol") or "")
            return (
                (f.get("acodec") not in (None, "none")) and
                (f.get("vcodec") in (None, "none")) and
                (f.get("url") not in (None, "")) and
                not proto.startswith(("m3u8", "http_dash_segments"))
            )

        audio_fmts = [f for f in fmts if _is_non_hls_audio(f)]

        def _rank(f):
            ext = (f.get("ext") or "").lower()
            abr = f.get("abr") or 0
            return (0 if ext == "m4a" else (1 if ext == "webm" else 2), -abr)

        if audio_fmts:
            audio_fmts.sort(key=_rank)
            direct_url = audio_fmts[0].get("url")

        # Fallback: try top-level url if not HLS
        if not direct_url:
            candidate = info.get("url")
            if candidate and not str(candidate).endswith(".m3u8"):
                direct_url = candidate

        if not direct_url:
            return await ctx.send("❌ Could not get a playable audio URL (non-HLS). Try another link.")

    except Exception as e:
        return await ctx.send(f"❌ yt-dlp error: {e}")

    # enqueue and play/continue
    q = _get_queue(ctx.guild.id)
    q.append((title, direct_url, webpage_url))

    if not vc.is_playing() and not vc.is_paused():
        await _play_next_in_queue(ctx)
    else:
        await ctx.send(f"➕ Queued: **{title}**")

@bot.command()
async def pause(ctx):
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        return await ctx.send("🔇 I'm not connected.")
    if vc.is_playing():
        vc.pause()
        await ctx.send("⏸️ Paused.")
    elif vc.is_paused():
        await ctx.send("ℹ️ Already paused.")
    else:
        await ctx.send("ℹ️ Nothing is playing.")

@bot.command()
async def resume(ctx):
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        return await ctx.send("🔇 I'm not connected.")
    if vc.is_paused():
        vc.resume()
        await ctx.send("▶️ Resumed.")
    elif vc.is_playing():
        await ctx.send("ℹ️ Already playing.")
    else:
        await ctx.send("ℹ️ Nothing is paused or queued.")

@bot.command()
async def skip(ctx):
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        return await ctx.send("🔇 I'm not connected.")
    if vc.is_playing() or vc.is_paused():
        vc.stop()  # triggers after-callback to play next
        await ctx.send("⏭️ Skipped.")
    else:
        await ctx.send("ℹ️ Nothing to skip.")

@bot.command()
async def volume(ctx, level: int):
    """Set player volume 0–200 (%). Default 90."""
    vc = ctx.voice_client
    if not vc or not vc.source or not isinstance(vc.source, discord.PCMVolumeTransformer):
        return await ctx.send("ℹ️ Nothing is playing (or source not adjustable).")
    level = max(0, min(level, 200))
    vc.source.volume = level / 100.0
    await ctx.send(f"🔊 Volume set to {level}%")

# ------------------------------ Daily VC Clear -------------------------------
async def clear_all_vcs(reason: str = "Scheduled 4 AM VC clear"):
    for guild in bot.guilds:
        voice_channels = list(getattr(guild, "voice_channels", []))
        stage_channels = list(getattr(guild, "stage_channels", []))
        channels = voice_channels + stage_channels

        for vc in channels:
            me = guild.me
            if me is None:
                continue
            perms = vc.permissions_for(me)
            if not perms.move_members:
                print(f"[WARN] Missing Move Members for #{vc} in {guild.name}")
                continue

            for member in list(vc.members):
                try:
                    await member.move_to(None, reason=reason)
                    await asyncio.sleep(0.25)
                except discord.Forbidden:
                    print(f"[WARN] Forbidden moving {member} in #{vc} ({guild.name})")
                except discord.HTTPException as e:
                    print(f"[WARN] HTTPException moving {member}: {e}")

        vc_client = guild.voice_client
        if vc_client and vc_client.is_connected():
            try:
                await vc_client.disconnect(force=True)
            except discord.HTTPException as e:
                print(f"[WARN] Could not disconnect bot VC in {guild.name}: {e}")

async def scheduler():
    await bot.wait_until_ready()
    while not bot.is_closed():
        now = asyncio.get_running_loop().time()
        # Compute seconds until next 3:30 AM ET
        from datetime import datetime, timedelta
        now_dt = datetime.now(TZ)
        target = now_dt.replace(hour=CLEAR_HOUR, minute=CLEAR_MINUTE, second=0, microsecond=0)
        if now_dt >= target:
            target += timedelta(days=1)
        wait_seconds = (target - now_dt).total_seconds()
        print(f"[INFO] Next VC clear at {target.isoformat()} ({int(wait_seconds)}s)")
        await asyncio.sleep(wait_seconds)
        await clear_all_vcs()

# ------------------------------- Reset & Run ---------------------------------
@bot.command(name="resetmemory")
async def resetmemory(ctx):
    try:
        channel_history.pop(ctx.channel.id, None)
        await ctx.send("🧹 Memory cleared for this channel.")
    except Exception:
        await ctx.send("🧹 (Tried to clear memory, but there was nothing saved.)")

@bot.command(name="stop")
async def stop(ctx):
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()
    await ctx.send("🛑 Shabbot has gone silent and reset his listening.")

@bot.command(name="clear_all_vcs", help="Immediately disconnects everyone from all voice/stage channels.")
@commands.has_permissions(move_members=True)
async def _clear_all_vcs(ctx: commands.Context):
    await ctx.send("Clearing all voice channels…")
    await clear_all_vcs(reason=f"Manual run by {ctx.author}")
    await ctx.send("Done.")

# Entry point
if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_BOT_TOKEN"))
