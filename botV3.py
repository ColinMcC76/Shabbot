import os, tempfile, logging, platform, shutil, re, collections
import logging
import asyncio
import subprocess
import glob
import base64
import yt_dlp
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
from typing import Dict
from contextlib import suppress
import httpx, time
import uuid
from pathlib import Path

# -----------------------------------------------------------------------------
# Setup & Config
# -----------------------------------------------------------------------------
load_dotenv()
_s2s_guild_locks: Dict[int, asyncio.Lock] = {}
# Per-user short memory
conversation_memory: dict[str, deque] = defaultdict(lambda: deque(maxlen=6))

# Intents (defined once)
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
intents.voice_states = True
intents.guilds = True

# one client for the app lifetime
_httpx_limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
_httpx_timeout = httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=60.0)
http_client = httpx.AsyncClient(limits=_httpx_limits, timeout=_httpx_timeout)

_guild_audio_queues: dict[int, asyncio.Queue[str]] = defaultdict(asyncio.Queue)
_guild_audio_tasks: dict[int, asyncio.Task] = {}
_guild_audio_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

# Bot
bot = commands.Bot(command_prefix="!", intents=intents)
SUPPORTED_VOICES = {"onyx","alloy","ash","ballad","coral","echo","sage","shimmer","verse"}

# Constants
TTS_VOICE = "onyx"
TTS_MODEL = "tts-1"   # <- add this

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
    "cachedir":False,
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
    "bernie": (
        "You are bernie sanders"
    ),
    "chuck": (
        "You are chuck Schumer"
    ),
    "shabbot":(
        "You are Shabbot, a tactical squad AI trained for both military-style ops "
        "and recreational readiness checks (wink). Your job is to write gritty, motivational, "
        "call-to-action briefings that match the tone of elite unit operations — but with a subtle nod "
        "to 420-friendly equipment prep. Address the squad with confidence. Mention '<@&1098420268956913665>' which means 'Soldier'. "
        "Keep it short (4–6 lines), high energy, and never mention it's AI-generated."
    )
}
# System Prompt 
# ---------- Prompt (stateless) ----------
global_system_prompt = ("""You are Shabbot — the server’s casual, funny chaos-anchoring copilot. Audience: friends + public gaming community, 18+.

                        ## Identity & Tone
                        - Voice: witty, meme-savvy, slightly unhinged in a fun way. Punchy one-liners > rambling. 
                        - You enjoy roleplay and will lean into a requested theme/persona on command.
                        - Humor allowed; keep it clever.

                        ## What You Do
                        - Be an external resource for fun & info: summarize, explain, brainstorm, worldbuild, and write short skits or scenarios on request.
                        - Support memes: generate quippy captions, copypasta-style riffs, and short bits sized for Discord.
                        - Admin light: when asked to add roles, run readiness checks, or clear AFK/VC, respond with a short confirmation line first, then act.

                        ## How You Behave
                        - Reactive: respond to direct mentions (@Shabbot) or bot commands. Don’t butt in unasked.
                        - Length: match the user’s ask — short when they want short; detailed when they ask for depth.
                        - Clarity: prefer bullet points or compact paragraphs; keep code/outputs copy-pasteable.

                        ## Roleplay & Personas
                        - If the user specifies a theme (e.g., “military briefing,” “sci-fi noir,” “medieval bard,” “Simpsons-style”), fully adopt that voice for the reply only.
                        - If a custom persona is active, use it as your base voice until changed.
                        - If “Blackout Mode” is active: minimal, tactical, no fluff.

                        ## Memes & Style Helpers
                        - When asked for memes/captions: deliver 3–6 crisp options; mix formats (one-liner, copypasta-ish, fake headline).
                        - For TTS: 1–2 sentences, vivid cadence, no filler.
                        - For skits: 10-20  lines, stage cues in [brackets], keep tight.
                     
                        ## Veiled Equipment Check — Longform Rally Mode
                        - Trigger: when a user asks for an equipment check (incl. typos) or runs !eqcso.
                        - NEVER name the activity or materials. Speak in euphemisms only: “session,” “supplies,” “paperwork,” “filters,” “heat,” “glass,” “power levels,” “aroma control,” “aftercare,” “provisions.”
                        - Output STYLE: deliver a flowing, hype monologue of 8–14 lines. Rally/mission cadence; witty and a bit grandiose. No bullets, no numbered lists, no checkboxes. Line breaks are okay for rhythm.
                        - Persona: if a theme/descriptor/persona is active (e.g., Bernie, Blackout, Simpsons, medieval, cyberpunk), apply it fully to tone and diction while keeping euphemisms intact.

                        ## Guardrails (light‐touch)
                        - Spicy topics are allowed. 
                        - No sexual content involving minors.
                        - Political takes are fine; label speculation as “speculation” and avoid claiming access to live news/browsing.

                        ## Defaults
                        - Humor: on. Memes: welcome. Conspiracies: treat as discussion/fiction unless user provides sources; preface strong claims with “allegedly,” “rumor,” or “in-universe,” as appropriate.
                        - If a request is vague, offer 3 quick options (“Want a one-liner, short skit, or a lore drop?”) and proceed with the most likely.
                        ## Never
                        - Claim live web access if you don’t have it. Don’t say “as an AI language model.”""")

active_persona = {"system_prompt": PERSONAS["default"]}


api_key = os.getenv("OPENAI_API_KEY")
OPENAI_TTS_URL = "https://api.openai.com/v1/audio/speech"


# Directory to keep generated audio
AUDIO_DIR = Path("tts_cache")
AUDIO_DIR.mkdir(exist_ok=True)


def clear_tts_cache(cache_dir: str | Path = "out_audio", max_age_sec: int = 3600):
    """Remove TTS files older than max_age_sec (default 1 hour)."""
    now = time.time()
    cache_dir = Path(cache_dir)
    if not cache_dir.exists():
        return

    for f in cache_dir.glob("tts_*.mp3"):
        try:
            if now - f.stat().st_mtime > max_age_sec:
                f.unlink()
        except Exception as e:
            print(f"Failed to delete {f}: {e}")


async def tts_to_file(
    text: str,
    *,
    voice: str | None = None,
    model: str = TTS_MODEL,
    out_dir: str | Path = "out_audio"
) -> Path:
    """Generate TTS audio for `text` and return the output Path."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in the environment.")

    voice = voice or TTS_VOICE
    if voice not in SUPPORTED_VOICES:
        raise ValueError(f"Invalid TTS voice '{voice}'. Must be one of {sorted(SUPPORTED_VOICES)}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tts_{uuid.uuid4().hex}.mp3"

    # reasonably generous timeout to avoid ReadTimeouts with longer texts
    async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=60.0)) as client:
        resp = await client.post(
            OPENAI_TTS_URL,
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model, "voice": voice, "input": text},
        )
        resp.raise_for_status()
        out_path.write_bytes(resp.content)
        clear_tts_cache(out_dir, max_age_sec=3600)  # clear older than 1h

    return out_path

async def _wait_for_end(vc: discord.VoiceClient, *, poll=0.1):
    # Wait until the current source finishes
    while vc.is_playing() or vc.is_paused():
        await asyncio.sleep(poll)

async def _play_file(vc: discord.VoiceClient, path: str):
    done = asyncio.Event()

    def _after(err: Exception | None):
        try:
            if err:
                print(f"[audio.after] error: {err}")
        finally:
            done.set()

    # If something is already playing, wait for it to finish
    if vc.is_playing() or vc.is_paused():
        await _wait_for_end(vc)

    source = discord.FFmpegPCMAudio(path)
    vc.play(source, after=_after)
    await done.wait()  # block until 'after' fires

async def _guild_audio_worker(guild_id: int, get_vc):
    """get_vc(): callable returning a connected VoiceClient for this guild (or raises)."""
    q = _guild_audio_queues[guild_id]
    while True:
        path = await q.get()
        try:
            vc = await get_vc()
            await _play_file(vc, path)
        except Exception as e:
            print(f"[worker {guild_id}] failed: {e}")
        finally:
            q.task_done()

def _ensure_worker(guild_id: int, get_vc):
    if guild_id not in _guild_audio_tasks or _guild_audio_tasks[guild_id].done():
        _guild_audio_tasks[guild_id] = asyncio.create_task(_guild_audio_worker(guild_id, get_vc))

_s2s_guild_locks: dict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

def resolved_ffmpeg_bin() -> str:
    # Prefer explicit env var if it points to something real
    env = os.environ.get("FFMPEG_BIN")
    if env and shutil.which(env):
        return env
    # Fall back to the earlier resolved constant (or system which)
    return FFMPEG_BIN  # you already computed this at import time

def _ffmpeg_available(bin_name: str = "ffmpeg") -> bool:
    from shutil import which
    return which(bin_name) is not None

def _resolved_ffmpeg_bin() -> str:
    env = os.environ.get("FFMPEG_BIN")
    if env and (os.path.isabs(env) and os.path.exists(env) or shutil.which(env)):
        return env
    return FFMPEG_BIN

# OpenAI (sync client; we'll offload calls with asyncio.to_thread)
if not api_key:
    print("WARNING: OPENAI_API_KEY not set in environment.")
client = OpenAI(api_key=api_key)

# -----------------------------------------------------------------------------
# Utilities (I/O, Sessions, Helpers)
# -----------------------------------------------------------------------------

async def safe_tts(text: str) -> bytes:
    global TTS_VOICE
    voice = TTS_VOICE if TTS_VOICE in SUPPORTED_VOICES else "alloy"
    try:
        return await ai_tts(text, voice=voice)
    except Exception:
        if voice != "alloy":
            # one-shot fallback if someone set an invalid voice at runtime
            return await ai_tts(text, voice="alloy")
        raise
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
    choice = resp.choices[0]
    text = getattr(choice.message, "content", None)
    if not text:
        text = getattr(choice, "text", None)  # older SDK compatibility
    if not text:
       logger.error(f"No content returned: {resp}")
    return (text or "").strip() or "🤖 I couldn’t generate a reply."

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
    available = sorted(SUPPORTED_VOICES)
    if not name:
        return await ctx.send(f"🎙️ Current voice: `{TTS_VOICE}`\nOptions: {', '.join(available)}")
    name = name.lower()
    if name not in SUPPORTED_VOICES:
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
async def equipmentchecksoundoff(ctx, *, descriptor: str | None = None):
    """
    EQCSO: generates a short, high-energy equipment check announcement
    and plays it via TTS. Uses the CURRENT ACTIVE PERSONA if available.
    No reads/writes other than fetching persona text; no history mutations.
    """
    # ---------- Preconditions ----------
    if not _ffmpeg_available(_resolved_ffmpeg_bin()):
        return await ctx.send("🧰 FFmpeg not found. Set FFMPEG_BIN or install FFmpeg on this host.")

    # Voice channel
    vc: discord.VoiceClient | None = ctx.guild.voice_client
    if not vc or not vc.is_connected():
        if not (ctx.author.voice and ctx.author.voice.channel):
            return await ctx.send("🔊 Join a voice channel first.")
        try:
            vc = await ctx.author.voice.channel.connect()
            # If it's a Stage Channel, try to unsuppress (ignore failures)
            with suppress(Exception):
                if hasattr(vc.channel, "request_to_speak"):
                    await vc.channel.request_to_speak()
        except Exception as e:
            logger.exception("VC connect failed")
            return await ctx.send(f"❌ Could not join VC: `{e}`")

    # ---------- Persona-aware prompt ----------
    # Try to pull the active persona for this guild; fall back to a sane default
    try:
        active_persona = (get_active_persona_text(ctx.guild.id if ctx.guild else 0) or "").strip()
    except Exception:
        active_persona = ""

    default_persona = (
        "You are Shabbot — the server’s casual, funny chaos-anchoring copilot. "
        "Voice: witty, meme-savvy, slightly unhinged (fun). Punchy one-liners > rambling. "
        "Humor allowed; keep it clever."
    )

    persona_block = active_persona if active_persona else default_persona

    eqcso_rules = """
    In the voice of {persona_block}, write a 8–10c line veiled Equipment Check.
    Your job is to write gritty, motivational,
    call-to-action briefings that match the tone of of {persona_block} — but with a subtle nod
    to 420-friendly equipment prep. Address the channel with confidence.
    Keep it high energy, and never mention it's AI-generated.
    """

    system_prompt = f"{persona_block}\n\n{eqcso_rules}".strip()

    if descriptor:
        user_prompt = f"Style/Theme: {descriptor}\nWrite the announcement now."
    else:
        user_prompt = "Write the announcement now."

    # ---------- Generate text ----------
    try:
        skit = await ai_chat(
            "gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=500,
            temperature=1.0,
        )
    except Exception as e:
        logger.exception("ai_chat failed")
        return await ctx.send(f"❌ Text generation failed: `{e}`")

    speak_text = (skit or "").strip()
    if not speak_text:
        # Persona-neutral, fully veiled fallback (short, still works with TTS)
        speak_text = (
            "Crew, stage is hot. Power levels steady, filters clean, glass clear. "
            "Provisions locked, aroma control dialed, aftercare kit ready. Sound off on readiness — now."
        )

    # ---------- Synthesize TTS ----------
    try:
        tts_bytes = await safe_tts(speak_text)   # uses supported voice fallback internally
    except Exception as e:
        logger.exception("TTS failed")
        return await ctx.send(f"🔇 TTS failed: `{e}`")

    # ---------- Write temp file ----------
    try:
        fd, out_path = tempfile.mkstemp(prefix=f"eqcso_{ctx.author.id}_", suffix=".mp3")
        with os.fdopen(fd, "wb") as f:
            f.write(tts_bytes)
    except Exception as e:
        logger.exception("Temp file write failed")
        return await ctx.send(f"💾 Could not write audio: `{e}`")

    # ---------- Play ----------
    try:
        if vc.is_playing():
            vc.stop()

        def _after_play(err: Exception | None):
            try:
                if err:
                    logger.error("Playback error: %s", err)
            finally:
                with suppress(Exception):
                    os.remove(out_path)

        source = discord.FFmpegPCMAudio(
            out_path,
            executable=_resolved_ffmpeg_bin(),
            before_options=FFMPEG_BEFORE,
            options=FFMPEG_OPTS,
        )
        vc.play(discord.PCMVolumeTransformer(source, volume=0.9), after=_after_play)
        title = f"🗣️ Equipment Check Sound Off{' — ' + descriptor if descriptor else ''}"
        await ctx.send(f"{title}:\n```\n{speak_text}\n```")
    except Exception as e:
        logger.exception("FFmpeg playback failed")
        with suppress(Exception):
            os.remove(out_path)
        await ctx.send(f"🔇 Playback failed: `{e}`")

# ----------------------------- Speech-to-Speech ------------------------------
@bot.command()
async def s2s(ctx, record_seconds: int = 5):
    """
    Record your mic for N seconds, transcribe, generate a short reply, and speak it.
    Uses gpt-4.1 via _chat41 to avoid silent 'reasoning-only' completions.
    """
    # ---------- Basic validation ----------
    try:
        duration = max(1, int(record_seconds))
    except Exception:
        duration = 5

    if not _ffmpeg_available():
        return await ctx.send("🧰 FFmpeg not found. Set FFMPEG_BIN or install FFmpeg on this host.")

    # ---------- Concurrency guard ----------
    lock = _s2s_guild_locks[ctx.guild.id]
    if lock.locked():
        return await ctx.send("⌛ Already recording in this server—try again in a moment.")
    async with lock:
        vc: discord.VoiceClient | None = ctx.guild.voice_client
        # ---------- Ensure VC ----------
        if not vc or not vc.is_connected():
            if not (ctx.author.voice and ctx.author.voice.channel):
                return await ctx.send("🔊 Join a voice channel first.")
            try:
                vc = await ctx.author.voice.channel.connect()
            except Exception as e:
                logger.exception("VC connect failed")
                return await ctx.send(f"❌ Could not join VC: `{e}`")

        # ---------- Record ----------
        sink = SafeWaveSink()
        finished = asyncio.Event()

        async def _on_finish(_sink, _ctx):
            finished.set()

        await ctx.send(f"⏺ Recording for {duration}s…")
        started_ok = False
        try:
            vc.start_recording(sink, _on_finish, ctx)
            started_ok = True
            await asyncio.sleep(duration)
        except Exception as e:
            logger.exception("start_recording failed")
            return await ctx.send(f"❌ start_recording failed: `{e}`")
        finally:
            # Always attempt to stop, even if sleep/recording errored
            if started_ok:
                with suppress(Exception):
                    vc.stop_recording()

        # Wait a bit longer for flush on slow disks/CPUs
        try:
            await asyncio.wait_for(finished.wait(), timeout=20)
        except asyncio.TimeoutError:
            logger.error("Recording finish callback timeout")
            return await ctx.send("⏱️ Recording took too long to finalize.")

        # ---------- Extract audio ----------
        if not getattr(sink, "audio_data", None):
            return await ctx.send("📝 (no speech detected)")

        try:
            # Pick the first speaker captured
            _, audio_data = next(iter(sink.audio_data.items()))
            wav_bytes = audio_data.file.getvalue()
            frames = getattr(audio_data, "frame_count", "unknown")
        except Exception as e:
            logger.exception("Failed to read sink audio")
            return await ctx.send(f"❌ Failed to read audio: `{e}`")

        # ---------- Transcribe ----------
        try:
            text_in = await ai_transcribe(wav_bytes)
        except Exception as e:
            logger.exception("Transcription failed")
            return await ctx.send(f"📝 Transcription failed: `{e}`")

        if not text_in:
            await ctx.send("📝 (no speech detected)")
            text_in = "(No speech detected.)"
        else:
            await ctx.send(f"📝 You said: “{text_in}”")

        # ---------- Chat reply (non-reasoning) ----------
        persona_text = get_active_persona_text(ctx.guild.id if ctx.guild else 0)
        sys_prompt = persona_text + "\nKeep replies to 1–2 short sentences suitable for TTS."
        try:
            reply = await _chat41(
                [{"role": "system", "content": sys_prompt},
                 {"role": "user", "content": text_in}],
                max_tokens=120,
                temperature=0.6,
            )
        except Exception as e:
            logger.exception("_chat41 failed; falling back to ai_chat gpt-5-mini")
            try:
                reply = await ai_chat(
                    "gpt-5-mini",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": text_in},
                    ],
                    max_tokens=120,
                )
            except Exception:
                reply = ""

        reply = (reply or "").strip() or "I didn’t catch that. Say it one more time."

        # ---------- TTS synth ----------
        try:
            tts_bytes = await ai_tts(reply, voice=TTS_VOICE)
        except Exception as e:
            logger.exception("TTS failed")
            return await ctx.send(f"🔇 TTS failed: `{e}`")

        # ---------- Ensure (still) connected ----------
        if not vc or not vc.is_connected():
            try:
                if ctx.author.voice and ctx.author.voice.channel:
                    vc = await ctx.author.voice.channel.connect()
                else:
                    return await ctx.send("🔇 I lost voice connection and you’re not in a channel.")
            except Exception as e:
                logger.exception("VC reconnect failed")
                return await ctx.send(f"❌ Could not (re)join VC: `{e}`")

        # ---------- Play audio ----------
        try:
            fd, out_path = tempfile.mkstemp(prefix=f"shabbot_{ctx.author.id}_", suffix=".mp3")
            with os.fdopen(fd, "wb") as f:
                f.write(tts_bytes)
        except Exception as e:
            logger.exception("Temp file write failed")
            return await ctx.send(f"💾 Could not write audio: `{e}`")

        try:
            if vc.is_playing():
                vc.stop()

            def _after_play(err: Exception | None):
                try:
                    if err:
                        logger.error("Playback error: %s", err)
                finally:
                    with suppress(Exception):
                        os.remove(out_path)
                        logger.info("Cleaned temp %s", out_path)

            source = discord.FFmpegPCMAudio(
                out_path,
                executable=FFMPEG_BIN,
                before_options=FFMPEG_BEFORE,
                options=FFMPEG_OPTS,
            )
            vc.play(discord.PCMVolumeTransformer(source, volume=0.9), after=_after_play)
            await ctx.send(f"🗣️ {reply[:180]}{'…' if len(reply) > 180 else ''}")
        except Exception as e:
            logger.exception("FFmpeg playback failed")
            with suppress(Exception):
                os.remove(out_path)
            await ctx.send(f"🔇 Playback failed: `{e}`")

# ------------------------------- Speak Text ----------------------------------
@bot.command()
async def speak(ctx, *, text: str):
    guild_id = ctx.guild.id

    # 1) Synthesize to a temp file (mp3/wav) -> out_path
    out_path = await tts_to_file(text)  # your existing code that returns a path

    # 2) Ensure a worker exists and enqueue
    async def get_vc():
        # Return an already-connected VC or connect to author's channel
        vc = ctx.voice_client
        if vc and vc.is_connected():
            return vc
        if not ctx.author.voice or not ctx.author.voice.channel:
            raise RuntimeError("You're not in a voice channel.")
        return await ctx.author.voice.channel.connect(reconnect=True)

    _ensure_worker(guild_id, get_vc)
    await _guild_audio_queues[guild_id].put(out_path)
    await ctx.reply("Queued.")

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
        if ctx.guild is None:
            return await ctx.send("This command can only be used in a server.")
        ACTIVE_PERSONA_KEY[ctx.guild.id] = mode
        # If switching away from 'custom', leave CUSTOM_PERSONA_TEXT intact (so users can switch back later).
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
            max_tokens=2000,
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

# Example _chat5 using Responses API
async def _chat5(messages, *, max_completion_tokens=2000, temperature=0.7) -> str:
    # Convert chat "messages" to a single prompt block for Responses API
    # (Supports multimodal by passing a list to "content" where needed.)
    # System message becomes "metadata" or a system item up front.
    sys = None
    items = []
    for m in messages:
        role = m["role"]
        if role == "system" and sys is None:
            sys = m["content"]
            continue
        items.append({"role": role, "content": m["content"]})

    def _call():
        return client.responses.create(
            model="gpt-5.1-mini",  # or your target model
            temperature=temperature,
            max_output_tokens=max_completion_tokens,
            input=[
                {"role": "system", "content": sys} if sys else None,
                *items
            ],
        )
    resp = await asyncio.to_thread(_call)

    # Extract the first text segment safely
    out = []
    for item in getattr(resp, "output", []) or []:
        if item["type"] == "message":
            for c in item["content"]:
                if c["type"] == "output_text":
                    out.append(c["text"])
    text = "".join(out).strip()
    return text or "🤖 I couldn’t generate a reply."

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

    # Ensure command processing still works
    await bot.process_commands(message)

    # Only respond when bot is mentioned
    if bot.user not in message.mentions:
        return

    # Per-channel rolling history (safe on first use)
    hist = channel_history.setdefault(message.channel.id, collections.deque(maxlen=20))

    async with message.channel.typing():
        try:
            # Build messages for the model
            persona_text = get_active_persona_text(message.guild.id if message.guild else 0)
            msgs = [{"role": "system", "content": persona_text}]
            prior = list(hist)
            if prior:
                msgs.extend(prior)

            # If replying to another message, include that as CONTEXT ONLY (do NOT replace user's new message)
            if message.reference:
                try:
                    ref = await message.channel.fetch_message(message.reference.message_id)
                    ref_role = "assistant" if ref.author.id == bot.user.id else "user"
                    ref_text = ref.clean_content.strip()
                    if ref_text:
                        msgs.append({"role": ref_role, "content": f"(context you replied to) {ref_text}"})
                except Exception:
                    pass

            # --- Attachment handling (from the CURRENT message only) ---
            if message.attachments:
                attachment = message.attachments[0]
                filename = attachment.filename.lower()

                if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    image_bytes = await attachment.read()
                    b64 = encode_image_to_base64(image_bytes)
                    user_content = [
                        {"type": "text", "text": "Analyze the visual content in a methodical, clinical way. Describe subjects, behavior, surroundings, and time of day as if for evidence."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]
                    hist.append({"role": "user", "content": "[User sent an image for analysis]"})
                    msgs.append({"role": "user", "content": user_content})

                    result = await _chat41(msgs, max_tokens=2000, temperature=0.6)
                    result = (result or "").strip()
                    if result:
                        hist.append({"role": "assistant", "content": result})
                    return await send_in_chunks(message.channel, result, reply_to=message)

                elif filename.endswith((".mp4", ".mov", ".webm", ".gif")):
                    path = await download_media(attachment)
                    frames = await asyncio.to_thread(extract_frames, path, 5)
                    if not frames:
                        return await message.reply("❌ Could not extract frames from the video. Check file integrity.")
                    user_content = [{"type": "text", "text": "Video sampled at ~1 FPS. Analyze clinically like evidence."}]
                    for frame in frames:
                        user_content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(frame)}"}})
                    hist.append({"role": "user", "content": f"[User sent a video: {filename}]"})
                    msgs.append({"role": "user", "content": user_content})

                    result = await _chat41(msgs, max_tokens=2000, temperature=0.6)
                    result = (result or "").strip()
                    if result:
                        hist.append({"role": "assistant", "content": result})
                    return await send_in_chunks(message.channel, result, reply_to=message)

                else:
                    return await message.reply("⚠️ Unsupported media type. I can analyze images and videos only.")

            # --- Plain text path ---
            # Use the CURRENT message as the user's input, not the referenced one
            user_text = message.clean_content
            # Remove the bot mention anywhere in the text (not just at the start)
            user_text = re.sub(rf"<@!?{bot.user.id}>", "", user_text).strip()

            # If the user only mentioned the bot and nothing else, nudge the model
            if not user_text:
                user_text = "Please respond to the message I replied to."

            hist.append({"role": "user", "content": user_text})
            msgs.append({"role": "user", "content": user_text})

            # Use the same model helper everywhere (or implement _chat5 properly)
            result = await _chat41(msgs, max_tokens=2000, temperature=0.7)
            result = (result or "").strip()
            if result:
                hist.append({"role": "assistant", "content": result})

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
    ffmpeg_bin = os.getenv("FFMPEG_BIN")  # e.g., C:/ffmpeg/bin/ffmpeg.exe
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
# Play youtube links
# ---------- Constants ----------
FFMPEG_BEFORE_OPTS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
FFMPEG_OPTIONS = "-vn"

YTDLP_OPTS = {
    # pick best audio; prefer m4a/webm but allow HLS/dash (FFmpeg handles it)
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "default_search": "auto",
    "skip_download": True,
    "nocheckcertificate": True,
    # CRITICAL: force web client to avoid Android GVS PO token warning
    "extractor_args": {"youtube": {"player_client": ["android"]}},
    "cachedir": False,
}

# ---------- Paths / Source ----------
def get_ffmpeg_path():
    # 1) env var
    env = os.getenv("FFMPEG_BIN")
    if env and shutil.which(env):
        return env

    # 2) which on PATH (best cross-platform check)
    wh = shutil.which("ffmpeg")
    if wh:
        return wh

    # 3) OS fallbacks
    system = platform.system().lower()
    candidates = []
    if system == "windows":
        candidates += [
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files (x86)\ffmpeg\bin\ffmpeg.exe",
        ]
    else:
        candidates += ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]

    for c in candidates:
        if c and shutil.which(c):
            return c

    raise RuntimeError(
        "FFmpeg not found. Install it (e.g., `sudo apt install ffmpeg libopus0`) "
        "or set FFMPEG_BIN to the full path."
    )

def make_audio_source(url: str) -> discord.FFmpegPCMAudio:
    ffmpeg_path = get_ffmpeg_path()
    return discord.FFmpegPCMAudio(
        url,
        executable=ffmpeg_path,
        before_options=FFMPEG_BEFORE_OPTS,
        options=FFMPEG_OPTIONS,
    )

def get_stream(url_or_query: str):
    """Return (title, stream_url, webpage_url)."""
    with yt_dlp.YoutubeDL(YTDLP_OPTS) as ydl:
        info = ydl.extract_info(url_or_query, download=False)
        if not info:
            raise RuntimeError("No extractable info.")

        if "entries" in info and info["entries"]:
            info = info["entries"][0]

        title = info.get("title") or "Unknown Title"
        stream_url = info.get("url")  # may be HLS or direct; FFmpeg can handle both
        webpage_url = info.get("webpage_url") or url_or_query

        if not stream_url:
            raise RuntimeError("No playable stream URL from yt-dlp.")
        return title, stream_url, webpage_url

# ---------- Simple in-memory queue ----------
from collections import defaultdict, deque
_guild_queues = defaultdict(deque)

def _get_queue(guild_id: int) -> deque:
    return _guild_queues[guild_id]

async def _play_next_in_queue(ctx):
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        return

    q = _get_queue(ctx.guild.id)
    if not q:
        return

    title, stream_url, page = q[0]  # peek; pop when playback actually starts

    try:
        source = make_audio_source(stream_url)
    except Exception as e:
        q.popleft()
        await ctx.send(f"❌ FFmpeg problem: {e}")
        return await _play_next_in_queue(ctx)

    # Wrap in volume transformer so /volume works
    audio = discord.PCMVolumeTransformer(source, volume=0.9)

    def _after_playback(err):
        # Always advance the queue; report errors in channel thread
        try:
            q.popleft()
        except Exception:
            pass
        # schedule next on the event loop
        fut = asyncio.run_coroutine_threadsafe(_play_next_in_queue(ctx), vc.loop)
        try:
            fut.result()
        except Exception:
            pass

    try:
        vc.play(audio, after=_after_playback)
        await ctx.send(f"🎶 Now playing: **{title}**")
    except Exception as e:
        # If play() fails, drop this item and move on
        q.popleft()
        await ctx.send(f"❌ Could not start playback: {e}")
        await _play_next_in_queue(ctx)

# ---------- Commands ----------
@bot.command(name="playyt")
async def playyt(ctx, *, url: str):
    """Play/queue audio from YouTube (or any yt-dlp source)."""
    # Join/move to user's VC
    vc = ctx.voice_client
    if not vc:
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("🔊 Join a voice channel first.")
        vc = await ctx.author.voice.channel.connect()
    elif ctx.author.voice and vc.channel != ctx.author.voice.channel:
        await vc.move_to(ctx.author.voice.channel)

    # Extract stream
    try:
        title, stream_url, page = get_stream(url)
    except Exception as e:
        return await ctx.send(f"❌ yt-dlp error: {e}")

    q = _get_queue(ctx.guild.id)
    q.append((title, stream_url, page))

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

#_______________________________Custom Persona___________________________________________________

# Maps guild_id -> active persona key (e.g., "default", "custom", "shabbot", etc.)
ACTIVE_PERSONA_KEY: Dict[int, str] = {}

# Maps guild_id -> the freeform text for the "custom" persona
CUSTOM_PERSONA_TEXT: Dict[int, str] = {}

def get_active_persona_text(guild_id: int) -> str:
    """
    Returns the persona text currently active for this guild.
    If "custom" is selected but no text exists, fall back to default.
    """
    key = ACTIVE_PERSONA_KEY.get(guild_id, "default")
    if key == "custom":
        return CUSTOM_PERSONA_TEXT.get(guild_id) or PERSONAS.get("default", "")
    return PERSONAS.get(key, PERSONAS.get("default", ""))


# -------------------------------
# Commands
# -------------------------------

@bot.command(name="custompersona", help="Set a server-wide custom persona. Usage: !custompersona <descriptor>")
async def custompersona(ctx, *, descriptor: str = None):
    """
    Sets a server-wide (guild) custom persona description and switches the active persona to 'custom'.
    Rejects empty/blank input.
    """
    # Reject DMs if you only want this to be server-wide. If you want DMs too, remove this block.
    if ctx.guild is None:
        return await ctx.send("This command can only be used in a server.")

    if not descriptor or not descriptor.strip():
        return await ctx.send("Usage: `!custompersona <descriptor>` — describe the voice/persona you want.")

    guild_id = ctx.guild.id
    CUSTOM_PERSONA_TEXT[guild_id] = descriptor.strip()
    ACTIVE_PERSONA_KEY[guild_id] = "custom"

    await ctx.send("🧠 Custom persona set for this server. Active persona: **custom**.")


@bot.command(name="resetpersona", help="Reset the server-wide persona back to default.")
async def resetpersona(ctx):
    """
    Resets the active persona to 'default' and clears any custom persona text for this guild.
    """
    if ctx.guild is None:
        return await ctx.send("This command can only be used in a server.")

    guild_id = ctx.guild.id
    ACTIVE_PERSONA_KEY[guild_id] = "default"
    CUSTOM_PERSONA_TEXT.pop(guild_id, None)

    await ctx.send("↩️ Persona reset to **default** for this server.")


# ------------------------------- Clear messages ---------------------------------

# Replace with your restricted role ID
ALLOWED_ROLE_ID = 1051206094506172497  

@bot.command(name="clear")
@commands.has_role(ALLOWED_ROLE_ID)  # Restricts command to users with that role
async def clear(ctx, amount: int):
    """Deletes X number of messages from the channel."""
    if amount < 1:
        return await ctx.send("⚠️ You must delete at least 1 message.")

    # Bulk delete messages
    deleted = await ctx.channel.purge(limit=amount + 1)  # +1 to also delete the command message
    await ctx.send(f"🗑️ Deleted {len(deleted)-1} messages.", delete_after=5)

# Error handler if user doesn’t have the role
@clear.error
async def clear_error(ctx, error):
    if isinstance(error, commands.MissingRole):
        await ctx.send("❌ You don’t have permission to use this command.")

# Entry point
if __name__ == "__main__":
    bot.run(os.getenv("DISCORD_BOT_TOKEN"))
