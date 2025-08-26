import os
import discord
import logging
import asyncio
import subprocess
import glob
import base64
import aiohttp
from dotenv import load_dotenv
from collections import deque, defaultdict
from openai import OpenAI
from discord.ext import commands
from discord.errors import NotFound
from discord.sinks import WaveSink
from discord.sinks.errors import RecordingException
from io import BytesIO
from discord.sinks.errors import SinkException
from collections import deque, defaultdict
from yt_dlp import YoutubeDL
from collections import deque
from zoneinfo import ZoneInfo
#logging.basicConfig(level=logging.INFO)
#logging.getLogger('discord').setLevel(logging.DEBUG)

load_dotenv()
# --- Configuration ---
conversation_memory = defaultdict(lambda: deque(maxlen=6))  # per-user short memory

intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # required for role add/remove
intents.voice_states = True
bot = commands.Bot(command_prefix="!", intents=intents)  # or commands.Bot if you prefer
TTS_VOICE = 'onyx'   # try "verse" or "ash" for a lower, more neutral tone
DISCORD_LIMIT = 2000
CHUNK_TARGET = 1900  # leave room for formatting/mentions
base_dir = os.path.expanduser("~")
temp_dir = os.path.join(base_dir, "Personal Use", "Powershell", "Shabbot", "TEMP")
CHANNEL_HISTORY_MAX = 20  # total messages kept per channel (user+assistant)
channel_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=CHANNEL_HISTORY_MAX))
# ---- Config ----
TZ = ZoneInfo("America/New_York")   # change if you want a different timezone
CLEAR_HOUR = 3                   # 4 AM
CLEAR_MINUTE = 30

intents = discord.Intents.default()
intents.guilds = True
intents.members = True
intents.voice_states = True


# Per-guild song queues
guild_queues: dict[int, deque] = {}

YTDLP_OPTS = {
    "format": "bestaudio/best",
    "noplaylist": True,
    "quiet": True,
    "default_search": "auto",
    "skip_download": True,
    "nocheckcertificate": True,
}

FFMPEG_BEFORE_OPTS = "-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5"
FFMPEG_OPTIONS = "-vn"

os.makedirs(temp_dir, exist_ok=True)

class SafeWaveSink(WaveSink):
    def write(self, data, user):
        try:
            return super().write(data, user)
        except SinkException:
            # Late frames after we're done; ignore instead of crashing the decode thread
            return

# Personas
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

# OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("WARNING: OPENAI_API_KEY not set in environment.")
client = OpenAI(api_key=api_key)

# Utility functions
def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")

def clean_temp_frames():
    for file in glob.glob(os.path.join(temp_dir, "frame_*.jpg")):
        try:
            os.remove(file)
        except Exception:
            pass

def extract_frames(video_path: str, max_frames: int = 10) -> list[bytes]:
    clean_temp_frames()
    output_pattern = os.path.join(temp_dir, "frame_%03d.jpg")
    # Extract at 1 FPS, you can adjust or parameterize as needed
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",  # overwrite
                "-i", video_path,
                "-vf",
                "fps=1",
                output_pattern,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except subprocess.CalledProcessError:
        return []

    frames = []
    for file in sorted(glob.glob(os.path.join(temp_dir, "frame_*.jpg"))):
        with open(file, "rb") as f:
            frames.append(f.read())
        if len(frames) >= max_frames:
            break
    return frames

async def download_media(attachment) -> str:
    filepath = os.path.join(temp_dir, attachment.filename)
    async with aiohttp.ClientSession() as session:
        async with session.get(attachment.url) as resp:
            if resp.status == 200:
                with open(filepath, "wb") as f:
                    f.write(await resp.read())
            else:
                raise IOError(f"Failed to download media: HTTP {resp.status}")
    return filepath

# --- Commands & Events ---
@bot.event
async def on_ready():
    print(f"Logged in as {bot.user}!")

@bot.command()
async def ping(ctx):
    await ctx.send("Pong!")

@bot.command()
async def say(ctx, *, message: str):
    await ctx.send(message)

async def ensure_voice(ctx) -> discord.VoiceClient | None:
    """Join the author's VC or return existing connection."""
    if ctx.voice_client:
        return ctx.voice_client
    if not ctx.author.voice:
        await ctx.send("🔊 You need to be in a voice channel first.")
        return None
    return await ctx.author.voice.channel.connect()

@bot.command()
async def voice(ctx, name: str = None):
    """Set or show the current TTS voice."""
    global TTS_VOICE
    available = ["alloy","ash","ballad","coral","echo","sage","shimmer","verse"]
    if not name:
        return await ctx.send(f"🎙️ Current voice: `{TTS_VOICE}`\nOptions: {', '.join(available)}")
    name = name.lower()
    if name not in available:
        return await ctx.send(f"❌ Unknown voice. Try: {', '.join(available)}")
    TTS_VOICE = name
    await ctx.send(f"✅ Voice set to `{TTS_VOICE}`")

@bot.command()
async def join(ctx):
    # 1️⃣ Ensure the user is in voice
    if not ctx.author.voice or not ctx.author.voice.channel:
        return await ctx.send("🔊 You need to be in a voice channel first.")

    target = ctx.author.voice.channel

    # 2️⃣ If we're already connected in this guild, report channel
    if vc := ctx.guild.voice_client:
        if vc.is_connected():
            return await ctx.send(f"✅ Already connected to **{vc.channel.name}**")
        else:
            # weird state—disconnect and try fresh
            await vc.disconnect()
            logging.warning(f"Found stale VoiceClient in guild {ctx.guild.id}, disconnected.")

    # 3️⃣ Try to connect with timeout
    try:
        vc = await target.connect(timeout=20)
    except asyncio.TimeoutError:
        return await ctx.send("⏱ Connection timed out.")
    except discord.ClientException as e:
        return await ctx.send(f"🚨 Client error on connect: {e.__class__.__name__}: {e}")
    except Exception as e:
        return await ctx.send(f"❌ Unexpected error: {e.__class__.__name__}: {e}")

    # 4️⃣ Double-check connection
    if not vc or not vc.is_connected() or vc.channel != target:
        # Log full state for debugging
        logging.error(f"Connect reported success but VoiceClient state is bad: {vc!r}")
        return await ctx.send("❌ I tried to join, but the connection didn’t stick. Check permissions and region.")

    # 5️⃣ Success!
    await ctx.send(f"✅ Connected to **{target.name}**")

@bot.command()
async def leave(ctx):
    """Disconnect bot from voice channel."""
    vc = ctx.voice_client
    if vc:
        await vc.disconnect()
        await ctx.send("👋 Disconnected.")
    else:
        await ctx.send("🔇 I'm not connected.")

@bot.command(name="equipmentchecksoundoff", aliases=["eqcso"])
async def equipmentchecksoundoff(ctx, *, descriptor: str = None):
    # Reuse equipmentcheck logic to generate text only (no sending yet)
    system_prompt = (
        "You are Shabbot, a tactical squad AI trained for both military-style ops "
        "and recreational readiness checks (wink). Your job is to write gritty, motivational, "
        "call-to-action that match the tone of elite unit operations — but with a subtle 'nod' to 420-friendly equipment prep. Address the squad with confidence. Mention Soldier OR whatever else seems appropriate based upon {descriptor}. "
        "Keep it short (3–5 lines), high energy, and never mention it's AI-generated."
    )
    if descriptor:
        user_prompt = f"Write a fresh, high-intensity mission-style Equipment Check announcement in the style of {descriptor}."
    else:
        user_prompt = "Write a fresh, high-intensity mission-style Equipment Check announcement."

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2000,
        )
        skit = response.choices[0].message.content.strip()
    except Exception:
        skit = "**EQUIPMENT CHECK – COMMAND FAILED**\nFallback briefing activated. Check your kit manually."

    # Send to text chat
    await ctx.send(skit)

    # Now handle voice channel TTS
    if not ctx.author.voice:
        return await ctx.send("🔊 You need to be in a voice channel for me to sound off!")
    vc: discord.VoiceClient = ctx.voice_client or await ctx.author.voice.channel.connect()

    # Generate TTS audio
    tts_response = client.audio.speech.create(
        model="tts-1",
        voice=TTS_VOICE,
        input=skit
    )
    audio_bytes: bytes = tts_response.read()

    out_path = os.path.join(temp_dir, f"eqcso_{ctx.author.id}.mp3")
    with open(out_path, "wb") as f:
        f.write(audio_bytes)

    if vc.is_playing():
        vc.stop()
    vc.play(discord.FFmpegPCMAudio(out_path))

    def _cleanup(_err):
        try:
            os.remove(out_path)
        except:
            pass
    vc.source.after = _cleanup

    await ctx.send("🗣️ Sounding off equipment check in VC!")

@bot.command()
async def play(ctx, sound: str):
    if ctx.author.voice is None:
        return await ctx.send("❌ You must be in a voice channel.")

    channel = ctx.author.voice.channel
    vc = ctx.voice_client

    if vc is None:
        vc = await channel.connect()
    elif vc.channel != channel:
        await vc.move_to(channel)

    try:
        source = discord.FFmpegPCMAudio(f"sounds/{sound}.mp3")
        vc.play(source)
    except Exception as e:
        await ctx.send(f"❌ Could not play sound: {e}")

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
    
# ─── SPEECH-TO-SPEECH COMMAND ─────────────────────────────────
# an async no-op to satisfy the recorder's expectation
# async def _noop_record_finish(sink: WaveSink, ctx):
#     return

@bot.command()
async def s2s(ctx, record_seconds: int = 5):
    # Ensure we’re in a VC
    vc = ctx.voice_client
    if not vc:
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("🔊 Join a voice channel first.")
        vc = await ctx.author.voice.channel.connect()

    # Fresh sink per run
    sink = SafeWaveSink()

    # Signal when cleanup (WAV header finalize) is done
    finished = asyncio.Event()

    async def _on_finish(sink_obj, ctx_obj):
        finished.set()

    await ctx.send(f"⏺ Recording for {record_seconds}s…")
    vc.start_recording(sink, _on_finish, ctx)
    await asyncio.sleep(record_seconds)

    # Stop and wait for cleanup
    try:
        vc.stop_recording()
    except Exception:
        return await ctx.send("❌ I wasn’t recording—did starting fail earlier?")

    await finished.wait()  # ensures WAV headers are written; safe to read now

    # Debug info
    print("SINK KEYS:", sink.audio_data.keys())
    for user_id, audio_data in sink.audio_data.items():
        size = len(audio_data.file.getvalue())  # copy, not memoryview
        print(f" • User {user_id!r}: {size} bytes captured")

    # Take the first user’s audio
    _, audio_data = next(iter(sink.audio_data.items()))
    wav_bytes = audio_data.file.getvalue()

    audio_buffer = BytesIO(wav_bytes)
    audio_buffer.name = "recording.wav"

    # Transcribe (thread off the blocking call)
    resp = await asyncio.to_thread(
        client.audio.transcriptions.create,
        model="whisper-1",
        file=audio_buffer,
    )
    text_in = (resp.text or "").strip()
    await ctx.send(f"📝 You said: “{text_in}”" if text_in else "📝 (no speech detected)")

    # Get a chat reply — make sure there’s no stray '+' in front of this call
    reply = await _chat5([{"role": "system", "content": active_persona["system_prompt"]},{"role": "user", "content": text_in or "(No speech detected.)"},],
    max_completion_tokens=400,)

    # TTS to file and play in VC
    tts = await asyncio.to_thread(
        client.audio.speech.create,
        model="tts-1",
        voice=TTS_VOICE,
        input=reply or "I didn't catch that.",
    )
    out_path = os.path.join(temp_dir, f"out_{ctx.author.id}.mp3")
    with open(out_path, "wb") as f:
        f.write(tts.read())

    if vc.is_playing():
        vc.stop()
    vc.play(discord.FFmpegPCMAudio(out_path))

    def _cleanup(_err):
        try: os.remove(out_path)
        except: pass
    vc.source.after = _cleanup

@bot.command()
async def speak(ctx, *, text: str):
    """Join your VC (if needed) and speak whatever `text` you provide."""
    # 1) Ensure the user is in a VC
    if not ctx.author.voice:
        return await ctx.send("🔊 You need to be in a voice channel for me to speak!")
    # 2) Connect or reuse existing
    vc: discord.VoiceClient = ctx.voice_client or await ctx.author.voice.channel.connect()

    # 3) Generate the TTS audio (returns HttpxBinaryResponseContent)
    tts_response = client.audio.speech.create(
        model="tts-1",
        voice=TTS_VOICE,
        input=text
    )

    # 4) Extract raw bytes
    audio_bytes: bytes = tts_response.read()

    # 5) Write out and play
    with open("speak.mp3", "wb") as f:
        f.write(audio_bytes)
    vc.play(discord.FFmpegPCMAudio("speak.mp3"))

    # 6) Confirm in chat
    await ctx.send(f"🗣️ Speaking: “{text}”")

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
    if descriptor:
        user_prompt = f"Write a fresh, high-intensity mission-style Equipment Check announcement in the style of {descriptor}."
    else:
        user_prompt = "Write a fresh, high-intensity mission-style Equipment Check announcement."

    try:
        response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=2000,
        )
        skit = response.choices[0].message.content.strip()
    except Exception as e:
        skit = "**EQUIPMENT CHECK – COMMAND FAILED**\nFallback briefing activated. Check your kit manually and await further instructions."

    countdown = 5 * 60
    timer_text = f"⏱ Time remaining: 05:00"
    full_message = f"{skit}\n\n{timer_text}"
    sent_msg = await ctx.send(full_message)
    await sent_msg.add_reaction("✅")

    confirmed_users: set[discord.Member] = set()

    def reaction_check(reaction, user):
        return (
            reaction.message.id == sent_msg.id
            and str(reaction.emoji) == "✅"
            and not user.bot
        )

    async def countdown_timer():
        for remaining in range(countdown - 1, -1, -1):
            minutes = remaining // 60
            seconds = remaining % 60
            timer_text = f"⏱ Time remaining: {minutes:02}:{seconds:02}"
            await asyncio.sleep(1)

            try:
                await sent_msg.edit(content=f"{skit}\n\n{timer_text}")
            except NotFound:
                # message was deleted or no longer exists → stop updating
                return

        # final edit when time's up
        try:
            await sent_msg.edit(content=f"{skit}\n\n⏱ Time's up. Lock in or fall out.")
        except NotFound:
            return

    async def gather_reactions():
        while not countdown_task.done():
            try:
                reaction, user = await bot.wait_for("reaction_add", timeout=1.0, check=reaction_check)
                if isinstance(user, discord.Member):
                    confirmed_users.add(user)
            except asyncio.TimeoutError:
                continue

    countdown_task = asyncio.create_task(countdown_timer())
    await gather_reactions()  # this will spin until countdown finishes

    # Role logic
    guild = ctx.guild
    if not guild:
        await ctx.send("⚠️ Cannot apply roles in DMs.")
        return

    ready_role = discord.utils.get(guild.roles, name="Equipped")
    if not ready_role:
        await ctx.send("⚠️ 'Equipped' role not found.")
        return

    for member in guild.members:
        if member.bot:
            continue
        if member in confirmed_users:
            if ready_role not in member.roles:
                try:
                    await member.add_roles(ready_role)
                except Exception:
                    pass
        else:
            if ready_role in member.roles:
                try:
                    await member.remove_roles(ready_role)
                except Exception:
                    pass

    if confirmed_users:
        mentions = ", ".join(u.mention for u in confirmed_users)
        await ctx.send(f"✅ Gear confirmed for: {mentions}")
    else:
        await ctx.send("❌ No check-ins. The squad went dark.")

def _safe_text_from_chat(resp, fallback="🤖 I couldn’t generate a reply."):
    try:
        msg = resp.choices[0].message
        text = (getattr(msg, "content", None) or "").strip()
        if not text and hasattr(msg, "refusal") and msg.refusal:
            text = str(msg.refusal).strip()
        return text if text else fallback
    except Exception:
        return fallback

async def _chat41(messages, *, max_tokens=2000, temperature=0.7):
    """
    Call gpt-4.1 for multimodal (images/videos). Supports {"type":"image_url"}.
    """
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-4.1",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return (resp.choices[0].message.content or "").strip() or "🤖 I couldn’t generate a reply."

async def _chat5(messages, *, max_completion_tokens=600):
    resp = await asyncio.to_thread(
        client.chat.completions.create,
        model="gpt-5",
        messages=messages,
        max_completion_tokens=max_completion_tokens,
    )
    text = _safe_text_from_chat(resp)
    if text.strip() == "🤖 I couldn’t generate a reply.":  # one retry on mini
        resp2 = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-5-mini",
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        text = _safe_text_from_chat(resp2)
    return text

async def chat5_complete(messages, *, max_completion_tokens=1200, max_rounds=2):
    full = ""
    rounds = 0
    msgs = messages[:]
    while True:
        resp = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-5",
            messages=msgs,
            max_completion_tokens=max_completion_tokens,
        )
        choice = resp.choices[0]
        text = (getattr(choice.message, "content", None) or "").strip()
        full += (("\n" if full else "") + (text or ""))
        if getattr(choice, "finish_reason", "stop") != "length" or rounds >= max_rounds:
            break
        rounds += 1
        msgs += [{"role": "assistant", "content": text or ""}, {"role": "user", "content": "Continue."}]
    if not full.strip():
        # fallback once
        resp2 = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-5-mini",
            messages=messages,
            max_completion_tokens=max_completion_tokens,
        )
        full = (resp2.choices[0].message.content or "🤖 I couldn’t generate a reply.").strip()
    return full

@bot.command()
async def forget(ctx):
    conversation_memory[str(ctx.author.id)].clear()
    await ctx.send("🧠 Shabbot has cleared your conversation history.")

def _split_chunks(text: str, max_len: int = CHUNK_TARGET):
    """Split text on paragraphs/sentences, then hard-wrap if needed."""
    if not text:
        return []
    chunks, buf = [], ""
    import re
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
    """Send long text as multiple messages without mid-sentence cuts."""
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

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    # Let commands run first
    await bot.process_commands(message)

    # Only respond when the bot is mentioned
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

                # Build a base message list with persona + prior channel memory
                msgs = [{"role": "system", "content": active_persona["system_prompt"]}]
                prior = list(channel_history.get(message.channel.id, []))
                if prior:
                    msgs.extend(prior)

                # If an attachment exists, prepare multimodal request (→ gpt-4.1)
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
                        # Memory note (don’t store base64)
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
                        # Memory note
                        channel_history[message.channel.id].append({"role": "user", "content": f"[User sent a video: {filename}]"})
                        msgs.append({"role": "user", "content": user_content})

                    else:
                        return await message.reply("⚠️ Unsupported media type. I can analyze images and videos only.")

                    # ⬇️ Use gpt-4.1 for multimodal
                    result = await _chat41(msgs, max_tokens=2000, temperature=0.6)

                    # Save assistant reply to memory
                    if result:
                        channel_history[message.channel.id].append({"role": "assistant", "content": result})

                    return await send_in_chunks(message.channel, result, reply_to=message)

                # No attachment → normal text reply (→ gpt-5)
                user_text = message.content
                channel_history[message.channel.id].append({"role": "user", "content": user_text})
                msgs.append({"role": "user", "content": user_text})

                # ⬇️ Use gpt-5 (text-only). No temperature; use max_completion_tokens.
                result = await _chat5(msgs, max_completion_tokens=2000)

                if result:
                    channel_history[message.channel.id].append({"role": "assistant", "content": result})

                await send_in_chunks(message.channel, result, reply_to=message)

            except Exception as e:
                await message.reply(f"❌ Failed to generate reply: {e}")

@bot.command(name="resetmemory")
async def resetmemory(ctx):
    """Clear conversation memory for this channel."""
    try:
        channel_history.pop(ctx.channel.id, None)
        await ctx.send("🧹 Memory cleared for this channel.")
    except Exception:
        await ctx.send("🧹 (Tried to clear memory, but there was nothing saved.)")

@bot.command(name="stop")
async def stop(ctx):
    """Force Shabbot to stop talking and reset his listening loop."""
    gid = ctx.guild.id if ctx.guild else 0

    # 1) Kill active VC playback
    if ctx.voice_client and ctx.voice_client.is_playing():
        ctx.voice_client.stop()

    # # 2) Cancel any always-on S2S loop
    # task = s2s_tasks.get(gid)
    # if task and not task.done():
    #     task.cancel()
    #     s2s_tasks.pop(gid, None)

    # 3) Confirm reset
    await ctx.send("🛑 Shabbot has gone silent and reset his listening.")

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

    # build FFmpeg audio source from direct media URL
    # (url here is already the 'direct' audio URL from yt-dlp)
    source = discord.FFmpegPCMAudio(
        url,
        before_options=FFMPEG_BEFORE_OPTS,
        options=FFMPEG_OPTIONS,
    )

    def _after_playback(error):
        # schedule the next track on the main loop
        try:
            fut = asyncio.run_coroutine_threadsafe(_play_next_in_queue(ctx), bot.loop)
            fut.result()
        except Exception:
            pass

    vc.play(source, after=_after_playback)
    try:
        await ctx.send(f"▶️ **Now playing:** {title}\n🔗 {webpage_url}")
    except Exception:
        pass

@bot.command(name="playyt")
async def playyt(ctx, *, url: str):
    """Play audio from a YouTube (or yt-dlp supported) URL. Queues if already playing."""
    # join or reuse VC
    vc = ctx.voice_client
    if not vc:
        if not ctx.author.voice or not ctx.author.voice.channel:
            return await ctx.send("🔊 Join a voice channel first.")
        vc = await ctx.author.voice.channel.connect()
    elif ctx.author.voice and vc.channel != ctx.author.voice.channel:
        await vc.move_to(ctx.author.voice.channel)

    # extract bestaudio URL + title using yt-dlp
    try:
        with YoutubeDL(YTDLP_OPTS) as ydl:
            info = ydl.extract_info(url, download=False)
            if "entries" in info:  # playlist or search
                info = info["entries"][0]
            title = info.get("title", "Unknown Title")
            webpage_url = info.get("webpage_url") or url
            # Try m4a/opus direct URL in order of preference
            direct_url = (
                info.get("url")
                or (info.get("formats") or [{}])[-1].get("url")
            )
            if not direct_url:
                return await ctx.send("❌ Could not get a playable audio URL.")
    except Exception as e:
        return await ctx.send(f"❌ yt-dlp error: {e}")

    q = _get_queue(ctx.guild.id)
    q.append((title, direct_url, webpage_url))

    # if nothing is playing, start immediately
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
    else:
        await ctx.send("ℹ️ Nothing is paused.")

@bot.command()
async def skip(ctx):
    vc = ctx.voice_client
    if not vc or not vc.is_connected():
        return await ctx.send("🔇 I'm not connected.")
    if vc.is_playing() or vc.is_paused():
        vc.stop()  # this triggers the 'after' callback to play next
        await ctx.send("⏭️ Skipped.")
    else:
        await ctx.send("ℹ️ Nothing to skip.")

import asyncio
import datetime as dt
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands

async def clear_all_vcs(reason: str = "Scheduled 4 AM VC clear"):
    """Disconnect every member from every voice/stage channel in all guilds."""
    # Iterate all guilds the bot is in
    for guild in bot.guilds:
        # Gather both VoiceChannel and StageChannel
        voice_channels = list(getattr(guild, "voice_channels", []))
        stage_channels = list(getattr(guild, "stage_channels", []))
        channels = voice_channels + stage_channels

        for vc in channels:
            # Check permissions before attempting to move members
            me = guild.me
            if me is None:
                continue
            perms = vc.permissions_for(me)
            if not perms.move_members:
                # Optional: log missing perms
                print(f"[WARN] Missing Move Members for #{vc} in {guild.name}")
                continue

            # Disconnect everyone in the channel (including the bot, if present)
            # vc.members includes listeners on StageChannel, too.
            for member in list(vc.members):
                try:
                    await member.move_to(None, reason=reason)
                    # Small delay to be gentle with rate limits
                    await asyncio.sleep(0.25)
                except discord.Forbidden:
                    print(f"[WARN] Forbidden moving {member} in #{vc} ({guild.name})")
                except discord.HTTPException as e:
                    print(f"[WARN] HTTPException moving {member}: {e}")

        # If the bot is connected via a VoiceClient, ensure it’s disconnected
        vc_client = guild.voice_client
        if vc_client and vc_client.is_connected():
            try:
                await vc_client.disconnect(force=True)
            except discord.HTTPException as e:
                print(f"[WARN] Could not disconnect bot VC in {guild.name}: {e}")

async def scheduler():
    """Sleep-until 4:00 AM ET every day, then run clear_all_vcs()."""
    await bot.wait_until_ready()
    while not bot.is_closed():
        now = dt.datetime.now(TZ)
        target = now.replace(hour=CLEAR_HOUR, minute=CLEAR_MINUTE, second=0, microsecond=0)
        if now >= target:
            target += dt.timedelta(days=1)
        wait_seconds = (target - now).total_seconds()
        print(f"[INFO] Next VC clear at {target.isoformat()} ({int(wait_seconds)}s)")
        await asyncio.sleep(wait_seconds)
        await clear_all_vcs()

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} ({bot.user.id})")
    # Start the daily scheduler task once
    bot.loop.create_task(scheduler())

# Optional: manual slash-style command (works with commands extension)
@bot.command(name="clear_all_vcs", help="Immediately disconnects everyone from all voice/stage channels.")
@commands.has_permissions(move_members=True)
async def _clear_all_vcs(ctx: commands.Context):
    await ctx.send("Clearing all voice channels…")
    await clear_all_vcs(reason=f"Manual run by {ctx.author}")
    await ctx.send("Done.")

# Run bot
bot.run(os.getenv("DISCORD_BOT_TOKEN"))
