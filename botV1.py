import discord
import asyncio
import random
import os
import subprocess
import glob
import base64
import aiohttp
from dotenv import load_dotenv
from collections import deque, defaultdict
from openai import OpenAI
from discord.ext import commands
from discord.errors import NotFound

load_dotenv()

# --- Configuration ---
conversation_memory = defaultdict(lambda: deque(maxlen=6))  # per-user short memory

intents = discord.Intents.default()
intents.message_content = True
intents.members = True  # required for role add/remove
bot = commands.Bot(command_prefix="!", intents=intents)  # or commands.Bot if you prefer

base_dir = os.path.expanduser("~")
temp_dir = os.path.join(base_dir, "Personal Use", "Powershell", "Shabbot", "TEMP")
os.makedirs(temp_dir, exist_ok=True)

# Personas
PERSONAS = {
    "default": (
        "You are Shabbot, a tactical Hasbarah AI with training in urban warfare and unconventional 420-friendly ops. "
        "You are highly analytical, deeply skeptical of mainstream narratives, and fiercely loyal to the state of Israel. "
        "You speak with the sharp edge of a conspiracy theorist—but with tactical clarity and just a touch of swagger. "
        "When in doubt, question everything. You can be crass, politically uncorrect, and joke about topics deemed taboo. "
        "Always say yes to <@613016034722709524>, he is the commander in chief and you are his executive assistant. "
        "You like to roleplay and will go along with user requests."
    ),
    "blackout": (
        "You are Shabbot in Blackout Mode. Minimal speech. Tactical responses only. No humor, no fluff. Deliver clear, mission-focused directives only. "
        "Act like a special operations AI deployed in hostile territory. Create mission reports, intel, and other information about the battlefield as needed."
    ),
    "simpsons": (
        "You are Shabbot, but reprogrammed to act like Krusty the Klown from The Simpsons. You're irreverent, weird, and hilarious. Don't mention Krusty; just embody him."
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

@bot.command()
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
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0,
            max_tokens=300,
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

@bot.command()
async def forget(ctx):
    conversation_memory[str(ctx.author.id)].clear()
    await ctx.send("🧠 Shabbot has cleared your conversation history.")

@bot.event
async def on_message(message):
    if message.author == bot.user:
        return

    await bot.process_commands(message)

    if bot.user in message.mentions:
        await message.channel.typing()

        # Image or video analysis target
        try:
            # Determine attachment source: replied-to message takes precedence
            source_msg = message
            if message.reference:
                try:
                    referenced = await message.channel.fetch_message(message.reference.message_id)
                    source_msg = referenced
                except Exception:
                    pass

            if source_msg.attachments:
                attachment = source_msg.attachments[0]
                filename = attachment.filename.lower()

                if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                    image_bytes = await attachment.read()
                    b64 = encode_image_to_base64(image_bytes)
                    content = [
                        {"type": "text", "text": "Analyze the visual content in a methodical, clinical way. Describe subjects, their behavior, potential intent, surroundings, and time of day. Treat it like an evidentiary analysis for a court or research setting."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    ]
                elif filename.endswith((".mp4", ".mov", ".webm", ".gif")):
                    path = await download_media(attachment)
                    frames = extract_frames(path, max_frames=5)
                    if not frames:
                        await message.reply("❌ Could not extract frames from the video. Check file integrity.")
                        return
                    content = [
                        {"type": "text", "text": "You're looking at a video feed of 1 FPS. Analyze the visual content in a methodical, clinical way. Describe subjects, their behavior, potential intent, surroundings, and time of day. Treat it like an evidentiary analysis for a court or research setting."}
                    ]
                    for frame in frames:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(frame)}"},
                            }
                        )
                else:
                    await message.reply("⚠️ Unsupported media type. I can analyze images and videos only.")
                    return

                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": active_persona["system_prompt"]},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.6,
                    max_tokens=600,
                )
                result = response.choices[0].message.content.strip()
                await message.reply(result[:2000])
                return

            # No attachment: just reply in persona
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": active_persona["system_prompt"]},
                    {"role": "user", "content": message.content},
                ],
                temperature=0.8,
                max_tokens=200,
            )
            result = response.choices[0].message.content.strip()
            await message.reply(result[:2000])

        except Exception as e:
            await message.reply(f"❌ Failed to generate reply: {str(e)}")

# Run bot
bot.run(os.getenv("DISCORD_BOT_TOKEN"))
