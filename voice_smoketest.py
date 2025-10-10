import os, asyncio, shutil
import discord
from contextlib import suppress

TOKEN = os.environ["DISCORD_TOKEN"]  # must be the long token with two dots
GUILD_ID = int(os.environ["GUILD_ID"])
VOICE_CHANNEL_ID = int(os.environ["VC_ID"])
AUDIO_FILE = "sounds/i-like-ya-and-i-want-ya.mp3"

def ensure_opus_loaded():
    if discord.opus.is_loaded():
        return
    for c in (
        "/lib/aarch64-linux-gnu/libopus.so.0",
        "/usr/lib/aarch64-linux-gnu/libopus.so.0",
        "libopus.so.0",
        "opus",
    ):
        with suppress(Exception):
            discord.opus.load_opus(c)
            if discord.opus.is_loaded():
                return
    raise RuntimeError("Opus not loaded")

intents = discord.Intents.default()
intents.voice_states = True

class Bot(discord.Client):
    async def on_ready(self):
        print("READY as", self.user)
        try:
            ensure_opus_loaded()
            print("opus loaded:", discord.opus.is_loaded())
        except Exception as e:
            print("OPUS LOAD FAIL:", e)
            await self.close(); return
        await self.run_test()

    async def run_test(self):
        ch = self.get_channel(VOICE_CHANNEL_ID)
        print("channel type:", type(ch).__name__, getattr(ch, "name", "?"))
        if not ch or not isinstance(ch, (discord.VoiceChannel, discord.StageChannel)):
            print("Bad VOICE_CHANNEL_ID"); await self.close(); return
        # connect
        try:
            vc = await ch.connect(timeout=20, reconnect=True)
            print("connected:", vc.is_connected(), "to", vc.channel.name)
        except discord.errors.ConnectionClosed as e:
            print("Connect WS closed code:", getattr(e, "code", None))
            await self.close(); return
        except Exception as e:
            print("Connect error:", type(e).__name__, e)
            await self.close(); return

        # Stage unsuppress if needed
        if isinstance(ch, discord.StageChannel):
            with suppress(Exception):
                await ch.request_to_speak()
                await ch.guild.me.edit(suppress=False)

        ff = os.getenv("FFMPEG_BIN") or shutil.which("ffmpeg") or "/usr/bin/ffmpeg"
        print("ffmpeg:", ff)
        if not (shutil.which(ff) or os.path.exists(ff)):
            print("FFMPEG missing"); await vc.disconnect(force=True); await self.close(); return
        if not os.path.isfile(AUDIO_FILE):
            print("Audio missing:", AUDIO_FILE); await vc.disconnect(force=True); await self.close(); return

        try:
            src = discord.FFmpegPCMAudio(AUDIO_FILE, executable=ff)
            vc.play(src)
            print("Playing…")
            for _ in range(200):
                await asyncio.sleep(0.1)
                if not vc.is_playing():
                    break
        except Exception as e:
            print("Play error:", type(e).__name__, e)
        finally:
            await vc.disconnect(force=True)
            await self.close()

client = Bot(intents=intents)
client.run(TOKEN)
