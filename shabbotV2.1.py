import os
import discord
from discord import app_commands
TOKEN = os.getenv("DISCORD_BOT_TOKEN")

class ShabBot(discord.Client):
    def __init__(self):
        super().__init__(intents=discord.Intents.default())
        self.tree = app_commands.CommandTree(self)

    async def setup_hook(self) -> None:
        # Sync commands globally. For faster testing, provide a guild ID to sync to a single server.
        await self.tree.sync()
        print("Slash commands registered")

    async def on_ready(self) -> None:
        print(f"Logged in as {self.user} (ID: {self.user.id})")

    # Define a slash command called "ping"
    @app_commands.command(name="ping", description="Replies with pong")
    async def ping(self, interaction: discord.Interaction) -> None:
        await interaction.response.send_message("pong")

# Read token from environment variable
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
if not TOKEN:
    raise RuntimeError("Set DISCORD_BOT_TOKEN to your bot token")

client = ShabBot()
client.run(TOKEN)
