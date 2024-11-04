import discord
from discord.ext import commands

class Events(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.Cog.listener()
    async def on_message(self, message):
        if message.author == self.bot.user:
            return
        
        #if message is command
        if message.content.startswith('!'):
            #await self.bot.process_commands(message)
            return

        if message.content == 'hi':
            await message.channel.send('Hello from event!')

async def setup(bot):
    await bot.add_cog(Events(bot))
