import discord
from discord.ext import commands
import io, aiohttp
from PIL import Image
from modules.detect_service import yolov8_service

class Commands(commands.Cog):
    def __init__(self, bot):
        self.bot = bot

    @commands.command()
    async def hello(self, ctx):
        await ctx.send('Hello from command!') 
        await self.bot.reload_extension("cogs.commands")  
          
    @commands.command('d')
    async def detect(self, ctx):
        #Check attachments
        if not ctx.message.attachments:         
            await ctx.send(embed=getMessageEmbed(f"Please upload an image."))
            return
        
        all_detect_result_stream: list[io.BytesIO] = []
        for item in ctx.message.attachments:
            if not isImageFile(item.filename):
                continue
            await ctx.send(embed=getMessageEmbed(f"{item.filename}: Detecting"))
            
            #Read image data
            img_data = await item.read()
            
            #Get PIL image
            try:                      
                image = Image.open(io.BytesIO(img_data))
                if isPNGFile(item.filename):
                    image = image.convert('RGB') #if there are 4 channels(RGBA) covert to 3 channels(RGB)
            except:    
                await ctx.send(embed=getMessageEmbed(f"{item.filename}: Get PIL error", "⛔"))    
                continue

            result_stream = yolov8_service().detect_object_info(image, item.filename)
            if result_stream is None:
                #await ctx.send(embed=getMessageEmbed(f"{item.filename}: Detected nothing", "😓"))
                continue  

            all_detect_result_stream.append(result_stream)

        if len(all_detect_result_stream) <= 0:
            await ctx.send(embed=getMessageEmbed(f"{item.filename}: Detected nothing", "😓"))
            return

        files = [discord.File(fp=stream, filename=f'detected_{i}.jpg') for i, stream in enumerate(all_detect_result_stream)]
        await ctx.send(embed=getMessageEmbed("Detected", "😎"), files=files)   

@discord.app_commands.context_menu(name="Detect Images")
async def detect_image_context(interaction: discord.Interaction, message: discord.Message):
    #Check attachments
    if not message.attachments:   
        await interaction.response.send_message(embed=getMessageEmbed("No Picture to Detect"), ephemeral=True)      
        return
            
    all_detect_result_stream: list[io.BytesIO] = []
    
    await interaction.response.defer()

    for item in message.attachments:
        if not isImageFile(item.filename):
            continue
        #await interaction.response.send_message(embed=__getMessageEmbed(f"{item.filename}: Detecting"))
        await interaction.followup.send(embed=getMessageEmbed(f"{item.filename}: Detecting"))
            
        #Read image data
        img_data = await item.read()        
        print("img_data read done!")
        #Get PIL image
        try:                      
            image = Image.open(io.BytesIO(img_data))
            if isPNGFile(item.filename):
                image = image.convert('RGB') #if there are 4 channels(RGBA) covert to 3 channels(RGB)
        except:    
            await interaction.followup.send(embed=getMessageEmbed(f"{item.filename}: Get PIL error", "⛔"), ephemeral=True)    
            continue

        result_stream = yolov8_service().detect_object_info(image, item.filename)
        if result_stream is None:
            #await interaction.followup.send(embed=getMessageEmbed(f"{item.filename}: Detected nothing", "😓"), ephemeral=True)            
            continue  

        print("detect done!")

        all_detect_result_stream.append(result_stream)

    if len(all_detect_result_stream) <= 0:
        await interaction.followup.send(embed=getMessageEmbed(f"{item.filename}: Detected nothing", "😓"), ephemeral=True)            
        return

    files = [discord.File(fp=stream, filename=f'detected_{i}.jpg') for i, stream in enumerate(all_detect_result_stream)]
    await interaction.followup.send(embed=getMessageEmbed("Detected", "😎"), files=files)
    
async def setup(bot):
    await bot.add_cog(Commands(bot))
    # 將上下文菜單指令加入到 bot 的 tree 中
    bot.tree.add_command(detect_image_context)

def isPNGFile(filename:str) -> bool:
    if filename.lower().endswith(('png')):
        return True
    return False
        
def isImageFile(filename:str) -> bool:
    if filename.lower().endswith(('png', 'jpg', 'jpeg')):
        return True
    return False
      
def getMessageEmbed(message:str, icon:str="🔊") -> discord.Embed:
    return discord.Embed(
        description=f"{icon} {message}",
        color=discord.Color.from_rgb(184, 134, 11)
    )