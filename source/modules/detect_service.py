from PIL import Image, ImageDraw, ImageFont
import io, random
from modules.models.yolov8.yolov8_model import get_detect_result_info
from modules.models.dogvision.dogvision_model import get_dogs_breed
import numpy as np
class yolov8_service:
    def __init__(self):
        pass
    
    def pil_to_bytes(self, image):
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='JPEG')  # 或者其他適合的格式，例如 'PNG'
        img_byte_arr = img_byte_arr.getvalue()  # 獲取 bytes 資料
        return img_byte_arr
    
    def detect_object_info(self, image: Image, filename: str) -> io.BytesIO:
        #resized_stream = self.__resizeAndGetImageStream(image=image)
        resized_stream = io.BytesIO()
        image.save(resized_stream, format='JPEG')

        detect_result = get_detect_result_info(resized_stream)
        
        classNames = list(map(lambda item: item.className, detect_result.detections))
        if not classNames or len(classNames) <= 0:
            return None
        
        uniqueClassNames = set(classNames)
        colors = self.__getUniqueColors(len(uniqueClassNames))

        #create dict about className: color
        colorDict = dict(zip(uniqueClassNames, colors))
        
        image = Image.open(resized_stream)
        draw = ImageDraw.Draw(image)
        
        for detection in detect_result.detections:
            
            x, y, w, h = detection.box[0]
            className = detection.className
            if className != 'dog':
                continue
            
            # 裁切出邊界框內的狗
            cropped_image = image.crop((x - w/2, y - h/2, x + w/2, y + h/2))
            breed = get_dogs_breed([self.pil_to_bytes(cropped_image)])[0]
            
            confidence = round(detection.confidence, 2)
            #label_text = f"{className} {confidence:.2f}"
            label_text = breed

            #set frame and font style
            color = colorDict[className]
            frame_color = self.__getRgb(color['background'])
            font = ImageFont.truetype("configs/fonts/Arial.ttf", 35)
            font_color = self.__getRgb(color['font'])

            #calculate frame position and draw the frame
            top_left = (x - w / 2, y - h / 2)
            bottom_right = (x + w / 2, y + h / 2)
            draw.rectangle([top_left, bottom_right], outline=frame_color, width=5)
            
            #use textbbox to calculate text width and height
            bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            #the padding of text background
            padding = 15
                  
            label_position = (top_left[0], top_left[1] - text_height - 3 - 8)
            label_background = [label_position, (label_position[0] + text_width, label_position[1] + text_height + padding)]
        
            #if label is high than pic, move label into frame
            if label_position[1] < 0:
                label_position = (top_left[0], top_left[1] + 3)
                label_background = [label_position, (label_position[0] + text_width, label_position[1] + text_height + padding)]
            
            #draw label background
            draw.rectangle(label_background, fill=frame_color)
            
            #draw label text
            draw.text((label_position[0] + 3, label_position[1]), label_text, fill=font_color, font=font)
        
        output = io.BytesIO()
        image.save(output, format="JPEG")
        output.seek(0)  
        return output

    def __resizeAndGetImageStream(self, image: Image) -> io.BytesIO:
        output = io.BytesIO()
        while True:
            output.seek(0)
            image.save(output, format='JPEG')

            file_size = output.tell()  # `tell()` return stream size
            print(file_size)
            if file_size <= 200 * 1024:  # stream size <= 200KB stop compression
                break
            else:
                #resize image
                width, height = image.size
                new_size = (int(width * 0.8), int(height * 0.8))  #80%
                image = image.resize(new_size, Image.Resampling.LANCZOS) 
        output.seek(0)
        return output   
    
    def __getColorList(self) -> list:
        return [
            #{"name": "Carnation Pink", "background": "#FF7FAE", "font": "#FFFFFF"},
            {"name": "Peach Orange", "background": "#FFCA99", "font": "#D32F2F"},
            {"name": "Moody Blue", "background": "#646BD9", "font": "#FFFFFF"},
            {"name": "Windstorm", "background": "#6C98C6", "font": "#FFFFFF"},
            #{"name": "White Smoke", "background": "#F8F7F4", "font": "#A57E6E"},
            #{"name": "Coconut Butter", "background": "#F3F0E2", "font": "#354B5E"},
            {"name": "Summer Lily", "background": "#F5D372", "font": "#7A7A47"},
            #{"name": "Grey Whisper", "background": "#E4E4E4", "font": "#A57E6E"},
            {"name": "Dandelion", "background": "#FED361", "font": "#D32F2F"},
            #{"name": "Grim Grey", "background": "#E5DDD7", "font": "#A57E6E"},
            {"name": "Rock n' Rose", "background": "#FC8CA8", "font": "#B23A5D"},
            {"name": "Midnight Blue", "background": "#2E364F", "font": "#EF3A3A"},
            {"name": "Bleach White", "background": "#FEEEDA", "font": "#6459D2"},
            {"name": "Mulberry", "background": "#CF5991", "font": "#FFFFFF"},
            {"name": "Often Orange", "background": "#FE7250", "font": "#FFFFFF"},
            #{"name": "Woodrose", "background": "#AE8D92", "font": "#FFFFFF"},
            #{"name": "Finest Silk", "background": "#F1E5D6", "font": "#D99268"},
            {"name": "Fiery Coral", "background": "#E25A53", "font": "#FFFFFF"}
        ]

    def __getUniqueColors(self, count: int) -> list:
        colors =  self.__getColorList()
        while (count > len(colors)):
            colors = colors.append(colors)
        return random.sample(colors, count)
    
    def __getRgb(self, hex_color: str) -> tuple:
        #Convert HEX to RGB
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))      