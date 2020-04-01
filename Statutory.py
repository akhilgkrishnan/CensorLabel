from PIL import Image 
import cv2 as cv


def cv2_to_pil(img): #Since you want to be able to use Pillow (PIL)
    return Image.fromarray(cv.cvtColor(img, cv.COLOR_BGR2RGB))    


def add_warning(img,height,dtype):
    logo = Image.open(dtype)
    pil_img = cv2_to_pil(img)
    logo = logo.convert("RGBA")
    logo = logo.resize((250,40))
    image_copy = pil_img.copy()
    position = (10,height-65)
    image_copy.paste(logo, position,logo)
    image_copy.save("pasted_image.jpg")