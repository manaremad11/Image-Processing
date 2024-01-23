from PIL import Image
import os
path="C:\\Users\MSI\PycharmProjects\ImageProcessing\project\images\\"
images=os.listdir(path)
for i in images:
    image=img = Image.open(path+i)
    gray_img = img.convert('L')
    gray_img.save("C:\\Users\MSI\PycharmProjects\ImageProcessing\project\images_gray\\"+i)


