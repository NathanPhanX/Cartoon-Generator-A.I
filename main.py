from PIL import Image

origin_img = Image.open("brand-origin.jpg")
img = Image.new('RGB', size=origin_img.size)
height, width = origin_img.size

for i in range(height):
    for j in range(width):
        if origin_img.getpixel((i, j)) == (255, 255, 255):
            img.putpixel((i, j), (251, 37, 37))
        elif origin_img.getpixel((i, j)) == (0, 0, 0):
            img.putpixel((i, j), (255, 255, 255))
        else:
            img.putpixel((i, j), origin_img.getpixel((i, j)))


img.save('brand.jpg')
print('Success')
