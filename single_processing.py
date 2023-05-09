import numpy as np
import cv2
import PIL.Image as Image
#读入图片
image = cv2.imread("xxx.jpg")
image = cv2.resize(image, (500, 500))
#cv2.imshow("initial",image)
#阈值化
mask = np.logical_or(image[:, :, 2] <= 80, image[:, :, 2] >= 115)
image[mask] = np.array([255, 255, 255])
image[~mask] = np.array([0, 0, 0])
image = cv2.resize(image, (100, 100))
#media滤波
position = 8
if (position <= 0):
    position = 1
if(position%2 == 0):
    position = position + 1
media_img = cv2.medianBlur(image, position)
media_img = cv2.resize(media_img, (500, 500))
#jpg格式转为png格式
img_shape = media_img.shape
h = img_shape[0]
w = img_shape[1]
dst = 255 - media_img
#cv2.imshow('dst', dst)
media_img = cv2.resize(media_img, (28, 28))
cv2.imwrite("media.jpg", dst)
img = Image.open('media.jpg')
img = img.convert('RGBA')
L, H = img.size
color_0 = (255, 255, 255, 255)
for h in range(H):
    for l in range(L):
        dot = (l, h)
        color_1 = img.getpixel(dot)
        if color_1 == color_0:
            color_1 = color_1[:-1] + (0,)
            img.putpixel(dot, color_1)
img.save('img2.png')
cv2.waitKey()
cv2.destroyAllWindows()
