import os
import cv2
import numpy as np
input_folder = "xxxpath"
output_folder = "yyypath"
media_folder = "zzzpath"
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):
        # Read image
        image = cv2.imread(os.path.join(input_folder, filename))
        image = cv2.resize(image, (500, 500))
        # Thresholding
        mask = np.logical_or(image[:, :, 2] <= 80, image[:, :, 2] >= 115)
        image[mask] = np.array([255, 255, 255])
        image[~mask] = np.array([0, 0, 0])
        image = cv2.resize(image, (100, 100))
        # Median filtering
        position = 8
        if (position <= 0):
            position = 1
        if(position%2 == 0):
            position = position + 1
        media_img = cv2.medianBlur(image, position)
        media_img = cv2.resize(media_img, (500, 500))
        cv2.imwrite(os.path.join(media_folder, filename[:-4] + "_media.jpg"), media_img)
        # Convert to PNG
        img_shape = media_img.shape
        h = img_shape[0]
        w = img_shape[1]
        dst = 255 - media_img
        media_img = cv2.resize(dst, (28, 28))
        cv2.imwrite(os.path.join(output_folder, filename[:-4] + "_processed.png"), media_img)
