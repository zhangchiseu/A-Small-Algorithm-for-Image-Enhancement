import cv2

# Read image
img = cv2.imread("C:/Users/15729/PycharmProjects/pythonProject5/thresholding.jpg")

def onSizeChange(x):
    position = cv2.getTrackbarPos("cannyBar", "Bar")
    print(position)
    if (position <= 0):
        return
    blur_img = cv2.blur(img, (position, position))
    bilater_img = cv2.bilateralFilter(img, position, position, position)
    if(position%2 == 0):
        position = position + 1
    gauss_img = cv2.GaussianBlur(img, (position, position), 1)
    media_img = cv2.medianBlur(img, position)
    cv2.imshow("Canny", blur_img)
    cv2.imshow("Gauss", gauss_img)
    gauss_img = cv2.resize(gauss_img, (50, 50))
    cv2.imwrite("gauss.jpg",gauss_img)

    cv2.imshow("Media", media_img)
    cv2.imshow("Bilater", bilater_img)

cv2.namedWindow("Bar")
cv2.createTrackbar("cannyBar", "Bar", 0, 15, onSizeChange)

cv2.waitKey(0)