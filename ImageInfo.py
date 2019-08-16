import cv2 as cv
import numpy as np
import time as time

def inverse(image):
    print(image.shape)
    print(image[0, 0, 0])
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width: {}, height: {}, channels: {}".format(height, width, channels))
    for row in range(height):
        for col in range(width):
            for c in range(channels):
                pv = image[row, col, c]
                image[row, col, c] = 255 - pv
    cv.imshow("img", image)


def inverse2(image):
    dst = cv.bitwise_not(image)# image = 255 - image
    cv.imshow("inverse demo", dst)

def create_image():
    img = np.zeros([400, 400, 3], np.uint8)
    img[:, :, 0] = np.ones([400,400]) * 255
    cv.imshow("new image", img)

path = "C:\\Users\\cht\\Pictures\\Camera Roll\\WIN_20190227_10_48_39_Pro.jpg"
img = cv.imread(path)
t1 = time.time()# t1 = cv.getCPUTickCount()
inverse(img)
t2 = time.time()# t2 = cv.getCPUTickCount()
print(t2 - t1)# print((t2 - t1) / cv.getTickFrequency())
create_image()

cv.waitKey(0)
cv.destroyAllWindows()
