import cv2 as cv
import numpy as np
import time as time


def split_channels(image):
    b, g, r = cv.split(image)
    cv.imshow("blue", b)
    cv.imshow("green", g)
    cv.imshow("red", r)
    cv.waitKey(0)


def extract_object_demo():
    path = "D:\\cht\\Pictures\\Camera Roll\\WIN_20190523_15_41_53_Pro.mp4"
    capture = cv.VideoCapture(path)
    while True:
        ret, frame = capture.read()
        if ret == False:
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        lower_hsv = np.array([0, 0, 110])  # hsv三通道低值
        higher_hsv = np.array([255, 255, 255])  # hsv三通道高值
        mask = cv.inRange(hsv, lowerb=lower_hsv, upperb=higher_hsv)
        cv.imshow("video", frame)
        cv.imshow("mask", mask)
        if cv.waitKey(1) == 27:
            break


def color_space_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("gray", gray)
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    cv.imshow("hsv", hsv)
    yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    cv.imshow("yuv", yuv)
    cv.waitKey(0)


path = "C:\\Users\\cht\\Pictures\\Camera Roll\\WIN_20190227_10_48_39_Pro.jpg"
cv.namedWindow("video", cv.WINDOW_NORMAL)
img = cv.imread(path)

t1 = time.time()  # t1 = cv.getCPUTickCount()
# color_space_demo(img)
# extract_object_demo()
split_channels(img)
t2 = time.time()  # t2 = cv.getCPUTickCount()
print(t2 - t1)  # print((t2 - t1) / cv.getTickFrequency())

cv.destroyAllWindows()
