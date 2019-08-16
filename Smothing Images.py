import cv2 as cv
import numpy as np
import time as time


# 均值模糊
def blur_demo(image):
    dst = cv.blur(image, (1, 20))
    cv.imshow("blur_demo", dst)


# 中值模糊
def median_blur_demo(image):
    dst = cv.medianBlur(image, 5)
    cv.imshow("median_blur_demo", dst)


# 自定义模糊
def custom_blur_demo(image):
    # kernel = np.ones([5,5], np.float32) / 25
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], np.float32)# 锐化算子 总和为1，是增强；总为为0：边缘梯度
    dst = cv.filter2D(image, -1, kernel=kernel)
    cv.imshow("custom_blur_demo", dst)


path = "C:\\Users\\cht\\Pictures\\Saved Pictures\\"
img = cv.imread(path + "327E9E4FB9613CDCB996065D4B61E4A3.jpg")
cv.imshow("src", img)

t1 = time.time()  # t1 = cv.getCPUTickCount()
blur_demo(img)
median_blur_demo(img)
custom_blur_demo(img)

t2 = time.time()  # t2 = cv.getCPUTickCount()
print(t2 - t1)  # print((t2 - t1) / cv.getTickFrequency())

cv.waitKey(0)
cv.destroyAllWindows()
