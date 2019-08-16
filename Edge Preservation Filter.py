import cv2 as cv
import numpy as np
import time as time


# 高斯双边
def bi_demo(image):
    dst = cv.bilateralFilter(image, 0, 100, 15)
    cv.imshow("bi_demo", dst)

# 均值迁移，边缘过渡模糊
def mean_shift_demo(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 50)
    cv.imshow("shift_demo", dst)


path = "C:\\Users\\cht\\Pictures\\Saved Pictures\\"
img = cv.imread(path + "327E9E4FB9613CDCB996065D4B61E4A3.jpg")
cv.imshow("src", img)

t1 = time.time()  # t1 = cv.getCPUTickCount()
bi_demo(img)
mean_shift_demo(img)
t2 = time.time()  # t2 = cv.getCPUTickCount()
print(t2 - t1)  # print((t2 - t1) / cv.getTickFrequency())

cv.waitKey(0)
cv.destroyAllWindows()
