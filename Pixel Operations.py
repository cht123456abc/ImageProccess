import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def add_demo(m1, m2):
    dst = cv.add(m1, m2)
    cv.imshow("add_demo", dst)
    dst2 = cv.addWeighted(m1, 0.7, m2, 0.3, 0)
    cv.imshow("addWeighted_demo", dst2)


def substact_demo(m1, m2):
    dst = cv.subtract(m1, m2)
    cv.imshow("subtract_demo", dst)


def divide_demo(m1, m2):
    dst = cv.divide(m1, m2)
    cv.imshow("dicide_demo", dst)


# 逻辑运算 bitwise:位运算
def bitwise_and(m1, m2):
    # dst1 = cv.bitwise_and(m1, m2)  # 逻辑与实现遮罩层
    # cv.imshow("and_demo", dst1)

    # 用mask实现遮罩层
    img2gray = cv.cvtColor(m2, cv.COLOR_RGB2GRAY)
    ret, mask = cv.threshold(img2gray, 175, 255, cv.THRESH_BINARY)
    plt.subplot(221)
    plt.imshow(mask, "gray")
    plt.title("mask")
    mask_inv = cv.bitwise_not(mask)
    plt.subplot(222)
    plt.imshow(mask_inv, "gray")
    plt.title("mask_inv")
    img1_bg = cv.bitwise_and(m1, m1, mask=mask_inv)
    plt.subplot(223)
    plt.imshow(img1_bg)
    plt.title("img1_bg")
    img2_fg = cv.bitwise_and(m1, m1, mask=mask)
    plt.subplot(224)
    plt.imshow(img2_fg)
    plt.title("img2_fg")
    plt.show()
    dst = cv.add(img1_bg, m2)
    plt.imshow(dst)
    plt.show()


def bitwise_not(m1):
    dst2 = cv.bitwise_not(m1)
    cv.imshow("not_demo", dst2)


def bitwise_or(m1, m2):
    dst3 = cv.bitwise_or(m1, m2)
    cv.imshow("or_demo", dst3)


def bitwise_xor(m1, m2):
    dst4 = cv.bitwise_xor(m1, m2)
    cv.imshow("xor_demo", dst4)


# c:对比度，b:亮度
def contrast_brightness_demo(img, c, b):
    h, w, ch = img.shape
    blank = np.zeros([h, w, ch], img.dtype)
    dst = cv.addWeighted(img, c, blank, 1 - c, b)
    cv.imshow("con-bri-demo", dst)


def multiple_demo(m1, m2):
    dst = cv.multiply(m1, m2)
    cv.imshow("multiple_demo", dst)


def others(m1, m2):
    M1, dev1 = cv.meanStdDev(m1)
    M2, dev2 = cv.meanStdDev(m2)
    print(M1, dev1)
    print(M2, dev2)


path = "C:\\Users\\cht\\Pictures\\Saved Pictures\\"
img1 = plt.imread(path + "u=566325761,2083849247&fm=26&gp=0.jpg")
img2 = plt.imread(path + "u=1087030622,884055929&fm=26&gp=0.jpg")
print(img1.shape)
print(img2.shape)
plt.subplot(121)
plt.imshow(img1)
plt.title("img1")
plt.subplot(122)
plt.imshow(img2)
plt.title("img2")
plt.show()
# cv.imshow("image1", img1)
# cv.imshow("image2", img2)

# add_demo(img1, img2)
# substact_demo(img1, img2)
# divide_demo(img1, img2)
# multiple_demo(img1, img2)
bitwise_and(img1, img2)
# others(img1, img2)
# contrast_brightness_demo(img1, 15, 0)

cv.waitKey(0)
cv.destroyAllWindows()
