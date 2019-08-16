import cv2 as cv
import numpy as np
import time as time
from matplotlib import pyplot as plt


# 图像中的直方图：主要指以像素值为 x 轴，像素个数为y轴的直方图
def plot_demo(image):
    plt.hist(image.ravel(), 256, [0, 256])  # ravel()函数与flatten()函数都是将多维数组变为一维数组
    print(image.ravel().shape)
    plt.show()

# bgr 三色分开的直方图
def image_hist(image):
    color = ('blue', 'green', 'red')
    for i, color in enumerate(color):
        hist = cv.calcHist([image], [i], None, [256], [0, 256])
        print(hist.shape)
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()


# 均衡直方图:对比度增强
def equalHist_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)# 一定是灰度图
    dst = cv.equalizeHist(gray)
    cv.imshow("queal_demo", dst)
    plot_demo(dst)


# 局部自适应直方图
def clahe_demo(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)# 灰度图
    clahe = cv.createCLAHE(5.0, (8, 8))
    dst = clahe.apply(gray)
    cv.imshow("clahe_demo", dst)
    plot_demo(dst)

# 三通道直方图
def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16 ** 3, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = np.int(b / bsize) * 16 * 16 + np.int(g / bsize) * 16 + np.int(r / bsize)
            rgbHist[np.int(index), 0] += 1
    print(rgbHist.shape)
    plt.plot(rgbHist)
    plt.show()
    return rgbHist


# 直方图比较
def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv.compareHist(hist1, hist2, cv.HISTCMP_BHATTACHARYYA)
    match2 = cv.compareHist(hist1, hist2, cv.HISTCMP_CORREL)
    match3 = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
    print("巴氏距离：{}，相关性：{}，卡方：{}".format(match1, match2, match3))


path = "C:\\Users\\cht\\Pictures\\Saved Pictures\\"
img = cv.imread(path + "327E9E4FB9613CDCB996065D4B61E4A3.jpg")
cv.imshow("src", img)

t1 = time.time()  # t1 = cv.getCPUTickCount()
# plot_demo(img)
image_hist(img)
# equalHist_demo(img)
# clahe_demo(img)
img2 = cv.imread(path + "u=1087030622,884055929&fm=26&gp=0.jpg")
cv.imshow("src2",img2)
hist_compare(img,img2)
t2 = time.time()  # t2 = cv.getCPUTickCount()
print(t2 - t1)  # print((t2 - t1) / cv.getTickFrequency())

cv.waitKey(0)
cv.destroyAllWindows()
