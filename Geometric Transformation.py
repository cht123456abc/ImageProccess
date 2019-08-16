import cv2 as cv
import numpy as np
import time as time
from matplotlib import pyplot as plt


def resize(image):
    dst = cv.resize(image, dsize=None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    cv.imshow("dst2x", dst)
    dst = cv.resize(image, (3 * image.shape[1], 3 * image.shape[0]))
    cv.imshow("dst3x", dst)


def translation(image):
    M = np.float32([[1, 0, 100], [0, 1, 50]])
    dst = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    cv.imshow("translation", dst)


def rotation(image):
    M = cv.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), 90, 1)
    dst = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    cv.imshow("rotation", dst)


def affine_transformation(image):
    # 用三个点来标志
    pst1 = np.float32([[50, 50], [200, 50], [50, 200]])
    pst2 = np.float32([[10, 100], [200, 50], [100, 250]])
    M = cv.getAffineTransform(pst1, pst2)
    dst = cv.warpAffine(image, M, (image.shape[1], image.shape[0]))
    plt.subplot(1, 2, 1), plt.imshow(image), plt.title("input")
    plt.subplot(1, 2, 2), plt.imshow(dst), plt.title("output")
    plt.show()


def perspective_transformation(winname, image):
    mouse2axis(winname, image)
    # 将图像的四个角的位置重新定义
    pts1 = np.float32([[112, 111], [364, 142], [44, 365], [341, 407]])
    pts2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])
    M = cv.getPerspectiveTransform(pts1, pts2)
    dst = cv.warpPerspective(image, M, ( 300, 300))
    cv.imshow("perspective_transformation", dst)

def mouse2axis(winname, image):
    def event_showAxis(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            text = "{},{}".format(x, y)
            print(text)
            # cv.putText(image, text, (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 1, cv.LINE_AA)

    cv.setMouseCallback(winname, event_showAxis)


def main():
    img = cv.imread("img/suduko.jpg")
    cv.namedWindow("src")
    cv.imshow("src", img)
    # cv.putText(img, "haha", (50, 50), cv.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 1, cv.LINE_AA)
    t1 = time.time()  # t1 = cv.getCPUTickCount()
    # resize(img)
    # translation(img)
    # rotation(img)
    # affine_transformation(img)
    perspective_transformation("src", img)

    t2 = time.time()  # t2 = cv.getCPUTickCount()
    print(t2 - t1)  # print((t2 - t1) / cv.getTickFrequency())

    cv.waitKey(0)
    cv.destroyAllWindows()


main()
