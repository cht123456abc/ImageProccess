import cv2 as cv
import numpy as np


def ROI(image):
    face = image[50:250, 100:300]
    gray = cv.cvtColor(face, cv.COLOR_BGR2GRAY)
    backface = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    image[50:250, 100:300] = backface
    cv.imshow("image2", image)


def flood_fill(image):
    copyImg = image.copy()
    h, w = image.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv.floodFill(copyImg, mask, (30, 30), (0, 255, 0), (40, 40, 40), (50, 50, 50), cv.FLOODFILL_FIXED_RANGE)
    cv.imshow("fill", copyImg)


def fill_binary():
    image = np.zeros([400, 400, 3], np.uint8)
    image[100:300, 100:300] = 255

    mask = np.ones([402, 402, 1], np.uint8)
    mask[101:301, 101:301] = 0
    cv.floodFill(image, mask, (200, 200), (0, 0, 255), cv.FLOODFILL_MASK_ONLY)
    cv.imshow("fill_binary", image)


path = "C:\\Users\\cht\\Pictures\\Saved Pictures\\"
imag = cv.imread(path + "327E9E4FB9613CDCB996065D4B61E4A3.jpg")
print(imag.shape)
cv.imshow("image1", imag)

ROI(imag)

flood_fill(imag)

fill_binary()



cv.waitKey(0)
cv.destroyAllWindows()
