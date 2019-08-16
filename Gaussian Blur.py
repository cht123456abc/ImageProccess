import cv2 as cv
import numpy as np
import time as time


# 加上高斯噪音
def add_gausian_noise(image):
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            # s = np.random.randint(0, 255, 3)
            add = s + image[row, col]
            for i in range(len(add)):
                if add[i] > 255:
                    add[i] = 255
                elif add[i] < 0:
                    add[i] = 0
            image[row, col] = add
    cv.imshow("add_noise", image)


# 高斯模糊
def gaussian_blur(image):
    dst = cv.GaussianBlur(image, (0, 0), 10)
    cv.imshow("gaussian_blur", dst)


# 高斯去噪（去高斯噪声）
def gaussian_remove(image):
    dst = cv.GaussianBlur(image, (5, 5), 0)
    cv.imshow("gaussian_remove",dst)


path = "C:\\Users\\cht\\Pictures\\Saved Pictures\\"
img = cv.imread(path + "327E9E4FB9613CDCB996065D4B61E4A3.jpg")
cv.imshow("src", img)

t1 = time.time()  # t1 = cv.getCPUTickCount()
add_gausian_noise(img)
# gaussian_blur(img)
gaussian_remove(img)
t2 = time.time()  # t2 = cv.getCPUTickCount()
print(t2 - t1)  # print((t2 - t1) / cv.getTickFrequency())

cv.waitKey(0)
cv.destroyAllWindows()
