import cv2 as cv
import numpy as np
import os

# count number of images
path = "input"
files = os.listdir(path)
num_bills = len(files)

# read images
for i in range(1, num_bills + 1):
    # read img
    img = cv.imread(f"precrop/{i}.jpg")
    assert img is not None, f"Image {i} was not read"

    # find edges
    edges = cv.Canny(img, 100, 200)

    # find min max of canny edges
    pts = np.argwhere(edges > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    # crop img to those points
    cropped_img = img[y1:y2, x1:x2]
    cv.imwrite(f"cropped/{i}.jpg", cropped_img)
