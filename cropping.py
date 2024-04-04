import cv2 as cv
import numpy as np
import os
import glob

# count number of images
path = "source_img/precrop"
files = glob.glob(os.path.join(path, "*.jpg"))

# read images
for file in files:
    # read img
    img = cv.imread(file)
    assert img is not None, f"Image {file} was not read"

    # find edges
    edges = cv.Canny(img, 100, 200)

    # find min max of canny edges
    pts = np.argwhere(edges > 0)
    y1, x1 = pts.min(axis=0)
    y2, x2 = pts.max(axis=0)

    # crop img to those points
    cropped_img = img[y1:y2, x1:x2]

    # file basename
    filename = os.path.basename(file)
    cv.imwrite(f"source_img/cropped/{filename}", cropped_img)
