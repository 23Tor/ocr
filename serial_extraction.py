import cv2 as cv
import pytesseract
import os
import glob
import mariadb
import sys

# Tesseract path
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\tmorton\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# relative pos of serial num 
x = 0.618  # horizontal 
y = 0.28 # vertical 
w = 0.219  # width 
h = 0.099  # height 

# files in cropped folder
path = "source_img/cropped"
files = glob.glob(os.path.join(path, "*.jpg"))

# connect to db
try:
    conn = mariadb.connect(
        user="root",
        password='',
        host="localhost",
        database="bills",
        port=3306
    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

# get cursor
cur = conn.cursor()

# extract serial number from image
for file in files:
    # load img
    img = cv.imread(file)
    assert img is not None, "file could not be read, check path"

    # relative pos to absolute pos based on img size
    height, width = img.shape[:2]
    x_abs = int(x * width)
    y_abs = int(y * height)
    w_abs = int(w * width)
    h_abs = int(h * height)

    # crop
    img_cropped = img[y_abs:y_abs+h_abs, x_abs:x_abs+w_abs]

    # to grey
    img_gray = cv.cvtColor(img_cropped, cv.COLOR_BGR2GRAY)

    # add gaussian blur
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)

    # rescale
    img_rescaled = cv.resize(img_blur, None, fx=3, fy=3, interpolation=cv.INTER_CUBIC)
    cv.imwrite("source_img/rescaled.jpg", img_rescaled)

    # binarize
    _, img_bin = cv.threshold(img_rescaled, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    cv.imwrite("source_img/binarized.jpg", img_bin)

    try:
        # extract text
        config = '--oem 3 --psm 6'
        text = pytesseract.image_to_string(img_bin, config=config)
    
        if text:
            print(text)
            cur.execute("INSERT INTO bills (serial) VALUES (?)", (text,))
            conn.commit()

        else:
            print("No text extracted from image")
    
    except Exception as e:
        print("An error occurred: ", e)
