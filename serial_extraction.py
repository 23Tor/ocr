import easyocr
import cv2 as cv
import os
import glob
import mariadb
import sys
import re

# create reader
reader = easyocr.Reader(['en'])

# files in cropped folder
path = "input"
files = glob.glob(os.path.join(path, "*.jpg"))

# function to find serial number pattern
def find_serial(results):
    pattern = r'\b[0-9]{8}\b'
    for result in results:
        text = result[1]  # not the bound boxes
        match = re.findall(pattern, text)
        if match:
            return match[0]
    return None

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

    result = reader.readtext(img)

    # extract serial number
    serial = find_serial(result)

    # insert serial number into db
    if serial:
        cur.execute("INSERT INTO bills (serial) VALUES (?)", (serial,))
        conn.commit()
        print(f"Serial number {serial} inserted into db")

# close db connection
conn.close()


