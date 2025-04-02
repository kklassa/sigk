import rawpy
import cv2
import os
from dotenv import load_dotenv

load_dotenv()
FILE_PATH = os.path.normpath(f"{os.getenv("SIHDR_DATA_PATH")}\\raw\\001\\_07A5797.CR2")

raw = rawpy.imread(FILE_PATH) # access to the RAW image
rgb = raw.postprocess() # a numpy RGB array
image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
image_resized = cv2.resize(image, (512, 512))
cv2.imshow('original', image_resized)
cv2.waitKey(0)
