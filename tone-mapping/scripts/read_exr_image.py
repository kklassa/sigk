import os
# enable using OpenEXR with OpenCV
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = "1"
import cv2
from numpy import ndarray
from brisque import BRISQUE
from dotenv import load_dotenv

load_dotenv()
FILE_PATH = os.path.normpath(f"{os.getenv("SIHDR_DATA_PATH")}\\reconstructions\\maskhdr\\clip_95\\001.exr")

def read_exr(im_path: str) -> ndarray:
    return cv2.imread(
        filename=im_path,
        flags=cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
    )
    
def tone_map_reinhard(image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapReinhard(
        gamma=2.2,
        intensity=0.0,
        light_adapt=0.0,
        color_adapt=0.0
    )
    result = tonemap_operator.process(src=image)
    return result

def tone_map_mantiuk(image: ndarray) -> ndarray:
    tonemap_operator = cv2.createTonemapMantiuk(
        gamma=2.2,
        scale=0.85,
        saturation=1.2
    )
    result = tonemap_operator.process(src=image)
    return result

def evaluate_image(image: ndarray) -> float:
    metric = BRISQUE(url=False)
    return metric.score(img=image)

if __name__ == '__main__':
    image = read_exr(im_path=FILE_PATH)
    width, height = image.shape[1], image.shape[0]
    print(f"Width: {width}, Height: {height}")
    image = cv2.resize(image, (512, 512))
    tone_mapped_reinhard = tone_map_reinhard(image)
    tone_mapped_mantiuk = tone_map_mantiuk(image)
    cv2.imshow('original', image)
    cv2.imshow('tone_mapped_reinhard', tone_mapped_reinhard)
    cv2.imshow('tone_mapped_mantiuk', tone_mapped_mantiuk)
    print('tone_mapped_reinhard', evaluate_image(image=tone_mapped_reinhard))
    print('tone_mapped_mantiuk', evaluate_image(image=tone_mapped_mantiuk))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
