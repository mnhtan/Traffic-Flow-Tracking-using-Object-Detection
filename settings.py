from pathlib import Path
import sys

# Get the absolute path of the current file
FILE = Path(__file__).resolve()
# Get the parent directory of the current file
ROOT = FILE.parent
# Add the root path to the sys.path list if it is not already there
if ROOT not in sys.path:
    sys.path.append(str(ROOT))
# Get the relative path of the root directory with respect to the current working directory
ROOT = ROOT.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
#WEBCAM = 'Webcam'
# RTSP = 'RTSP'
# YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'city_traffic.jpg'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'city_traffic_detected.jpg'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEOS_DICT = {
    'city_traffic': VIDEO_DIR / 'city_traffic.mp4',
    'Video_sample_2': VIDEO_DIR / 'Video_sample_2.mp4',
    'video_1': VIDEO_DIR / 'video_1.mp4',
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
DETECTION_MODEL = MODEL_DIR / 'trained_yolov8n_S640R15.pt'
# In case of your custome model comment out the line above and
# Place your custom model pt file name at the line below 
# DETECTION_MODEL = MODEL_DIR / 'my_detection_model.pt'
DETECTION_DICT = {
    'yolov8n': MODEL_DIR / 'trained_S640R15_yolov8n.pt' ,
    'yolov8n_trained': MODEL_DIR / 'trained_SR_DSrc_yolov8n.pt',
    'yolov8s_trained': MODEL_DIR / 'trained_SR_DSrc_yolov8s.pt',
    'yolov8m_trained': MODEL_DIR / 'trained_SR_DSrc_yolov8m.pt',
    'yolov8l_trained': MODEL_DIR / 'trained_SR_DSrc_yolov8l.pt',
    'yolov8x_trained': MODEL_DIR / 'trained_SR_DSrc_yolov8x.pt',

}

SEGMENTATION_MODEL = MODEL_DIR / 'yolov8n-seg.pt'

# Webcam
WEBCAM_PATH = 0
