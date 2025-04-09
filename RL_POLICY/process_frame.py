import torch
import cv2
import numpy as np
import scipy.special
from PIL import Image
from model.model import parsingNet
from data.constant import tusimple_row_anchor
from torchvision import transforms
from ultralytics import YOLO
import time
import logging
from ultralytics.utils import LOGGER

# Suppress YOLO logs
LOGGER.setLevel(logging.ERROR)

# --- CONFIG ---
IMG_W, IMG_H = 1280, 720
GRIDDING_NUM = 100
CLS_NUM_PER_LANE = 56
ROW_ANCHOR = tusimple_row_anchor
BACKBONE = '18'
LANE_MODEL_PATH = 'models/tusimple_18.pth'
YOLO_MODEL_PATH = 'models/yolo11s.pt'  

# --- DEVICE ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load LaneNet ---
lane_net = parsingNet(pretrained=False, backbone=BACKBONE,
                      cls_dim=(GRIDDING_NUM + 1, CLS_NUM_PER_LANE, 4),
                      use_aux=False).to(device)
state_dict = torch.load(LANE_MODEL_PATH, map_location='cpu')['model']
lane_net.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}, strict=False)
lane_net.eval()

# --- Load YOLO ---
yolo_model = YOLO(YOLO_MODEL_PATH)

# --- Transforms ---
img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def process_frame(rgb_image: np.ndarray) -> np.ndarray:
    """
    Process raw RGB frame:
    - Detect lanes using LaneNet
    - Detect objects using YOLOv8
    - Draw both on black background
    - Return the single processed frame (uint8, shape=(720, 1280, 3))

    :param rgb_image: np.ndarray, (H, W, 3), dtype=uint8
    :return: processed_frame (np.ndarray)
    """
    start_time = time.time()

    # Resize input to standard size
    rgb_image = cv2.resize(rgb_image, (IMG_W, IMG_H))
    black_background = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)

    # ----- Lane Detection -----
    pil_img = Image.fromarray(rgb_image)
    net_input = img_transforms(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = lane_net(net_input)

    col_sample = np.linspace(0, 800 - 1, GRIDDING_NUM)
    col_sample_w = col_sample[1] - col_sample[0]
    out_j = out[0].data.cpu().numpy()
    out_j = out_j[:, ::-1, :]
    prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
    idx = np.arange(GRIDDING_NUM).reshape(-1, 1, 1) + 1
    loc = np.sum(prob * idx, axis=0)
    out_j = np.argmax(out_j, axis=0)
    loc[out_j == GRIDDING_NUM] = 0
    out_j = loc

    # Draw lane points
    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    x = int(out_j[k, i] * col_sample_w * IMG_W / 800) - 1
                    y = int(IMG_H * (ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - k] / 288)) - 1
                    cv2.circle(black_background, (x, y), 5, (0, 255, 0), -1)

    # ----- YOLO Object Detection -----
    yolo_results = yolo_model(rgb_image, verbose=False, device=device)
    for result in yolo_results:
        boxes = result.boxes.xyxy.cpu().numpy().astype(int)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(black_background, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # # Optional: Draw FPS on top left
    # fps = 1.0 / (time.time() - start_time)
    # cv2.putText(black_background, f"FPS: {fps:.2f}", (10, 30),
    #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return black_background
