# import torch
# import cv2
# import numpy as np
# import scipy.special
# from PIL import Image
# from model.model import parsingNet
# from data.constant import tusimple_row_anchor
# from torchvision import transforms
# from ultralytics import YOLO
# import time
# import logging
# from ultralytics.utils import LOGGER

# # Suppress YOLO logs
# LOGGER.setLevel(logging.ERROR)

# # --- CONFIG ---
# IMG_W, IMG_H = 640, 360
# GRIDDING_NUM = 100
# CLS_NUM_PER_LANE = 56
# ROW_ANCHOR = tusimple_row_anchor
# BACKBONE = '18'
# LANE_MODEL_PATH = 'models/tusimple_18.pth'
# YOLO_MODEL_PATH = 'models/yolo11s.pt'  # Segmentation model (renamed from yolo11s-seg.pt)

# # --- DEVICE ---
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # --- Load LaneNet ---
# lane_net = parsingNet(pretrained=False, backbone=BACKBONE,
#                       cls_dim=(GRIDDING_NUM + 1, CLS_NUM_PER_LANE, 4),
#                       use_aux=False).to(device)
# state_dict = torch.load(LANE_MODEL_PATH, map_location='cpu')['model']
# lane_net.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}, strict=False)
# lane_net.eval()

# # --- Load YOLO ---
# yolo_model = YOLO(YOLO_MODEL_PATH)

# # --- Transforms ---
# img_transforms = transforms.Compose([
#     transforms.Resize((288, 800)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])

# def process_frame(rgb_image: np.ndarray) -> tuple[np.ndarray, float, float]:
#     """
#     Process raw RGB frame:
#     - Detect lanes using LaneNet
#     - Detect objects using YOLOv11-seg
#     - Draw both on black background
#     - Return the processed frame (uint8, shape=(360, 640, 3)), lane deviation, and obstacle proximity
#     """
#     rgb_image = cv2.resize(rgb_image, (IMG_W, IMG_H))  # 640x360
#     black_background = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)  # 360x640x3

#     # ----- Lane Detection -----
#     pil_img = Image.fromarray(rgb_image)
#     net_input = img_transforms(pil_img).unsqueeze(0).to(device)

#     with torch.no_grad():
#         out = lane_net(net_input)

#     col_sample = np.linspace(0, 800 - 1, GRIDDING_NUM)
#     col_sample_w = col_sample[1] - col_sample[0]
#     out_j = out[0].data.cpu().numpy()
#     out_j = out_j[:, ::-1, :]
#     prob = scipy.special.softmax(out_j[:-1, :, :], axis=0)
#     idx = np.arange(GRIDDING_NUM).reshape(-1, 1, 1) + 1
#     loc = np.sum(prob * idx, axis=0)
#     out_j = np.argmax(out_j, axis=0)
#     loc[out_j == GRIDDING_NUM] = 0
#     out_j = loc

#     lane_points = []
#     for i in range(out_j.shape[1]):
#         if np.sum(out_j[:, i] != 0) > 2:
#             for k in range(out_j.shape[0]):
#                 if out_j[k, i] > 0:
#                     x = int(out_j[k, i] * col_sample_w * IMG_W / 800) - 1
#                     y = int(IMG_H * (ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - k] / 288)) - 1
#                     lane_points.append((x, y))
#                     cv2.circle(black_background, (x, y), 5, (0, 255, 0), -1)

#     # Compute lane deviation
#     if lane_points:
#         lane_center_x = np.mean([p[0] for p in lane_points])
#         img_center_x = IMG_W / 2  # 320
#         lane_deviation = (lane_center_x - img_center_x) / (IMG_W / 2)  # Normalized [-1, 1]
#     else:
#         lane_deviation = 0.0

#     # ----- YOLOv11 Segmentation -----
#     yolo_results = yolo_model(rgb_image, verbose=False, device=device, conf=0.1)
#     min_obstacle_y = IMG_H  # Default to bottom of image (farthest)
#     for result in yolo_results:
#         if result.boxes is not None and len(result.boxes) > 0:
#             if result.masks is not None:
#                 masks = result.masks.data.cpu().numpy()
#                 for mask in masks:
#                     mask_resized = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
#                     color = (0, 0, 255)  # Red
#                     color_mask = np.zeros_like(black_background, dtype=np.uint8)
#                     color_mask[mask_resized.astype(bool)] = color
#                     black_background = cv2.addWeighted(black_background, 1.0, color_mask, 1.0, 0)
#                     # Find topmost point of mask for proximity
#                     mask_y = np.where(mask_resized)[0]
#                     if len(mask_y) > 0:
#                         min_obstacle_y = min(min_obstacle_y, min(mask_y))

#     # Compute obstacle proximity
#     if min_obstacle_y < IMG_H:
#         obstacle_proximity = (IMG_H - min_obstacle_y) / IMG_H  # Normalized [0, 1], 0 is far, 1 is close
#     else:
#         obstacle_proximity = 0.0  # No obstacles

#     return black_background, lane_deviation, obstacle_proximity




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
IMG_W, IMG_H = 640, 360  # Updated from 1280, 720 to match new camera settings
GRIDDING_NUM = 100
CLS_NUM_PER_LANE = 56
ROW_ANCHOR = tusimple_row_anchor
BACKBONE = '18'
LANE_MODEL_PATH = 'models/tusimple_18.pth'
YOLO_MODEL_PATH = 'models/yolo11m-seg'  # Note: Use 'yolo11s-seg.pt' if segmentation is needed

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
    - Detect objects using YOLOv8-seg
    - Draw both on black background
    - Return the single processed frame (uint8, shape=(360, 640, 3))
    """
    rgb_image = cv2.resize(rgb_image, (IMG_W, IMG_H))  # 640x360
    black_background = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)  # 360x640x3

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

    for i in range(out_j.shape[1]):
        if np.sum(out_j[:, i] != 0) > 2:
            for k in range(out_j.shape[0]):
                if out_j[k, i] > 0:
                    x = int(out_j[k, i] * col_sample_w * IMG_W / 800) - 1
                    y = int(IMG_H * (ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - k] / 288)) - 1
                    cv2.circle(black_background, (x, y), 5, (0, 255, 0), -1)

    # ----- YOLOv8 Segmentation -----
    yolo_results = yolo_model(rgb_image, verbose=False, device=device, conf=0.1)
    for result in yolo_results:
        if result.boxes is not None and len(result.boxes) > 0:
            # print(f"Detected {len(result.boxes)} objects")
            if result.masks is not None:
                # print(f"Found {len(result.masks)} masks")
                masks = result.masks.data.cpu().numpy()  # Shape: (N, H, W), e.g., (N, 384, 640)
                for mask in masks:
                    # Resize mask to match black_background (360x640)
                    mask_resized = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
                    color = (0, 0, 255)  # Red
                    color_mask = np.zeros_like(black_background, dtype=np.uint8)
                    color_mask[mask_resized.astype(bool)] = color
                    black_background = cv2.addWeighted(black_background, 1.0, color_mask, 0.5, 0)
                    
    return black_background
