import os
import cv2
import sys
import torch
import logging
import numpy as np
import scipy.special
from PIL import Image
from ultralytics import YOLO
sys.path.append('/home/kaustubh/Documents/GitHub/Self_Driving_Car/Imitation_Learning_RL')
sys.path.append('/home/kaustubh/Documents/GitHub/Self_Driving_Car/Imitation_Learning_RL/model')
sys.path.append('/home/kaustubh/Documents/GitHub/Self_Driving_Car/Imitation_Learning_RL/model')
from model.model import parsingNet
from torchvision import transforms
from ultralytics.utils import LOGGER
from data.constant import tusimple_row_anchor

# Suppress YOLO logs
LOGGER.setLevel(logging.ERROR)



IMG_W, IMG_H = 128, 128 
GRIDDING_NUM = 100
CLS_NUM_PER_LANE = 56
ROW_ANCHOR = tusimple_row_anchor
BACKBONE = '18'
LANE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'tusimple_18.pth')
YOLO_MODEL_PATH = 'models/yolo11m-seg'  

# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"

#  Load LaneNet 
lane_net = parsingNet(pretrained=False, backbone=BACKBONE,
                      cls_dim=(GRIDDING_NUM + 1, CLS_NUM_PER_LANE, 4),
                      use_aux=False).to(device)
state_dict = torch.load(LANE_MODEL_PATH, map_location='cpu')['model']
lane_net.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}, strict=False)
lane_net.eval()

#  Load YOLO 
yolo_model = YOLO(YOLO_MODEL_PATH)

#  Transforms 
img_transforms = transforms.Compose([
    transforms.Resize((288, 800)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def process_frame(rgb_image: np.ndarray) -> np.ndarray:
    rgb_image = cv2.resize(rgb_image, (IMG_W, IMG_H))  # 640x360
    output_image = rgb_image.copy()  # <<- FIX: Use this instead of black background

    # -- Lane Detection --
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
                    cv2.circle(output_image, (x, y), 5, (0, 255, 0), -1)

    # -- YOLOv8 Segmentation --
    yolo_results = yolo_model(rgb_image, verbose=False, device=device, conf=0.1)
    # for result in yolo_results:
    #     if result.boxes is not None and len(result.boxes) > 0:
    #         if result.masks is not None:
    #             masks = result.masks.data.cpu().numpy()  # Shape: (N, H, W)
    #             for mask in masks:
    #                 mask_resized = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    #                 color = (0, 0, 255)  # Red
    #                 color_mask = np.zeros_like(output_image, dtype=np.uint8)
    #                 color_mask[mask_resized.astype(bool)] = color
    #                 output_image = cv2.addWeighted(output_image, 1.0, color_mask, 0.5, 0)
    
    for result in yolo_results:
        if result.boxes is not None and len(result.boxes) > 0:
            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()  # Shape: (N, H, W)
                for mask in masks:
                    mask_resized = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
                    output_image[mask_resized.astype(bool)] = (0, 0, 255)  # Set to red in BGR, blue in RGB

    return output_image



# def process_frame(rgb_image: np.ndarray) -> np.ndarray:
#     """
#     Process raw RGB frame:
#     - Detect lanes using LaneNet
#     - Detect objects using YOLOv8-seg
#     - Draw both on black background
#     - Return the single processed frame (uint8, shape=(360, 640, 3))
#     """
#     rgb_image = cv2.resize(rgb_image, (IMG_W, IMG_H))  # 640x360
#     black_background = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)  # 360x640x3

#     # -- Lane Detection --
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

#     for i in range(out_j.shape[1]):
#         if np.sum(out_j[:, i] != 0) > 2:
#             for k in range(out_j.shape[0]):
#                 if out_j[k, i] > 0:
#                     x = int(out_j[k, i] * col_sample_w * IMG_W / 800) - 1
#                     y = int(IMG_H * (ROW_ANCHOR[CLS_NUM_PER_LANE - 1 - k] / 288)) - 1
#                     cv2.circle(black_background, (x, y), 5, (0, 255, 0), -1)

#     # -- YOLOv8 Segmentation --
#     yolo_results = yolo_model(rgb_image, verbose=False, device=device, conf=0.1)
#     for result in yolo_results:
#         if result.boxes is not None and len(result.boxes) > 0:
#             # print(f"Detected {len(result.boxes)} objects")
#             if result.masks is not None:
#                 # print(f"Found {len(result.masks)} masks")
#                 masks = result.masks.data.cpu().numpy()  # Shape: (N, H, W), e.g., (N, 384, 640)
#                 for mask in masks:
#                     # Resize mask to match black_background (360x640)
#                     mask_resized = cv2.resize(mask, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
#                     color = (0, 0, 255)  # Red
#                     color_mask = np.zeros_like(black_background, dtype=np.uint8)
#                     color_mask[mask_resized.astype(bool)] = color
#                     black_background = cv2.addWeighted(black_background, 1.0, color_mask, 0.5, 0)
                    
#     return black_background
