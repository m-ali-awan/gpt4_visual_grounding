# Here we will have fns related to Reference Grounding Model(currently YOLOv8 model, can be some other)


import os
import ast
import sys
sys.path.append('../../Creds')   # Here yoyu can use any method, main purpose is to have your OPENAI API Key, as it is requied
from openai_config import OPENAI_API_KEY
import random
import requests
import traceback
import json
from io import BytesIO
import httpx
from aiohttp import ClientSession
import asyncio
import aiohttp
import time
from PIL import Image
import base64
from PIL import Image, ImageDraw, ImageFont
import re
import cv2
from utils import *

def yolo_predict(model,img_pth):
    results = model.predict(img_pth)
    return results[0]

def draw_ref_det_results_on_image(img_pth, results):
    # Load the image
    img = cv2.imread(img_pth)
    height, width, _ = img.shape
    x_pos_calc = int(width/10)
    # Extract xywh in absolute pixel values
    box = results[0].boxes.xywh[0]  # Assuming one box for simplicity
    x_center, y_center, w, h = box

    # Calculate rectangle coordinates
    x1 = int(x_center - w/2)
    y1 = int(y_center - h/2)
    x2 = int(x_center + w/2)
    y2 = int(y_center + h/2)

    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)

    # Add rulers on the top
    for i in range(0, width, width // 10):
        cv2.line(img, (i, 0), (i, 20), (0, 255, 0), 1)
        cv2.putText(img, str(i), (i, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Add rulers on the left
    for i in range(0, height, height // 10):
        cv2.line(img, (0, i), (20, i), (0, 255, 0), 1)
        cv2.putText(img, str(i), (5, i + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    
    # Draw coordinate texts within the rectangle with updated font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    color = (255, 0, 0)  # Blue color in BGR
    thickness = 1  # For bold text
    
    cv2.putText(img, f'({x1}, {y1})', (x1, y1 + 15), font, font_scale, color, thickness)
    cv2.putText(img, f'({x2}, {y1})', (x2 - x_pos_calc, y1 + 15), font, font_scale, color, thickness)
    cv2.putText(img, f'({x1}, {y2})', (x1, y2 - 5), font, font_scale, color, thickness)
    cv2.putText(img, f'({x2}, {y2})', (x2 - x_pos_calc, y2 - 5), font, font_scale, color, thickness)

    # Draw width arrow
    midpoint_y = y1 + (y2 - y1) // 2
    cv2.arrowedLine(img, (x1, midpoint_y), (x2, midpoint_y), color, thickness)

    # Draw height arrow
    midpoint_x = x1 + (x2 - x1) // 2
    cv2.arrowedLine(img, (midpoint_x, y1), (midpoint_x, y2), color, thickness)
    
    # Draw center point coordinates
    center_coordinates = (midpoint_x, midpoint_y)
    cv2.putText(img, f'({center_coordinates[0]}, {center_coordinates[1]})', (center_coordinates[0] - 65, center_coordinates[1] - 10), font, font_scale, color, thickness)
    
    # Save the image with 'image_with_ref_bbox.jpeg' name under the folder named after the input image
    img_dir, img_name = os.path.split(os.path.splitext(img_pth)[0])
    output_folder = os.path.join("ImagesWithReferences", img_name)
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, "image_with_ref_bbox.jpeg")
    cv2.imwrite(output_path, img)
    print(f"Final result image saved to: {output_path}")