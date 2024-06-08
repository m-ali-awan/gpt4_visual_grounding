# here we will have fns related to VLM(right now model/s from Openai)

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

def encode_image_in_memory(image):
    """
    Encodes a PIL Image to a base64 string without saving it as a file.

    :param image: PIL Image object
    :return: Base64-encoded string of the image
    """
    img_buffer = BytesIO()
    image.save(img_buffer, format='PNG')  # You can change JPEG to the appropriate format (e.g., PNG) as needed
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')  # Convert bytes to base64 string and decode to UTF-8
    return base64_str

async def get_vg_gpt4o_response(api_key, image_with_grounding, system_message):
    try_count = 0
    max_retries = 1  # Allow for 1 retry
    timeout_duration = 90  # 15 seconds timeout for the HTTP request
    while try_count <= max_retries:
        base64_image = encode_image_in_memory(image_with_grounding)
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "gpt-4o",
            "seed": 1420413216,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}]
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Give me [X1,Y1,WIDTH,HEIGHT] coordinates of bounding-box for eyeglasses in this image. Bounding-Box should cover the eyeglasses fully, and \
                         should be tightly bounded."},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ]
        }
        async with httpx.AsyncClient(timeout=timeout_duration) as client:
            try:
                response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                if response.status_code == 200:
                    data = response.json()
                    return data
                else:
                    raise Exception("Request failed with status code {}".format(response.status_code))
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                print(f"Attempt {try_count + 1}: {e}")
                if try_count < max_retries:
                    await asyncio.sleep(1)  # Non-blocking sleep before retrying
                try_count += 1
    return None




def save_labeled_image_with_xywh(results_dict, img_path, to_save_dir, to_save_name):
    """
    Saves an image with bounding boxes and labels based on input dictionary, in the same directory as the input image.

    Args:
        results_dict (dict): Dictionary containing object labels and coordinates.
            Format: {label: [x1, y1, width, height], ...}
        img_path (str): Path to the input image.
        to_save_dir : can be "InProcessResultImages" or "FinalResults"
        to_save_name : can be "1st_predicted_bbox" or "Final_predicted_bbox"
    """

    # Load image
    img = cv2.imread(img_path)
    height, width, _ = img.shape

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    text_color = (255, 0, 0)  # Blue color in BGR
    thickness = 1  # For bold text

    # Iterate over objects and draw bounding boxes
    for label, coords in results_dict.items():
        x1, y1, w, h = coords
        x2 = x1 + w
        y2 = y1 + h

        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), thickness)

        # Write coordinates on the edges of the rectangle
        cv2.putText(img, f'({x1}, {y1})', (x1, y1 - 10), font, font_scale, text_color, thickness)
        cv2.putText(img, f'({x2}, {y1})', (x2, y1 - 10), font, font_scale, text_color, thickness)
        cv2.putText(img, f'({x1}, {y2})', (x1, y2 + 20), font, font_scale, text_color, thickness)
        cv2.putText(img, f'({x2}, {y2})', (x2, y2 + 20), font, font_scale, text_color, thickness)

        # Write class name on top-middle of the rectangle
        text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
        #text_x = x1 + (w - text_size[0]) // 2
        #text_y = y1 - 10

        # Ensure text is within image bounds
        #if text_y < 0:
        text_y = y1 + text_size[1] + 10  # Move text inside the rectangle

        #if text_x < 0:
        text_x = x1 + 5  # Move text slightly right inside the rectangle
        if text_x + text_size[0] > width:
            text_x = x2 - text_size[0] - 5  # Move text slightly left inside the rectangle

        cv2.putText(img, label, (text_x, text_y), font, font_scale, text_color, thickness)

    # Construct output path in the same directory as the input image
    img_dir, img_filename = os.path.split(img_path)
    output_folder = os.path.join(to_save_dir, os.path.splitext(img_filename)[0])
    os.makedirs(output_folder, exist_ok=True)
    output_filename = f"{to_save_name}.jpeg" 
    output_path = os.path.join(output_folder, output_filename)
    cv2.imwrite(output_path, img)
    print(f"Labeled image saved to: {output_path}")

    return output_path



async def bounding_box_validator(api_key,object_class, prediction_string, src_image_pth):
    
    
    good_example_response = """
            ```json
{
  "accurate": "no",
  "reason": "The predicted bounding box is partially covering the eyeglasses, and extends too far to the right. To adjust: Shift X1 to the right by approximately 30 pixels (new X1= X1 + 30), shift Y1 down by approximately 10 pixels (new Y1 = Y1 + 10), increase the width by approximately 20 pixels, and reduce the height by approximately 20 pixels."
}
    """
    bad_example_response = """```json
{
    "accurate": "no",
    "reason": "The bounding box is wider than necessary as it includes parts of the person's face and hair. To make it fit more tightly around the eyeglasses, adjust the width and height of the bounding box."
}
```"""
    step_value = calculate_valid_step_value_from_image_dimensions(src_image_pth)
    print(f"Step value is :::{step_value}")
    prompt = f"""
        You are an Object Detection Validation workflow for class : {object_class}. You have to check quality of the Predicted Bounding Box for the desired class :{object_class}, and suggest steps(moving X,Y coordinates of Top-Left corner of Bounding-Box, or changing width, height of bounding box) to make it accurate.\
           You have to check if Bounding Box is fully covering the {object_class} and not partially.\n
           Also if bouding Box is too loose, i.e maybe covering full face of a person, you have to squeeze it to cover only eyeglasses.
           
           \n You will get 2 inputs:\n\
        - "predicted_bounding_box_coordinates" : predicted bounding box coordinates dict where values will be in format : [X1,Y1,width,height],  (X1,Y1) are coordinates of Top-Left Corner of Bounding Box. and width and height \
        are horizontal length of bounding box from X1, and vertical length of bounding box from Y1.
        - "visual_result" : The input Image with the bounding box Drawn on it using the predicted bounding box coordinates

        You have to respond in json form, a dict having keys : "accurate", and "reason". \n
        "accurate" can have only two values : "yes", "no"
        # If Bouding box is covering more than 90% of the {object_class}, we will consider it ACCURATE. And if it is a bit loose \
        we will ignore. But, if for example it is partially covering the {object_class}, we have to make Bouding-box biger to fully cover it.

        "reason" : here you will tell reason. Specifically if 'no', you have to tell how to adjust the "predicted_bounding_box_coordinates", i.e shifting X1,Y1 by how much value, \
        changing width or height or maybe both. \n\n

        Good-response example:: 
        {good_example_response} \n
        Bad-response example (As here concrete steps to take are not given ) :: 
        {bad_example_response} \n\n

        **Rulers are drawn on top and left of image, to give you reference of coordinates, so use them to decide how much modification to be made in X1,Y1, and width and height**
        And keep in mind if you have to move rectangle to right you have to ADD to X1, and to move left you have to SUBTRACT from X1.\
        And to move down you have to ADD to Y1, and to move up , SUBTRACT from Y1.
        **Donot take steps greater than {step_value} in anything, so maximum variation suggestion should be +/- {step_value}. This way \
            gradually we will reach perefection.**
        **Also you are not good at understanding LEFT and RIGHT of image. For your help, Image will have 'Left' and 'Right' tags drawn on it, so use it as reference to \
            suggest in which directions to take steps.** \n\n

        Here is the prediction you have to analyze.:: \n
        "predicted_bounding_box_coordinates" : {prediction_string}
        ""visual_result"" :


        
    """
    try_count = 0
    max_retries = 1  # Allow for 1 retry
    while try_count <= max_retries:
        

        src_image = Image.open(src_image_pth)
        src_image_with_rulers = add_rulers_to_image(src_image_pth)
        base64_image = encode_image_in_memory(src_image_with_rulers)

        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json',
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": "gpt-4o",
            "seed": 1420413216,
            "messages": [
              {
                "role": "user",
                "content": [
                  {
                    "type": "text",
                    "text": prompt
                  },
                  {
                    "type": "image_url",
                    "image_url": {
                      "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                  }
                ]
              }
            ],
            #"max_tokens": 4096
        }

        try:
            with requests.post("https://api.openai.com/v1/chat/completions",
                                headers=headers, json=payload, timeout = 20) as response:
                if response.status_code == 200:
                    data = response.json()
                    return data
                else:
                    raise Exception("Request failed with status code {}".format(response.status_code))
        except Exception as e:
            print(f"Attempt {try_count + 1}: {e}")
            if try_count < max_retries:
                time.sleep(3)  # Sleep for 3 seconds before retrying
            try_count += 1
    return None  # Return None if both attempts fail


async def self_checking_response(api_key, raw_image, first_result_image,
                                  system_message, first_vlm_response, review_response):

    try_count = 0
    max_retries = 2  # Allow for 1 retry
    timeout_duration = 90  # 15 seconds timeout for the HTTP request
    base64_raw_image = encode_image_in_memory(raw_image)
    base64_first_result_image = encode_image_in_memory(first_result_image)
    payload = {
        "model": "gpt-4o",
        "seed": 443521423,
        "messages":[
            {
                "role":"system",
                "content":[
                    {
                        "type":"text",
                        "text":system_message
                    }
                ]
            },
            {
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text": "Give me [X1,Y1,WIDTH,HEIGHT] coordinates of bounding-box for eyeglasses in this image. Bounding-Box should cover the eyeglasses fully, and \
                         should be tightly bounded."
                    },
                    {
                        "type":"image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{base64_raw_image}"
                        }
                    }
                ]
            },
            {
                "role":"assistant",
                "content":[
                    {
                        "type":"text",
                        "text": first_vlm_response
                    }
                ]
            },
            {
                "role":"user",
                "content":[
                    {
                        "type":"text",
                        "text" : review_response
                    },
                    {
                        "type":"image_url",
                        "image_url":{
                            "url": f"data:image/jpeg;base64,{base64_first_result_image}"
                        }
                    }
                ]
            },


        ]
    }
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json',
        "Authorization": f"Bearer {api_key}"
    }
    while try_count <= max_retries:
        async with httpx.AsyncClient(timeout=timeout_duration) as client:
            try:
                response = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
                print(response)
                if response.status_code == 200:
                    data = response.json()
                    return data
                else:
                    raise Exception("Request failed with status code {}".format(response.status_code))
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                print(f"Attempt {try_count + 1}: {e}")
                if try_count < max_retries:
                    await asyncio.sleep(1)  # Non-blocking sleep before retrying
                try_count += 1

    return None  # Return None if both attempts fail