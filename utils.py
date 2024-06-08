import os
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
import imageio 
from IPython.display import HTML
import base64


def save_image_with_ruler(image_path, save_dir, save_name):
    # Load the image
    image = Image.open(image_path)
    
    # Plot the image with rulers
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.xticks(ticks=range(0, image.width, 100))
    plt.yticks(ticks=range(0, image.height, 100))
    plt.grid(True, which='both', color='gray', linestyle='--', linewidth=0.5)
    
    # Save the image with rulers
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path)
    plt.close()


def add_rulers_to_image(src_image_pth, to_save=False):
    src_image = Image.open(src_image_pth)
    image_np = np.array(src_image)
    height, width, _ = image_np.shape
    
    # Add rulers on the top
    for i in range(0, width, width // 10):
        cv2.line(image_np, (i, 0), (i, 20), (0, 255, 0), 1)
        cv2.putText(image_np, str(i), (i, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    
    # Add rulers on the left
    for i in range(0, height, height // 10):
        cv2.line(image_np, (0, i), (20, i), (0, 255, 0), 1)
        cv2.putText(image_np, str(i), (5, i + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2, cv2.LINE_AA)

    # Add "Left" and "Right" text
    left_text = "Left"
    right_text = "Right"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 3

    # Text sizes
    left_text_size = cv2.getTextSize(left_text, font, font_scale, font_thickness)[0]
    right_text_size = cv2.getTextSize(right_text, font, font_scale, font_thickness)[0]

    # Left text position
    left_text_x = 10
    left_text_y = 40

    # Right text position
    right_text_x = width - right_text_size[0] - 10
    right_text_y = 40

    # Add red border for "Left"
    cv2.putText(image_np, left_text, (left_text_x, left_text_y), font, font_scale, (0, 0, 255), font_thickness + 2, cv2.LINE_AA)
    # Add blue text for "Left"
    cv2.putText(image_np, left_text, (left_text_x, left_text_y), font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)

    # Add red border for "Right"
    cv2.putText(image_np, right_text, (right_text_x, right_text_y), font, font_scale, (0, 0, 255), font_thickness + 2, cv2.LINE_AA)
    # Add blue text for "Right"
    cv2.putText(image_np, right_text, (right_text_x, right_text_y), font, font_scale, (255, 0, 0), font_thickness, cv2.LINE_AA)

    image_with_rulers = Image.fromarray(image_np)
    if to_save:
        # Save the image with rulers
        dir_name, file_name = os.path.split(src_image_pth)
        file_base, file_ext = os.path.splitext(file_name)
        new_file_name = f"{file_base}_With_Rulers{file_ext}"
        new_file_path = os.path.join(dir_name, new_file_name)
        image_with_rulers.save(new_file_path)
        print(f"Image with rulers saved at: {new_file_path}")

    return image_with_rulers



def calculate_valid_step_value_from_image_dimensions(image_path):
    # Load the image
    img = cv2.imread(image_path)
    
    # Get the dimensions of the image
    height, width = img.shape[:2]
    
    # Determine the larger dimension
    larger_dimension = max(width, height)
    
    # Calculate the value
    calculated_value = round(larger_dimension / 30)
    
    return calculated_value



def create_iteration_video(image_pth, num_iterations=10, duration=5):
    # Define the directory and filename for saving the video
    image_name = os.path.splitext(os.path.basename(image_pth))[0]
    final_results_dir = os.path.join("FinalResults", image_name)
    os.makedirs(final_results_dir, exist_ok=True)
    video_filename = os.path.join(final_results_dir, "iterations_video.mp4")
    
    # Collect all iteration images
    images = []
    for i in range(1, num_iterations + 1):
        iter_image_path = f"InProcessResultImages/{image_name}/iter_{i}_predicted_bbox.jpeg"
        if os.path.exists(iter_image_path):
            images.append(imageio.imread(iter_image_path))
        else:
            print(f"Image not found: {iter_image_path}")
    
    # Calculate the frame duration
    frame_duration = duration / len(images)
    
    # Create video from images
    imageio.mimwrite(video_filename, images, format='mp4', fps=1/frame_duration)
    print(f"Video saved at: {video_filename}")




def play_video(video_path):
    # Read video file and encode it in base64
    video = open(video_path, "rb").read()
    video_encoded = base64.b64encode(video).decode("ascii")
    
    # Create HTML to display video
    video_html = f'''
    <video width="600" controls>
        <source src="data:video/mp4;base64,{video_encoded}" type="video/mp4">
        Your browser does not support the video tag.
    </video>
    '''
    return HTML(video_html)


