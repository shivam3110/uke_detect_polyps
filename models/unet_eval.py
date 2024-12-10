import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw
import pandas as pd
from pathlib import Path
import os

# Define paths (Ensure these are correctly set up for your Flask application)
BASE_DIR = Path('/mnt/d/ml_projects/ukw_detect_polyps')
UPLOAD_DIR = BASE_DIR / 'data' / 'uploads'
FRAME_SAVE_DIR = BASE_DIR / 'data' / 'preprocessed_frames'
ANNOTATED_FRAME_DIR = BASE_DIR / 'data' / 'annotated_frames'
RESULT_DIR = BASE_DIR / 'data' / 'result_dataframe'


# Make sure directories exist
os.makedirs(FRAME_SAVE_DIR, exist_ok=True)
os.makedirs(ANNOTATED_FRAME_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def get_latest_video(directory):
    """Get the most recently uploaded video file from the directory."""
    video_files = list(directory.glob('*.mp4'))  # Assuming videos are in mp4 format
    latest_video = max(video_files, key=os.path.getctime, default=None)
    return latest_video

def preprocess_video(video_path, output_folder, panel_start_x, frame_rate=1):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps / frame_rate)
    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            cropped_frame = frame[:, :panel_start_x]
            height, width = cropped_frame.shape[:2]
            center = (width // 2, height // 2)
            radius = min(center[0], center[1]) - 50
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, center, radius, 255, -1)
            masked_frame = cv2.bitwise_and(cropped_frame, cropped_frame, mask=mask)
            output_path = os.path.join(output_folder, f"frame_{saved_count:04d}.jpg")
            cv2.imwrite(output_path, masked_frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Processed {saved_count} frames and saved them to {output_folder}.")

#TODO

def load_model():
    """Load the pre-trained U-Net model"""
    model = smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1)
    model.eval()
    return model

# Image preprocessing specific for U-Net
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to fit U-Net expectations
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization for ResNet
    ])
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0)  # Add a batch dimension

def segment_and_save(image_path, model):
    image_tensor = preprocess_image(image_path)
    with torch.no_grad():
        output = model(image_tensor)
        output = output.squeeze().cpu().numpy()  # Convert PyTorch tensor to numpy array
    output_image = (output > 0.5).astype(np.uint8)  # Threshold the output to obtain binary mask
    output_path = Path(ANNOTATED_FRAME_DIR) / f"annotated_unet_{image_path.name}"
    cv2.imwrite(str(output_path), output_image * 255)  # Save the binary mask as an image
    return output_path

def process_frames(folder_path, model):
    results = []
    images = list(Path(folder_path).glob('*.jpg'))
    for image_path in images:
        output_path = segment_and_save(image_path, model)
        results.append({"Image": image_path.name, "Segmentation_Result": output_path})
    return pd.DataFrame(results)


def main():
    latest_video = get_latest_video(UPLOAD_DIR)
    if not latest_video:
        print("No video found.")
        return
    print(f"Processing video: {latest_video}")
    preprocess_video(latest_video, FRAME_SAVE_DIR, panel_start_x=1500)
    model = load_model()
    results_df = process_frames(FRAME_SAVE_DIR, model)
    results_df.to_csv(RESULT_DIR / 'result_df_unet.csv', index=False)
    print(f"Results saved to {RESULT_DIR / 'result_df_unet.csv'}")
    print("Processing complete.")
    return results_df


if __name__ == "__main__":
    df_results = main()
    print(df_results)