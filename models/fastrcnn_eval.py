import cv2
import numpy as np
import torch
from torchvision import models, transforms
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

def load_model():
    """Load a pre-trained Faster R-CNN model."""
    model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose([transforms.ToTensor()])
    image = Image.open(image_path).convert("RGB")
    return transform(image), image

def detect_objects(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        predictions = model([image_tensor])[0]
    filtered_indices = [i for i, score in enumerate(predictions['scores']) if score > threshold]
    return predictions['boxes'][filtered_indices], predictions['labels'][filtered_indices], predictions['scores'][filtered_indices]


def draw_boxes(image, boxes, labels, scores):
    """Draw bounding boxes and labels on an image."""
    draw = ImageDraw.Draw(image)
    for box, label, score in zip(boxes, labels, scores):
        draw.rectangle([(box[0], box[1]), (box[2], box[3])], outline="red", width=3)
        draw.text((box[0], box[1]), f'{label} {score:.2f}', fill="red")
    return image


def process_frames(folder_path, model, threshold=0.5):
    results = []
    for image_filename in os.listdir(folder_path):
        image_path = Path(folder_path) / image_filename
        image_tensor, original_image = preprocess_image(image_path)
        boxes, labels, scores = detect_objects(model, image_tensor, threshold)
        annotated_image = draw_boxes(original_image.copy(), boxes, labels, scores)
        annotated_image_path = Path(ANNOTATED_FRAME_DIR) / f"annotated_fastrcnn{image_filename}"
        annotated_image.save(annotated_image_path)
        results.append({
            "Image": annotated_image_path.name,
            "Boxes": [list(box) for box in boxes],
            "Labels": [label.item() for label in labels],
            "Scores": [score.item() for score in scores]
        })
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
    results_df.to_csv(RESULT_DIR / 'result_df_fastrcnn.csv', index=False)
    print(f"Results saved to {RESULT_DIR / 'result_df_fastrcnn.csv'}")
    print("Processing complete.")
    return results_df


if __name__ == "__main__":
    df_results = main()
    print(df_results)
