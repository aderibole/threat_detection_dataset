import os
import cv2

def extract_frames_from_videos(video_dir, output_dir, every_n_frames=30):
    """
    Extracts frames from videos and saves them to output_dir, one every `every_n_frames` frames.
    """
    os.makedirs(output_dir, exist_ok=True)
    # video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    frame_count = 0

    # for video_file in video_files:
    video_file = '09152008flight2tape1_1.mpg'
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    if not cap.isOpened():
        print(f"Error opening video {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every_n_frames == 0:
            filename = f"{os.path.splitext(video_file)[0]}_frame{frame_idx}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            frame_count += 1

        frame_idx += 1

    cap.release()

    print(f"✅ Extracted {frame_count}")

# Run this:
extract_frames_from_videos(
    video_dir='C:/Users/jlrab/OneDrive/Documents/threat_detection_dataset',
    output_dir='threat_captioning_dataset/frames',
    every_n_frames=30  # every 1 second for 30fps video
)