import cv2
import os

# Path to your input video
video_dir = f"C:/Users/charoensupthawornt/Work/Securade.ai/hub/videos/train_3/new"    # <-- Change this to your video filename

# Directory where frames will be saved
output_dir = "C:/Users/charoensupthawornt/Work/Securade.ai/hub/images/train/new"
os.makedirs(output_dir, exist_ok=True)

# Get list of all mp4 video files in the directory
video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]

if not video_files:
    print(f"No .mp4 files found in {video_dir}")
    exit(1)


for video_basename in video_files:
    video_path = os.path.join(video_dir, video_basename)
    if not os.path.exists(video_path): #check file exist
        print(f"Error: Video file not found at {video_path}")
        continue
    # Open the video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while cap.isOpened() and saved_count < 5: # change number of saved frames here
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        # Save every 30th frame
        if frame_count % 90 == 0:
            frame_filename = os.path.join(output_dir, f"{video_basename}_frame_{frame_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Saved {saved_count} frames out of {frame_count} total frames.")
