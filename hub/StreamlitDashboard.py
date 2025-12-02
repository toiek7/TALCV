import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import tempfile
import os
import pandas as pd
import time
Model_path = "bestModel1.pt"  # Path to your YOLOv8 model
model = YOLO(Model_path)

st.title("YOLOv8 Demo")


class_id_to_name = model.names if hasattr(model, 'names') else {0: 'person', 1: 'sewing machine', 2: 'garment holder'}

media_type = st.radio("Choose media type", ("Image", "Video"))

if media_type == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        np_image = np.array(image)
        results = model(np_image)
        result_img = results[0].plot()
        st.image(result_img, caption='Detection Result', use_column_width=True)
        # Gather detection data
        boxes = results[0].boxes
        detection_list = []
        for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
            class_name = class_id_to_name.get(int(cls_id), str(cls_id))
            detection_list.append({'Class': class_name, 'Confidence Score': float(conf)})

        # Show table (per this image)
        if detection_list:
            st.subheader("Detections & Confidence")
            df_detection = pd.DataFrame(detection_list)
            st.table(df_detection)

            avg_table = df_detection.groupby('Class')['Confidence Score'].mean().reset_index()
            avg_table.columns = ['Class', 'Average Confidence Score']
            st.subheader("Average class Confidence Score (image)")
            st.table(avg_table)
        else:
            st.info("No objects detected.")
else:
    uploaded_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        tfile.close()  # <--- Make sure to CLOSE the temp file
        video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        stframe = st.empty()
        
        # For avg accuracy
        total_scores = {}
        total_count = {}

        frame_count = 0
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1
                results = model(frame)
                annotated_frame = results[0].plot()
                stframe.image(annotated_frame, channels="BGR")
                # Get detection info for this frame
                boxes = results[0].boxes
                
                for cls_id, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
                    class_name = class_id_to_name.get(int(cls_id), str(cls_id))
                    
                    # Accumulate for averaging
                    if class_name not in total_scores:
                        total_scores[class_name] = 0.0
                        total_count[class_name] = 0
                    total_scores[class_name] += float(conf)
                    total_count[class_name] += 1
                    print(f"Frame {frame_count}: Detected {class_name} with confidence {conf:.2f}")
                if frame_count >= max_frames:  # <--- This ensures we break after last frame, even if cap.isOpened() is still True
                    break
        finally:
            cap.release()
            # add a tiny sleep to let Windows release the video handle
            time.sleep(0.1)
        # After processing, try to remove the video file
        try:
            os.remove(video_path)
        except PermissionError:
            st.warning("Could not delete the temporary video file. Please remove it manually later.")
        except Exception as e:
            st.warning(f"Could not delete temp video: {e}")
        # After the loop, show average accuracy per class
        avg_data = []
        monitor_classes = ['Person', 'Sewing machine', 'Garment Holder']
        for class_name in monitor_classes:
            if class_name in total_scores:
                avg_score = total_scores[class_name] / total_count[class_name]
                avg_data.append({'Class': class_name, 'Average Confidence Score': avg_score})
            else:
                avg_data.append({'Class': class_name, 'Average Confidence Score': None})

        # txt_filename = "results.txt"
        # # Build the text block exactly as we would write
        # result_block = f"\n===== Results for video: {uploaded_file.name} =====\n"
        # result_block += f"{'Class':<20} {'Average Confidence Score':<20}\n"
        # result_block += "-" * 40 + "\n"
        # for row in avg_data:
        #     result_block += f"{row['Class']:<20} {str(row['Average Confidence Score']):<20}\n"
        # result_block += "-" * 40 + "\n"
        # result_block += f"Model used: {Model_path}\n"

        csv_filename = "results.csv"
        
        # --- Build a single row with required headers ---
        row_data = {
            'Video': uploaded_file.name,
            'Model': Model_path,
            'Person': None,
            'Sewing machine': None,
            'Garment Holder': None
        }
        # Fill in scores for each class, as available
        for row in avg_data:
            key = row['Class']  # e.g., "Person"
            if key in row_data:
                row_data[key] = row['Average Confidence Score']
        
        df_row = pd.DataFrame([row_data])

        # --- Write or append, preventing duplicates ---
        should_write = True
        if os.path.exists(csv_filename):
            if os.stat(csv_filename).st_size == 0:
                # File exists but empty
                prev = None
            else:
                try:
                    prev = pd.read_csv(csv_filename)
                except Exception as e:
                    st.error(f"Failed to read {csv_filename}: {e}")
                    prev = None
            if prev is not None:
                # Check if this video-model combo already exists
                mask = (prev['Video'] == row_data['Video']) & (prev['Model'] == row_data['Model'])
                if mask.any():
                    should_write = False

        if should_write:
            # Write as new if not exists, else append (no index)
            df_row.to_csv(csv_filename, mode='a', header=not os.path.exists(csv_filename), index=False)
            st.success(f"Saved average results to {csv_filename}")
        else:
            st.info(f"Results for this video/model already in {csv_filename}; not writing duplicate.")

        # Show result as table still
        st.subheader("Average Confidence Score (video)")
        st.table(df_row)
        # Read old file, if exists
        # should_write = True
        # if os.path.exists(txt_filename):
        #     with open(txt_filename, "r") as f:
        #         content = f.read()
        #     if result_block in content:
        #         should_write = False

        # if should_write:
        #     with open(txt_filename, "a") as f:
        #         f.write(result_block)
        #     st.success(f"Saved average results to {txt_filename}")
        # else:
        #     st.info(f"Results for this video/model are already in {txt_filename}; not writing duplicate.")