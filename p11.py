import cv2
import numpy as np
import torch
import os
from deepface import DeepFace
import pandas as pd
import time

# Load the pre-trained YOLOv5 model from Ultralytics hub
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Get the list of class labels used by YOLOv5
class_labels = model.names

# Initialize attendance list
attendance_list = []

# Function to load known face images from the specified directory
def load_known_faces(known_faces_dir):
    """
    This function loads face images and their corresponding names from the given directory.
    """
    known_faces = ["./photo1"]  # List to store paths of known face images
    known_names = ["biggan","ontu"]  # List to store names associated with known face images

    if not os.path.exists(known_faces_dir):  # Check if the directory exists
        raise Exception(f"Error: Directory {known_faces_dir} does not exist")

    # Loop over the files in the directory to collect known faces
    for filename in os.listdir(known_faces_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):  # Only accept .jpg and .png files
            image_path = os.path.join(known_faces_dir, filename)
            known_faces.append(image_path)
            known_names.append(os.path.splitext(filename)[0])  # Store the filename as the name (without extension)

    if len(known_faces) == 0:  # Check if any faces were loaded
        raise Exception(f"No face images found in {known_faces_dir}")

    return known_faces, known_names

# Function to perform real-time video feed, object detection, and face recognition
def real_time_object_detection(known_faces, known_names):
    """
    This function opens the webcam video stream, performs YOLO object detection,
    and recognizes faces using DeepFace.
    """
    # Start capturing video from the webcam (ID 0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():  # Ensure the video stream is opened
        raise Exception("Error: Unable to open video stream")

    # Main loop to process the video stream
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:  # Break if frame not captured correctly
            print("Error: Unable to fetch frame")
            break

        # Perform YOLOv5 object detection on the current frame
        results = model(frame)
        detections = results.xyxy[0].cpu().numpy()  # Get detection coordinates and classes

        # Get the current timestamp
        current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())  # Format the current time

        # Loop through detected objects (detections)
        for det in detections:
            x1, y1, x2, y2, conf, class_id = det  # Get bounding box coordinates and class
            class_id = int(class_id)
            label = f"{class_labels[class_id]} {conf:.2f}"  # Create a label with class and confidence

            # Draw bounding box and label for detected objects
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Detect faces in the current frame using DeepFace
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB as DeepFace works on RGB images
        cv2.imwrite("current_frame.jpg", rgb_frame)  # Save the current frame for recognition

        # Loop over known faces and attempt recognition
        for known_face_path, known_name in zip(known_faces, known_names):
            try:
                # Find the known face in the current frame using DeepFace
                result = DeepFace.find(img_path="current_frame.jpg", db_path=known_faces_dir, model_name='VGG-Face', enforce_detection=False)

                if len(result) > 0:  # If a match is found
                    if known_name not in attendance_list:  # Check if attendance for this person has already been logged
                        attendance_list.append(known_name)  # Add the recognized name to the attendance list
                        print(f"Attendance logged for: {known_name}")

                    # Draw a box and name around the recognized face
                    for res in result:
                        top, right, bottom, left = res['facial_area'].values()  # Get bounding box for the face
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)  # Green box for recognized face
                        cv2.putText(frame, known_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error in face recognition: {str(e)}")

        # Display the timestamp on the frame
        cv2.putText(frame, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the video frame
        cv2.imshow("Real-Time Attendance System", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Main function to load known faces, run the real-time object detection, and save attendance
if __name__ == "__main__":
    # Directory containing the images of known faces (update with your actual directory)
    known_faces_dir = "./photo"  # Update this path if necessary

    # Load known faces and names
    known_faces, known_names = load_known_faces(known_faces_dir)

    # Run the real-time object detection and face recognition
    real_time_object_detection(known_faces, known_names)

    # Save the attendance list to an Excel file using pandas
    df = pd.DataFrame(attendance_list, columns=["Names"])
    df.to_excel("attendance.xlsx", index=False)  # Save attendance to 'attendance.xlsx'

    print("Attendance has been successfully saved to attendance.xlsx")
