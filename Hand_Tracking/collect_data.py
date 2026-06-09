import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open the webcam
cap = cv2.VideoCapture(0)

# Dataset filename
csv_file = 'hand_dataset.csv'

print("="*50)
print("HAND GESTURE DATA COLLECTOR")
print("1. Show hand gesture in front of the camera.")
print("2. Press number keys '0' to '9' to record data with that specific class/label.")
print("3. Press 'q' to exit.")
print("="*50)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Failed to open camera.")
            break

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        # Convert the BGR image to RGB before processing
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hand landmarks
        results = hands.process(image_rgb)
        
        # Read keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        # Exit if 'q' is pressed
        if key == ord('q'):
            break

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # If a number key (0-9) is pressed, save the coordinates to the CSV file
                if ord('0') <= key <= ord('9'):
                    landmark_list = []
                    # Extract x, y, z coordinates for all 21 hand landmarks
                    for lm in hand_landmarks.landmark:
                        landmark_list.extend([lm.x, lm.y, lm.z])
                    
                    label = chr(key)
                    row = [label] + landmark_list
                    
                    # Append the recorded data into the CSV file
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    
                    print(f"-> Data with label '{label}' successfully saved!")
                    
        # Display the video feed
        cv2.imshow('Collecting Data - Press 0-9 to record, q to exit', image)

# Release resources
cap.release()
cv2.destroyAllWindows()