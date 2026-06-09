import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

# Trained model filename
model_filename = 'hand_model.pkl'

if not os.path.exists(model_filename):
    print(f"Error: Model '{model_filename}' not found.")
    print("Please run 'train_model.py' first to train the model.")
    exit()

print("Loading model...")
# Load the trained model
model = joblib.load(model_filename)

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

print("Starting Main Program. Press 'q' on the video window to exit.")

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Flip the image horizontally for a selfie-view display
        image = cv2.flip(image, 1)
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and detect hand landmarks
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks and connections
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Extract x, y, z coordinates for all 21 hand landmarks
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])
                
                # Transform into numpy array (1, 63) suitable for model input
                input_data = np.array([landmark_list])
                
                # Dictionary mapping for output labels (can be edited as needed)
                label_map = {
                    '0': 'Zero',
                    '1': 'One',
                    '2': 'Two',
                    '3': 'Three',
                    '4': 'Four',
                    '5': 'Five',
                    '6': 'Thumbs Up',
                    '7': 'Thumbs Down',
                    '8': 'Metal',
                    '9': 'Nice'
                }
                
                # Predict the label using the trained model
                prediction = model.predict(input_data)
                raw_label = str(prediction[0])
                
                # Get the name from the mapping, if not found display the raw label
                display_label = label_map.get(raw_label, f"Class {raw_label}")
                
                # Add the prediction text to the screen
                cv2.putText(image, f"Prediction: {display_label}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Display the video feed
        cv2.imshow('Real-time Hand Gesture Detection', image)
        
        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()