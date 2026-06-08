import cv2
import mediapipe as mp
import joblib
import numpy as np
import os

model_filename = 'model_tangan.pkl'

# Periksa apakah model sudah dilatih
if not os.path.exists(model_filename):
    print(f"Error: Model '{model_filename}' tidak dapat ditemukan.")
    print("Silakan jalankan 'train_model.py' terlebih dahulu.")
    exit()

print("Memuat model...")
model = joblib.load(model_filename)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

print("Mulai Program Utama. Tekan 'q' pada jendela video untuk keluar.")

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Proses frame menggunakan MediaPipe
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar keypoints di atas tangan
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Ekstrak 63 koordinat
                landmark_list = []
                for lm in hand_landmarks.landmark:
                    landmark_list.extend([lm.x, lm.y, lm.z])
                
                # Transformasi ke numpy array (1, 63)
                input_data = np.array([landmark_list])
                
                # Prediksi label menggunakan model yang dilatih
                prediction = model.predict(input_data)
                label = str(prediction[0])
                
                # Tambahkan teks hasil prediksi di layar
                cv2.putText(image, f"Pola Prediksi: Kelas {label}", (10, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

        cv2.imshow('Deteksi Pola Tangan Real-time', image)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()