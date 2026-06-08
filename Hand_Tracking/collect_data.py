import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Buka kamera webcam
cap = cv2.VideoCapture(0)

# Nama file dataset
csv_file = 'dataset_tangan.csv'

print("="*50)
print("ALAT PENGUMPUL DATA POLA TANGAN")
print("1. Tunjukkan gestur/pola tangan ke depan kamera.")
print("2. Tekan tombol angka '0' sampai '9' pada keyboard untuk merekam data dengan label/kelas tersebut.")
print("3. Tekan 'q' untuk keluar.")
print("="*50)

with mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Gagal membuka kamera.")
            break

        # Flip image secara horizontal agar seperti cermin
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Proses deteksi landmark tangan
        results = hands.process(image_rgb)
        
        # Baca input keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Gambar titik dan garis di tangan
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Jika tombol angka ditekan, simpan koordinat ke CSV
                if ord('0') <= key <= ord('9'):
                    landmark_list = []
                    # Ada 21 landmark, masing-masing punya x, y, z
                    for lm in hand_landmarks.landmark:
                        landmark_list.extend([lm.x, lm.y, lm.z])
                    
                    label = chr(key)
                    row = [label] + landmark_list
                    
                    with open(csv_file, mode='a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow(row)
                    
                    print(f"-> Data with label '{label}' succesfully saved!")
                    
        cv2.imshow('Collecting Data - Press 0-9 for record, q for exit', image)

cap.release()
cv2.destroyAllWindows()