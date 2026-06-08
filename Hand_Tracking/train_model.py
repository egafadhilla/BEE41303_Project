import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

csv_file = 'dataset_tangan.csv'

if not os.path.exists(csv_file):
    print(f"Error: File '{csv_file}' tidak ditemukan!")
    print("Silakan jalankan 'collect_data.py' terlebih dahulu untuk mengumpulkan data.")
    exit()

print("Memuat data dataset...")
data = pd.read_csv(csv_file, header=None)

# Kolom 0 adalah label (kelas dari 0-9), kolom sisanya adalah fitur koordinat (x, y, z)
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

print(f"Total data: {len(data)}")

# Bagi data menjadi data latih (80%) dan data uji (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nSedang melatih model (Support Vector Machine / Random Forest)...")
# Menggunakan Support Vector Classifier (SVC) - model ringan dan sangat cocok untuk klasifikasi pola dan Jetson Nano
model = SVC(kernel='linear', probability=True)

# Alternatif: Random Forest
# model = RandomForestClassifier(n_estimators=100)

# Latih model dengan data
model.fit(X_train, y_train)

# Lakukan prediksi pada data uji
y_pred = model.predict(X_test)

# Hitung akurasi
accuracy = accuracy_score(y_test, y_pred)
print(f"-> Akurasi model: {accuracy * 100:.2f}%")

# Simpan model agar bisa digunakan ulang tanpa training lagi
model_filename = 'model_tangan.pkl'
joblib.dump(model, model_filename)
print(f"Model berhasil disimpan sebagai '{model_filename}'.")
print("Sekarang Anda dapat menjalankan 'main.py' untuk mendeteksi gestur.")