import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# ==========================================================
# Download NLTK Resource (hanya pertama kali)
# ==========================================================

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# ==========================================================
# Membaca Dataset
# ==========================================================

print("="*60)
print("MEMBACA DATASET")
print("="*60)

df = pd.read_csv("dataset/customer_support_tickets.csv")

print("\nJumlah Data :", len(df))
print("\nKolom Dataset :")
print(df.columns)

# ==========================================================
# Mengambil Kolom yang Digunakan
# ==========================================================

documents = df["Ticket Description"]

# ==========================================================
# Stopword Bahasa Inggris
# ==========================================================

stop_words = set(stopwords.words("english"))

# ==========================================================
# Fungsi Preprocessing
# ==========================================================

def preprocess(text):

    # jika kosong
    if pd.isna(text):
        return ""

    # ubah ke string
    text = str(text)

    # lowercase
    text = text.lower()

    # hapus url
    text = re.sub(r'http\S+', '', text)

    # hapus angka
    text = re.sub(r'\d+', '', text)

    # hapus tanda baca
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # tokenizing
    tokens = word_tokenize(text)

    # stopword removal
    tokens = [word for word in tokens if word not in stop_words]

    # gabungkan lagi
    text = " ".join(tokens)

    return text

# ==========================================================
# Menjalankan Preprocessing
# ==========================================================

print("\nMelakukan preprocessing...")

df["processed_text"] = documents.apply(preprocess)

print("Selesai!")

# ==========================================================
# Menampilkan Contoh Hasil
# ==========================================================

print("\n")
print("="*60)
print("CONTOH HASIL PREPROCESSING")
print("="*60)

for i in range(5):

    print("\nData ke-", i+1)

    print("\nSebelum :")
    print(df["Ticket Description"].iloc[i])

    print("\nSesudah :")
    print(df["processed_text"].iloc[i])

# ==========================================================
# Menghapus Data Kosong
# ==========================================================

df = df[df["processed_text"] != ""]

print("\nJumlah Data Setelah Cleaning :", len(df))

# ==========================================================
# Menyimpan Dataset Baru
# ==========================================================

output_file = "dataset/clean_dataset.csv"

df.to_csv(output_file, index=False)

print("\nDataset berhasil disimpan!")

print(output_file)

print("\nPreprocessing selesai.")