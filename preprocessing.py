import pandas as pd
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ==========================================================
# Download NLTK Resource (hanya pertama kali)
# ==========================================================

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ==========================================================
# Membaca Dataset
# ==========================================================

print("="*60)
print("MEMBACA DATASET")
print("="*60)

df = pd.read_csv("dataset/customer_support_tickets.csv")

print("\nJumlah Data :", len(df))
print("\nKolom Dataset :")
print(df.columns.tolist())

# ==========================================================
# Informasi Missing Values
# ==========================================================

print("\n")
print("="*60)
print("INFORMASI MISSING VALUES")
print("="*60)

missing = df.isnull().sum()
missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

for col in df.columns:
    if missing[col] > 0:
        print(f"  {col}: {missing[col]} ({missing_pct[col]}%)")

print(f"\nTotal kolom dengan missing value: {(missing > 0).sum()}")

# ==========================================================
# Cek dan Hapus Duplikat
# ==========================================================

print("\n")
print("="*60)
print("CEK DUPLIKAT")
print("="*60)

jumlah_duplikat = df.duplicated().sum()
print(f"\nJumlah data duplikat: {jumlah_duplikat}")

if jumlah_duplikat > 0:
    df = df.drop_duplicates()
    print(f"Data duplikat dihapus. Sisa data: {len(df)}")
else:
    print("Tidak ada data duplikat.")

# ==========================================================
# Mengambil Kolom yang Digunakan
# ==========================================================

documents = df["Ticket Description"]

# ==========================================================
# Inisialisasi Lemmatizer & Stopwords
# ==========================================================

lemmatizer = WordNetLemmatizer()

stop_words = set(stopwords.words("english"))

# Tambahan custom stopwords untuk domain customer support
custom_stopwords = {
    "please", "assist", "issue", "product", "purchased",
    "would", "could", "also", "im", "ive", "get", "got",
    "one", "use", "using", "used", "need", "try", "tried",
    "product_purchased", "help", "want", "like", "know"
}

stop_words.update(custom_stopwords)

# ==========================================================
# Fungsi Preprocessing
# ==========================================================

def preprocess(text):
    """
    Melakukan preprocessing teks:
    1. Case Folding (lowercase)
    2. Menghapus placeholder {product_purchased}
    3. Menghapus URL
    4. Menghapus angka
    5. Menghapus tanda baca
    6. Menghapus spasi berlebih
    7. Tokenizing
    8. Stopword Removal
    9. Lemmatization
    """

    # Jika kosong / NaN
    if pd.isna(text):
        return ""

    # Ubah ke string
    text = str(text)

    # Case Folding: ubah ke lowercase
    text = text.lower()

    # Hapus placeholder {product_purchased}
    text = re.sub(r'\{product_purchased\}', '', text)

    # Hapus URL
    text = re.sub(r'http\S+', '', text)

    # Hapus angka
    text = re.sub(r'\d+', '', text)

    # Hapus tanda baca dan karakter spesial
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenizing
    tokens = word_tokenize(text)

    # Stopword Removal
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Filter kata yang terlalu pendek (< 3 karakter)
    tokens = [word for word in tokens if len(word) >= 3]

    # Gabungkan kembali
    text = " ".join(tokens)

    return text

# ==========================================================
# Menjalankan Preprocessing
# ==========================================================

print("\n")
print("="*60)
print("MENJALANKAN PREPROCESSING")
print("="*60)

print("\nMelakukan preprocessing...")

df["processed_text"] = documents.apply(preprocess)

print("Preprocessing selesai!")

# ==========================================================
# Menampilkan Contoh Hasil
# ==========================================================

print("\n")
print("="*60)
print("CONTOH HASIL PREPROCESSING (5 DATA PERTAMA)")
print("="*60)

for i in range(5):
    print(f"\n--- Data ke-{i+1} ---")
    print(f"\nSebelum :")
    print(df["Ticket Description"].iloc[i])
    print(f"\nSesudah :")
    print(df["processed_text"].iloc[i])

# ==========================================================
# Menghapus Data Kosong Setelah Preprocessing
# ==========================================================

print("\n")
print("="*60)
print("MEMBERSIHKAN DATA KOSONG")
print("="*60)

jumlah_sebelum = len(df)
df = df[df["processed_text"].str.strip() != ""]
jumlah_sesudah = len(df)

print(f"\nJumlah data sebelum cleaning : {jumlah_sebelum}")
print(f"Jumlah data setelah cleaning  : {jumlah_sesudah}")
print(f"Data yang dihapus             : {jumlah_sebelum - jumlah_sesudah}")

# ==========================================================
# Statistik Hasil Preprocessing
# ==========================================================

print("\n")
print("="*60)
print("STATISTIK HASIL PREPROCESSING")
print("="*60)

# Rata-rata jumlah kata per dokumen
avg_words = df["processed_text"].apply(lambda x: len(x.split())).mean()
min_words = df["processed_text"].apply(lambda x: len(x.split())).min()
max_words = df["processed_text"].apply(lambda x: len(x.split())).max()

print(f"\nRata-rata kata per dokumen : {avg_words:.1f}")
print(f"Minimum kata per dokumen  : {min_words}")
print(f"Maksimum kata per dokumen : {max_words}")

# ==========================================================
# Menyimpan Dataset Bersih
# ==========================================================

output_file = "dataset/clean_dataset.csv"
df.to_csv(output_file, index=False)

print(f"\nDataset berhasil disimpan ke: {output_file}")
print(f"Jumlah data final: {len(df)}")

print("\n" + "="*60)
print("PREPROCESSING SELESAI")
print("="*60)