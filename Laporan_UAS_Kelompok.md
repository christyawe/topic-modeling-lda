# LAPORAN AKHIR PROYEK DATA MINING / NLP
**Judul:** Pemodelan Topik Keluhan Pelanggan E-Commerce Menggunakan Algoritma Latent Dirichlet Allocation (LDA)

---

## BAB I: PENDAHULUAN
Keluhan pelanggan (*customer support tickets*) merupakan salah satu sumber data tekstual terbesar dalam bisnis e-commerce. Untuk membantu perusahaan memahami masalah utama yang sedang dihadapi oleh pelanggan, diperlukan sebuah sistem yang dapat mengelompokkan ribuan keluhan tersebut secara otomatis. Proyek ini bertujuan untuk membangun model *Machine Learning* menggunakan metode *Topic Modeling* (Pemodelan Topik) untuk mengklasifikasikan keluhan pelanggan ke dalam beberapa topik utama secara otomatis.

---

## BAB II: LANDASAN TEORI *(Oleh: Jordan)*

### 2.1 Natural Language Processing (NLP)
*Natural Language Processing* (NLP) adalah cabang dari kecerdasan buatan (AI) yang berfokus pada interaksi antara komputer dan bahasa manusia. NLP memungkinkan komputer untuk membaca, memahami, dan menganalisis data tekstual (seperti keluhan pelanggan) untuk diekstrak maknanya.

### 2.2 Topic Modeling
*Topic Modeling* adalah teknik pembelajaran mesin tanpa pengawasan (*unsupervised learning*) yang digunakan untuk menemukan struktur topik tersembunyi (*latent*) di dalam sekumpulan dokumen besar. Teknik ini tidak memerlukan data teks yang sudah diberi label sebelumnya.

### 2.3 Latent Dirichlet Allocation (LDA)
LDA adalah algoritma *Topic Modeling* yang paling populer. Prinsip kerja LDA didasarkan pada asumsi bahwa:
1. Setiap dokumen (keluhan pelanggan) terdiri dari campuran beberapa topik.
2. Setiap topik terdiri dari kumpulan kata-kata yang memiliki probabilitas kemunculan tertentu.
LDA bekerja secara iteratif untuk mengoptimalkan distribusi kata dalam topik dan distribusi topik dalam dokumen.

---

## BAB III: METODOLOGI & PREPROCESSING *(Oleh: Erfa)*

### 3.1 Deskripsi Dataset
Dataset yang digunakan dalam penelitian ini diambil dari platform Kaggle dengan nama *Customer Support Tickets*. Dataset mentah berisi data operasional layanan pelanggan. Atribut utama yang digunakan untuk pemodelan NLP adalah kolom **`Ticket Description`**, yang berisi teks penjelasan lengkap dari pelanggan mengenai kendala yang mereka alami (berbahasa Inggris).

### 3.2 Tahapan Preprocessing
Agar teks mentah dapat diproses oleh algoritma LDA, dilakukan tahapan pembersihan data (*preprocessing*) menggunakan *library* NLTK dan Pandas:
1. **Pembersihan Missing Value & Duplikat:** Membuang baris data yang kosong atau memiliki nilai ganda.
2. **Case Folding:** Mengubah seluruh teks keluhan menjadi huruf kecil (*lowercase*).
3. **Data Cleansing:** Menghapus URL/Link, angka, tanda baca, simbol khusus, dan spasi ganda menggunakan *Regular Expression* (Regex).
4. **Tokenizing:** Memecah kalimat keluhan utuh menjadi potongan kata-kata tunggal (*tokens*).
5. **Stopword Removal:** Membuang kata-kata hubung bahasa Inggris (seperti *and, the, is, of*) serta kata khusus domain (seperti *please, assist, issue*) yang tidak memiliki makna topik.
6. **Lemmatization:** Mengubah kata-kata yang memiliki imbuhan kembali ke bentuk dasarnya (contoh: *running* menjadi *run*) menggunakan *WordNetLemmatizer*.

Output dari tahap ini adalah dataset bersih (`clean_dataset.csv`) yang siap diproses oleh model.

---

## BAB IV: MODELING DAN IMPLEMENTASI SISTEM

### 4.1 Pemodelan Topik LDA *(Oleh: Jordan)*
Model dibangun menggunakan *library* Gensim. Dokumen yang telah bersih diubah menjadi *Bag-of-Words* (Corpus). Parameter model diatur untuk menghasilkan **5 Topik Utama**.

*(Sisipkan Gambar Screenshot Hasil 10 Kata Kunci / Kotak Biru dari Web di sini)*

**Hasil Analisis Topik (Interpretasi):**
Berdasarkan *10 kata kunci* teratas yang dihasilkan oleh model LDA, berikut adalah interpretasi dari kelima topik tersebut:
*   **Topik 1 (Isu Jaringan & Koneksi):** Didominasi kata *data, step, problem, resolve, network, unable*. Pelanggan mengeluhkan masalah jaringan dan kesulitan menemukan langkah penyelesaian.
*   **Topik 2 (Kendala Update Software):** Didominasi kata *update, software, problem, device, persists*. Berisi laporan error atau kendala yang tak kunjung hilang pasca pembaruan sistem/perangkat.
*   **Topik 3 (Masalah Akun & Firmware):** Didominasi kata *account, update, working, firmware*. Berkaitan dengan masalah login akun atau pembaruan firmware keras keras.
*   **Topik 4 (Error Pesan Layar):** Didominasi kata *screen, message, possible, error, assistance*. Berisi keluhan tentang munculnya pesan error di layar pengguna yang membutuhkan bantuan teknis.
*   **Topik 5 (Kerusakan Intermiten):** Didominasi kata *noticed, specific, facing, sometimes, intermittent, unexpectedly*. Berisi keluhan masalah kinerja yang muncul secara tiba-tiba atau kadang-kadang rusak.

Model ini juga dievaluasi menggunakan **Coherence Score (u_mass)** untuk memastikan keakuratan pengelompokan kata, di mana hasil pengujian menunjukkan skor yang cukup stabil dan logis secara kualitatif.

### 4.2 Implementasi Sistem *(Oleh: Farid)*
Untuk memudahkan pengguna akhir (seperti manajer layanan pelanggan) dalam menggunakan model ini, dibangun sebuah aplikasi berbasis *Web* menggunakan *framework* **Streamlit**. Fitur utama website meliputi:
*   **Upload CSV:** Pengguna dapat mengunggah data keluhan baru.
*   **Real-time Preprocessing:** Website langsung membersihkan data teks baru yang diunggah.
*   **Integrasi Model LDA:** Memuat (*load*) *pre-trained model* buatan Jordan untuk langsung mengklasifikasikan topik tanpa perlu melatih ulang (*training*) dari nol, sehingga web sangat ringan dan cepat.

### 4.3 Pengujian & Visualisasi Sistem *(Oleh: Farid)*
Sistem berhasil diuji dan divisualisasikan dengan dua metode:
1. **Bar Chart Distribusi Topik:** Menampilkan grafik batang jumlah keluhan yang masuk ke masing-masing kategori topik, memudahkan perusahaan mengetahui masalah mana yang paling banyak dialami pengguna (menggunakan *Matplotlib*).
   *(Sisipkan Gambar Screenshot Bar Chart Distribusi Topik di sini)*
2. **Word Cloud:** Memvisualisasikan seberapa sering atau penting suatu kata muncul di dalam satu topik tertentu. Kata yang semakin besar ukurannya menandakan bobot yang semakin dominan di topik tersebut.
   *(Sisipkan Gambar Screenshot Word Cloud di sini)*

---

## BAB V: KESIMPULAN *(Oleh: Farid)*
Proyek ini berhasil membuktikan bahwa algoritma **Latent Dirichlet Allocation (LDA)** dapat diimplementasikan untuk menganalisis dan mengelompokkan keluhan pelanggan *e-commerce* secara otomatis tanpa pelabelan manual. 

Dari pengujian sistem terintegrasi berbasis Streamlit, pipeline *Natural Language Processing* (mulai dari pembersihan data hingga visualisasi topik) mampu berjalan dengan efisien dan interaktif. Hasil ekstraksi 5 topik utama terbukti logis dan relevan, yang dapat dimanfaatkan oleh divisi *Customer Support* untuk mengambil keputusan penanganan keluhan secara lebih cepat dan akurat.
