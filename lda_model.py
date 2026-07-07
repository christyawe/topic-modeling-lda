import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

# ==========================================================
# Membaca Dataset Hasil Preprocessing
# ==========================================================

print("="*60)
print("MEMUAT DATASET HASIL PREPROCESSING")
print("="*60)

df = pd.read_csv("dataset/clean_dataset.csv")

print(f"\nJumlah data: {len(df)}")

# Pastikan kolom processed_text ada dan tidak kosong
df = df[df["processed_text"].notna()]
df = df[df["processed_text"].str.strip() != ""]

print(f"Jumlah data valid: {len(df)}")

# ==========================================================
# Membuat Tokenized Documents
# ==========================================================

print("\n" + "="*60)
print("MENYIAPKAN DATA UNTUK LDA")
print("="*60)

# Tokenisasi: pecah teks menjadi list kata
texts = df["processed_text"].apply(lambda x: x.split()).tolist()

print(f"\nJumlah dokumen: {len(texts)}")
print(f"Contoh dokumen pertama (tokens): {texts[0][:10]}...")

# ==========================================================
# Membuat Dictionary dan Corpus (Bag of Words)
# ==========================================================

print("\nMembuat Dictionary...")

# Membuat dictionary dari seluruh kata
dictionary = corpora.Dictionary(texts)

print(f"Jumlah kata unik (sebelum filter): {len(dictionary)}")

# Filter extremes:
# - Hapus kata yang muncul di kurang dari 15 dokumen
# - Hapus kata yang muncul di lebih dari 50% dokumen
dictionary.filter_extremes(no_below=15, no_above=0.5)

print(f"Jumlah kata unik (setelah filter): {len(dictionary)}")

# Membuat corpus (Bag of Words)
corpus = [dictionary.doc2bow(text) for text in texts]

print(f"Jumlah dokumen dalam corpus: {len(corpus)}")

# ==========================================================
# Membangun Model LDA
# ==========================================================

print("\n" + "="*60)
print("MEMBANGUN MODEL LDA")
print("="*60)

NUM_TOPICS = 5  # Jumlah topik

print(f"\nJumlah topik: {NUM_TOPICS}")
print("Melatih model LDA... (mohon tunggu)")

lda_model = LdaModel(
    corpus=corpus,
    id2word=dictionary,
    num_topics=NUM_TOPICS,
    random_state=42,
    update_every=1,
    chunksize=100,
    passes=10,
    alpha='auto',
    per_word_topics=True
)

print("Model LDA berhasil dibuat!")

# ==========================================================
# Menampilkan Hasil Topik (10 Kata Kunci per Topik)
# ==========================================================

print("\n" + "="*60)
print(f"HASIL TOPIC MODELING - {NUM_TOPICS} TOPIK")
print("="*60)

topics = lda_model.print_topics(num_words=10)

for idx, topic in topics:
    print(f"\n{'-'*50}")
    print(f"  TOPIK {idx + 1}")
    print(f"{'-'*50}")

    # Parse kata dan bobotnya
    words = topic.split(" + ")
    for word_weight in words:
        weight, word = word_weight.split("*")
        weight = float(weight)
        word = word.strip().strip('"')
        bar = "#" * int(weight * 100)
        print(f"  {word:<20} {weight:.4f}  {bar}")

# ==========================================================
# Interpretasi Topik
# ==========================================================

print("\n" + "="*60)
print("INTERPRETASI TOPIK")
print("="*60)

print("""
Berikut adalah interpretasi dari setiap topik berdasarkan kata kunci:

Catatan: Interpretasi di bawah ini bersifat umum dan dapat disesuaikan
setelah melihat kata kunci aktual yang dihasilkan oleh model.
Kata kunci di setiap topik menggambarkan tema utama dari keluhan
atau pertanyaan pelanggan dalam tiket customer support.
""")

for idx, topic in topics:
    words = topic.split(" + ")
    top_words = []
    for word_weight in words[:5]:
        weight, word = word_weight.split("*")
        word = word.strip().strip('"')
        top_words.append(word)
    print(f"  Topik {idx + 1}: {', '.join(top_words)}")

# ==========================================================
# Evaluasi Model - Coherence Score
# ==========================================================

print("\n" + "="*60)
print("EVALUASI MODEL")
print("="*60)

# Coherence Score menggunakan u_mass (lebih cepat, berbasis corpus)
coherence_model = CoherenceModel(
    model=lda_model,
    corpus=corpus,
    dictionary=dictionary,
    coherence='u_mass'
)

coherence_score = coherence_model.get_coherence()
print(f"\nCoherence Score (u_mass): {coherence_score:.4f}")

# u_mass menghasilkan nilai negatif, semakin mendekati 0 semakin baik
if coherence_score >= -1.0:
    print(">> Coherence Score BAIK (>= -1.0)")
elif coherence_score >= -2.0:
    print(">> Coherence Score CUKUP (>= -2.0)")
else:
    print(">> Coherence Score RENDAH (< -2.0)")

print("\nCatatan: u_mass menghasilkan nilai negatif.")
print("Semakin mendekati 0, semakin baik kualitas topik.")

# ==========================================================
# Assign Topik Dominan ke Setiap Dokumen
# ==========================================================

print("\n" + "="*60)
print("ASSIGN TOPIK KE DOKUMEN")
print("="*60)

def get_dominant_topic(lda_model, corpus, texts_df):
    """
    Mendapatkan topik dominan untuk setiap dokumen.
    """
    dominant_topics = []
    topic_percentages = []

    for i, row in enumerate(lda_model[corpus]):
        row_topics = row[0]  # (topic_id, probability)
        row_topics = sorted(row_topics, key=lambda x: x[1], reverse=True)

        dominant_topic_id = row_topics[0][0] + 1  # +1 supaya mulai dari 1
        dominant_topic_pct = round(row_topics[0][1], 4)

        dominant_topics.append(dominant_topic_id)
        topic_percentages.append(dominant_topic_pct)

    return dominant_topics, topic_percentages

dominant_topics, topic_percentages = get_dominant_topic(lda_model, corpus, df)

df["dominant_topic"] = dominant_topics
df["topic_probability"] = topic_percentages

print("\nDistribusi dokumen per topik:")
topic_dist = df["dominant_topic"].value_counts().sort_index()
for topic_id, count in topic_dist.items():
    pct = count / len(df) * 100
    bar = "#" * int(pct)
    print(f"  Topik {topic_id}: {count:>5} dokumen ({pct:.1f}%) {bar}")

# ==========================================================
# Contoh Dokumen per Topik
# ==========================================================

print("\n" + "="*60)
print("CONTOH DOKUMEN PER TOPIK")
print("="*60)

for topic_id in range(1, NUM_TOPICS + 1):
    print(f"\n{'-'*50}")
    print(f"  TOPIK {topic_id} - Contoh Dokumen:")
    print(f"{'-'*50}")

    topic_docs = df[df["dominant_topic"] == topic_id]
    if len(topic_docs) > 0:
        sample = topic_docs.head(2)
        for _, row in sample.iterrows():
            text = row["processed_text"]
            if len(text) > 150:
                text = text[:150] + "..."
            print(f"  >> {text}")

# ==========================================================
# Menyimpan Hasil
# ==========================================================

print("\n" + "="*60)
print("MENYIMPAN HASIL")
print("="*60)

# Simpan model LDA
lda_model.save("dataset/lda_model.model")
print("  [OK] Model LDA disimpan: dataset/lda_model.model")

# Simpan dictionary
dictionary.save("dataset/dictionary.dict")
print("  [OK] Dictionary disimpan: dataset/dictionary.dict")

# Simpan corpus
corpora.MmCorpus.serialize("dataset/corpus.mm", corpus)
print("  [OK] Corpus disimpan: dataset/corpus.mm")

# Simpan dataset dengan topik
df.to_csv("dataset/clean_dataset.csv", index=False)
print("  [OK] Dataset dengan topik disimpan: dataset/clean_dataset.csv")

print("\n" + "="*60)
print("TOPIC MODELING SELESAI")
print("="*60)
