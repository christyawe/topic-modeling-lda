import streamlit as st
import pandas as pd
from gensim.models import LdaModel
from gensim import corpora
from preprocessing import preprocess
from visualization import plot_wordcloud, plot_topic_distribution

# ==========================================
# KONFIGURASI HALAMAN STREAMLIT
# ==========================================
st.set_page_config(
    page_title="App Topic Modeling - LDA",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Aplikasi Topic Modeling Keluhan Pelanggan E-Commerce (LDA)")
st.markdown("""
**Dibuat oleh Kelompok:**
* 👨‍💻 **Erfa**: Data Preparation & Preprocessing
* 👨‍💻 **Jordan**: Topic Modeling (LDA)
* 👨‍💻 **Farid**: Website & Visualisasi
""")
st.divider()

# ==========================================
# 1. UPLOAD CSV & PREVIEW (Tugas Farid)
# ==========================================
st.header("1. Upload Dataset Keluhan")
uploaded_file = st.file_uploader("Upload file dataset keluhan (CSV format)", type=['csv'])

if uploaded_file is not None:
    # Membaca dataset
    df = pd.read_csv(uploaded_file)
    
    st.subheader("Preview Dataset")
    st.dataframe(df)
    
    # Pilih kolom yang berisi keluhan
    kolom_teks = st.selectbox("Pilih kolom yang berisi keluhan (teks) pelanggan:", df.columns)
    
    st.divider()
    
    # ==========================================
    # 2. LOAD MODEL LDA (Karya Jordan)
    # ==========================================
    @st.cache_resource
    def load_jordan_model():
        lda_model = LdaModel.load("dataset/lda_model.model")
        dictionary = corpora.Dictionary.load("dataset/dictionary.dict")
        return lda_model, dictionary
        
    try:
        lda_model, dictionary = load_jordan_model()
        num_topics = lda_model.num_topics
    except Exception as e:
        st.error(f"Gagal memuat model LDA dari Jordan! Pastikan file dataset/lda_model.model ada. Error: {e}")
        st.stop()
    
    # ==========================================
    # 3. TOMBOL PROSES (Integrasi)
    # ==========================================
    if st.button("🚀 Proses Analisis Topik", use_container_width=True):
        st.session_state.is_processed = True
        
        # --- A. PREPROCESSING (Karya Erfa) ---
        with st.spinner("⏳ Menjalankan Preprocessing (Case Folding, Tokenizing, Lemmatization)..."):
            # Batasi data jika terlalu besar untuk demo web (opsional)
            if len(df) > 1000:
                st.warning("Untuk menghemat memori website, kami hanya memproses 1000 sampel data pertama.")
                df = df.head(1000).copy()
                
            # Gunakan fungsi preprocess dari Erfa
            df['processed_text'] = df[kolom_teks].apply(preprocess)
            
            # Hapus yang kosong
            df = df[df['processed_text'].str.strip() != ""]
            
            # Ubah ke bentuk list of tokens untuk gensim
            texts = df['processed_text'].apply(lambda x: x.split()).tolist()
            
            st.success("✅ Preprocessing Selesai!")
            
        # --- B. INFERENSI LDA (Menggunakan Model Jordan) ---
        with st.spinner(f"⏳ Mengelompokkan Topik menggunakan Model LDA Jordan..."):
            # Ubah kata menjadi Bag of Words menggunakan dictionary Jordan
            corpus = [dictionary.doc2bow(text) for text in texts]
            
            # Dapatkan topik paling dominan untuk setiap dokumen
            dominant_topics = []
            for doc in corpus:
                if len(doc) == 0:
                    dominant_topics.append(-1)
                    continue
                topics = lda_model.get_document_topics(doc)
                dominant_topic = max(topics, key=lambda x: x[1])[0]
                dominant_topics.append(dominant_topic)
                
            df['Dominant_Topic'] = dominant_topics
            
            # Simpan data ke session_state agar tidak hilang
            st.session_state.corpus = corpus
            st.session_state.df = df
            
            st.success("✅ Klasifikasi Topik Selesai!")

    st.divider()

    # ==========================================
    # 4. HASIL TOPIK & VISUALISASI (Tugas Farid)
    # ==========================================
    if st.session_state.get('is_processed', False):
        st.header("2. Hasil Klasifikasi Topik")
        
        st.subheader("💡 10 Kata Kunci pada Setiap Topik")
        topics = lda_model.show_topics(num_topics=num_topics, num_words=10, formatted=False)
        
        cols = st.columns(2)
        for i, topic in enumerate(topics):
            topic_id = topic[0]
            words = [word[0] for word in topic[1]]
            with cols[i % 2]:
                st.info(f"**Topik {topic_id + 1}**: {', '.join(words)}")
        
        st.divider()
        
        st.subheader("📈 Visualisasi Distribusi & Word Cloud")
        
        tab1, tab2 = st.tabs(["📊 Bar Chart Distribusi Topik", "☁️ Word Cloud per Topik"])
        
        with tab1:
            st.write("Grafik distribusi topik dari dataset yang baru Anda upload.")
            fig_bar = plot_topic_distribution(st.session_state.corpus, lda_model)
            st.pyplot(fig_bar)
            
        with tab2:
            st.write("Word Cloud memvisualisasikan kata-kata yang paling dominan di setiap topik dari model yang dilatih Jordan.")
            selected_topic = st.selectbox("Pilih Topik untuk Word Cloud:", range(1, num_topics + 1))
            fig_wc = plot_wordcloud(lda_model, selected_topic - 1)
            st.pyplot(fig_wc)
