import matplotlib.pyplot as plt
from wordcloud import WordCloud
import streamlit as st

def plot_wordcloud(lda_model, topic_id):
    """
    Menghasilkan WordCloud untuk topik tertentu.
    """
    # Mengambil kata dan probabilitasnya dari model LDA
    word_freq = dict(lda_model.show_topic(topic_id, topn=30))
    
    # Generate WordCloud
    wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis')
    wc.generate_from_frequencies(word_freq)
    
    # Plot dengan matplotlib
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(f'Word Cloud - Topik {topic_id + 1}', fontsize=16)
    
    return fig

def plot_topic_distribution(corpus, lda_model):
    """
    Menghasilkan Bar Chart distribusi topik pada seluruh dokumen.
    """
    topic_counts = {}
    
    # Mencari topik paling dominan (probabilitas tertinggi) untuk tiap keluhan
    for doc in corpus:
        topics = lda_model.get_document_topics(doc)
        if topics:
            dominant_topic = max(topics, key=lambda x: x[1])[0]
            topic_counts[dominant_topic] = topic_counts.get(dominant_topic, 0) + 1
            
    # Sort berdasarkan ID topik agar berurutan
    topics_list = []
    counts_list = []
    for i in range(lda_model.num_topics):
        topics_list.append(f"Topik {i+1}")
        counts_list.append(topic_counts.get(i, 0))
    
    # Plotting Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(topics_list, counts_list, color='#4CAF50', edgecolor='black')
    
    ax.set_title('Distribusi Topik pada Keluhan Pelanggan', fontsize=16, fontweight='bold')
    ax.set_ylabel('Jumlah Keluhan', fontsize=12)
    ax.set_xlabel('Klaster Topik', fontsize=12)
    
    # Menambahkan label angka di atas setiap bar
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(counts_list)*0.01), 
                int(yval), ha='center', va='bottom', fontweight='bold')
        
    # Rapikan layout
    plt.tight_layout()
    return fig
