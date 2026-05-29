import streamlit as st
import gzip
import urllib.request
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Amazon Watches Reviews Dashboard")

# ── Load & parse data ──────────────────────────────────────────
@st.cache_data
def load_data():
    url = "https://snap.stanford.edu/data/amazon/Watches.txt.gz"
    response = urllib.request.urlopen(url)
    f = gzip.open(io.BytesIO(response.read()), 'rt', encoding='latin-1')
    entries = []
    entry = {}
    for line in f:
        line = line.strip()
        colon = line.find(':')
        if colon == -1:
            if entry:
                entries.append(entry)
            entry = {}
            continue
        entry[line[:colon]] = line[colon+2:]
    if entry:
        entries.append(entry)
    return pd.DataFrame(entries)

with st.spinner("Loading data..."):
    df = load_data()

# ── Clean data ─────────────────────────────────────────────────
df.replace('unknown', np.nan, inplace=True)
df.drop_duplicates(inplace=True)
df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')
df['review/time'] = pd.to_datetime(pd.to_numeric(df['review/time'], errors='coerce'), unit='s')
df[['helpful_votes', 'total_votes']] = df['review/helpfulness'].str.split('/', expand=True).astype(float)
df['review/text'] = df['review/text'].str.strip()
df.dropna(subset=['review/text'], inplace=True)
df.rename(columns={
    'product/productId': 'product_id',
    'product/title':     'title',
    'product/price':     'price',
    'review/userId':     'user_id',
    'review/profileName':'profile_name',
    'review/helpfulness':'helpfulness',
    'review/score':      'score',
    'review/time':       'date',
    'review/summary':    'summary',
    'review/text':       'review'
}, inplace=True)

# ── Preview ────────────────────────────────────────────────────
st.subheader("Data Preview")
st.write(f"Shape: {df.shape}")
st.dataframe(df.head(10))

# ── Chart 1: Reviews over time ─────────────────────────────────
st.subheader("Number of Reviews Over Time")
df['year'] = df['date'].dt.year
reviews_per_year = df['year'].value_counts().sort_index()

fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.bar(reviews_per_year.index, reviews_per_year.values, color='steelblue', edgecolor='white')
ax1.set_title('Number of Watch Reviews Over Time', fontsize=16, fontweight='bold', pad=15)
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Number of Reviews', fontsize=12)
ax1.set_xticks(reviews_per_year.index)
ax1.set_xticklabels(reviews_per_year.index, rotation=45)
for bar in ax1.patches:
    ax1.annotate(f'{int(bar.get_height()):,}',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='bottom', fontsize=9)
plt.tight_layout()
st.pyplot(fig1)

# ── Chart 2: Review length analysis ───────────────────────────
st.subheader("Review Length Analysis")
df['review_word_count'] = df['review'].str.split().str.len()
df['review_char_count'] = df['review'].str.len()
st.write(df['review_word_count'].describe())

fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df['review_word_count'].dropna(), bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Review Word Count', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Number of Reviews')
axes[0].axvline(df['review_word_count'].median(), color='red', linestyle='--',
                label=f"Median: {df['review_word_count'].median():.0f}")
axes[0].legend()

axes[1].hist(df['review_char_count'].dropna(), bins=50, color='coral', edgecolor='white')
axes[1].set_title('Distribution of Review Character Count', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Number of Reviews')
axes[1].axvline(df['review_char_count'].median(), color='blue', linestyle='--',
                label=f"Median: {df['review_char_count'].median():.0f}")
axes[1].legend()
plt.tight_layout()
st.pyplot(fig2)

from wordcloud import WordCloud, STOPWORDS

st.subheader("Word Cloud of Watch Reviews")

# Combine all reviews into one text
text = ' '.join(df['review'].dropna())

# Generate word cloud
wordcloud = WordCloud(
    width=800, 
    height=400,
    background_color='white',
    colormap='Blues',
    stopwords=STOPWORDS,
    max_words=200,
    collocations=False
).generate(text)

# Display
fig_wc, ax_wc = plt.subplots(figsize=(12, 6))
ax_wc.imshow(wordcloud, interpolation='bilinear')
ax_wc.axis('off')
ax_wc.set_title('Most Common Words in Watch Reviews', fontsize=16, fontweight='bold', pad=15)
st.pyplot(fig_wc)
