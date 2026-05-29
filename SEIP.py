import streamlit as st
st.title("Amazon Watches Reviews Dashboard")
from google.colab import auth, drive

auth.authenticate_user()
drive.mount('/content/drive')

import gzip
import simplejson

def parse(filename):
    with gzip.open(filename, 'rt', encoding='latin-1') as f:
        entry = {}
        for line in f:
            line = line.strip()
            colon = line.find(':')
            if colon == -1:
                if entry:
                    yield entry
                entry = {}
                continue
            entry[line[:colon]] = line[colon+2:]
        if entry:
            yield entry

for e in parse('/content/drive/MyDrive/Watches.txt.gz'):
    print(simplejson.dumps(e))

import pandas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

for i, e in enumerate(parse('/content/drive/MyDrive/Watches.txt.gz')):
    print(simplejson.dumps(e, indent=2))
    print("---")
    if i >= 4:  # show first 5
        break

import pandas as pd

data = list(parse('/content/drive/MyDrive/Watches.txt.gz'))
df = pd.DataFrame(data)
df.head(10)

import numpy as np

# Replace unknowns
df.replace('unknown', np.nan, inplace=True)

# Drop duplicates
df.drop_duplicates(inplace=True)

# Fix types
df['review/score'] = pd.to_numeric(df['review/score'], errors='coerce')
df['review/time'] = pd.to_datetime(df['review/time'], unit='s')
df[['helpful_votes', 'total_votes']] = df['review/helpfulness'].str.split('/', expand=True).astype(float)

# Clean text
df['review/text'] = df['review/text'].str.strip()
df.dropna(subset=['review/text'], inplace=True)

# Rename
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

print(df.shape)
df.head()

import matplotlib.pyplot as plt

# Group reviews by year
df['year'] = df['date'].dt.year
reviews_per_year = df['year'].value_counts().sort_index()

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(reviews_per_year.index, reviews_per_year.values, color='steelblue', edgecolor='white')

# Labels & formatting
ax.set_title('Number of Watch Reviews Over Time', fontsize=16, fontweight='bold', pad=15)
ax.set_xlabel('Year', fontsize=12)
ax.set_ylabel('Number of Reviews', fontsize=12)
ax.set_xticks(reviews_per_year.index)
ax.set_xticklabels(reviews_per_year.index, rotation=45)

# Add value labels on top of each bar
for bar in ax.patches:
    ax.annotate(f'{int(bar.get_height()):,}',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.show()

# Add a column for review length (word count and character count)
df['review_word_count'] = df['review'].str.split().str.len()
df['review_char_count'] = df['review'].str.len()

# Quick stats
print(df['review_word_count'].describe())


import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Word count histogram
axes[0].hist(df['review_word_count'].dropna(), bins=50, color='steelblue', edgecolor='white')
axes[0].set_title('Distribution of Review Word Count', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Word Count')
axes[0].set_ylabel('Number of Reviews')
axes[0].axvline(df['review_word_count'].median(), color='red', linestyle='--', label=f"Median: {df['review_word_count'].median():.0f}")
axes[0].legend()

# Character count histogram
axes[1].hist(df['review_char_count'].dropna(), bins=50, color='coral', edgecolor='white')
axes[1].set_title('Distribution of Review Character Count', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Character Count')
axes[1].set_ylabel('Number of Reviews')
axes[1].axvline(df['review_char_count'].median(), color='blue', linestyle='--', label=f"Median: {df['review_char_count'].median():.0f}")
axes[1].legend()

plt.tight_layout()
plt.show()
