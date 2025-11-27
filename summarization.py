import matplotlib.pyplot as plt
from datasets import load_dataset
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Load dataset
dataset = load_dataset("cnn_dailymail", '3.0.0')
df = dataset['train'].to_pandas().head(1000)

# Text preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words and token.isalpha()]
    return " ".join(tokens)

###====== Part 2.1 =====================
df['article_len'] = df['article'].apply(len)
df['highlights_len'] = df['highlights'].apply(len)

###====== Part 2.2 =====================
def plot_histograms(df):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.hist(df['article_len'], bins=30, color='skyblue', edgecolor='black')
    plt.title('Length of Articles')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    plt.hist(df['highlights_len'], bins=30, color='salmon', edgecolor='black')
    plt.title('Length of Highlights')
    plt.xlabel('Characters')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

plot_histograms(df)

###====== Part 2.3 =====================
from collections import Counter

def ngrams(text, n):
    processed_text = preprocess_text(text)
    words = processed_text.split()
    return list(zip(*[words[i:] for i in range(n)]))

def rouge_n(reference, candidate, n):
    ref_ngrams = ngrams(reference, n)
    cand_ngrams = ngrams(candidate, n)
    if not ref_ngrams:
        return 0.0
    ref_counts = Counter(ref_ngrams)
    cand_counts = Counter(cand_ngrams)
    overlap = sum((ref_counts & cand_counts).values())
    return overlap / len(ref_ngrams)

df['rouge_1'] = df.apply(lambda row: rouge_n(row['highlights'], row['article'], 1), axis=1)
df['rouge_2'] = df.apply(lambda row: rouge_n(row['highlights'], row['article'], 2), axis=1)

plt.figure(figsize=(12, 6))
plt.hist(df['rouge_2'], bins=30, color='blue', alpha=0.7)
plt.title('Rouge-2 score distribution on ground truth')
plt.xlabel('ROUGE-2')
plt.ylabel('Frequency')
plt.show()

max_idx = df['rouge_2'].idxmax()
min_idx = df['rouge_2'].idxmin()

print("=== Highest ROUGE-2 ===")
print("Article:", df.loc[max_idx]['article'])
print("\nHighlights:", df.loc[max_idx]['highlights'])

print("\n=== Lowest ROUGE-2 ===")
print("Article:", df.loc[min_idx]['article'])
print("\nHighlights:", df.loc[min_idx]['highlights'])

###====== Part 2.4 =====================
summarizer = pipeline("summarization", model="t5-small")

def summarize_text(text):
    summary = summarizer(text, max_length=20, min_length=5, do_sample=False)
    return summary[0]['summary_text']

# Compare model-generated summaries to ground truth using ROUGE-2
for i in range(10):
    generated = summarize_text(df.loc[i, 'article'])
    reference = df.loc[i, 'highlights']
    score = rouge_n(reference, generated, 2)
    print(f"\nExample {i+1}: ROUGE-2 score = {score:.4f}")
    print("Generated:", generated)
    print("Reference:", reference)
