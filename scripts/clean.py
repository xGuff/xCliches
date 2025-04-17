import pandas as pd
import re
import nltk
nltk.download('punkt_tab')
from nltk.corpus import stopwords
from nltk import bigrams
from collections import Counter
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer, util

print("🔄 Processing transcripts...")
# --- Setup ---
nltk.download('punkt')
nltk.download('stopwords')
print("🔄 Downloading NLTK data...")

# Load data
df = pd.read_csv("data/raw/xguff_pressers_raw.csv")

# Define cliché list
cliches = [
    "there are no easy games", "game of two halves", "it's a cliche", "take each game as it comes", "twelfth man",
    "form goes out the window", "every game is a cup final",
    "important thing is we got the three points", "goal worthy of winning any tie", "what dreams are made of", "over the moon", "got that hunger", "tough place to go", "park the bus", "makes himself big",
    "one game at a time", "almost hit it too well", "played a blinder", "good squad on paper",
    "that bit of quality", "kitchen sink", "dangerous score line", "in and around the football club", "can be as good as your"
]

# Load model for semantic matching
model = SentenceTransformer('all-MiniLM-L6-v2')
cliche_embeddings = model.encode(cliches)

# Preprocessing
stop_words = set(stopwords.words("english"))

def clean_and_tokenize(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return tokens

# Word count
df["word_count"] = df["transcript_text"].apply(lambda t: len(nltk.word_tokenize(t)))

# Exact match
def count_exact_cliches(text):
    text = text.lower()
    return {phrase: text.count(phrase) for phrase in cliches}

# Fuzzy match
def count_fuzzy_cliches(text, threshold=85):
    text = text.lower()
    tokens = nltk.word_tokenize(text)
    phrases = [' '.join(bg) for bg in bigrams(tokens)] + [' '.join(tokens[i:i+3]) for i in range(len(tokens)-2)]
    scores = {phrase: 0 for phrase in cliches}
    for p in phrases:
        for c in cliches:
            score = fuzz.ratio(p, c)
            if score >= threshold:
                scores[c] += 1
    return scores

# Semantic match
def count_semantic_cliches(text, threshold=0.7):
    sentences = nltk.sent_tokenize(text.lower())
    sentence_embeddings = model.encode(sentences)
    scores = {phrase: 0 for phrase in cliches}
    for i, emb in enumerate(sentence_embeddings):
        for j, c_emb in enumerate(cliche_embeddings):
            sim = util.cos_sim(emb, c_emb).item()
            if sim > threshold:
                scores[cliches[j]] += 1
    return scores

# Apply processing
df["exact_counts"] = df["transcript_text"].apply(count_exact_cliches)
df["fuzzy_counts"] = df["transcript_text"].apply(count_fuzzy_cliches)
df["semantic_counts"] = df["transcript_text"].apply(count_semantic_cliches)

# Combine scores
def combine_scores(row):
    total = {}
    for phrase in cliches:
        total[phrase] = (
            row["exact_counts"][phrase]
            + row["fuzzy_counts"][phrase]
            + row["semantic_counts"][phrase]
        )
    return total

df["total_cliche_counts"] = df.apply(combine_scores, axis=1)

# Normalize by club
summary_df = pd.json_normalize(df["total_cliche_counts"]).join(df[["club", "word_count"]])
club_grouped = summary_df.groupby("club")
club_cliche_totals = club_grouped[cliches].sum()
club_word_totals = club_grouped["word_count"].sum()

# Add total cliche score and normalize
club_cliche_totals["total_cliche_score"] = club_cliche_totals.sum(axis=1)
club_cliche_totals["total_words"] = club_word_totals
club_cliche_totals["cliches_per_1000_words"] = club_cliche_totals["total_cliche_score"] / club_cliche_totals["total_words"] * 1000

# Save results
output_path = "data/processed/club_cliche_scores_normalized.csv"
club_cliche_totals.to_csv(output_path)

print(f"✅ Done! Normalized cliché scores saved to: {output_path}")
