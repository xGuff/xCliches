import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import os

# Download tokenizer if not already available
nltk.download('punkt')

# === Load transcript data ===
df = pd.read_csv("data/raw/transcripts.csv")

# === Tokenize transcript text and count words ===
df["word_count"] = df["transcript_text"].apply(lambda t: len(nltk.word_tokenize(str(t))))

# === Aggregate by club ===
word_totals = df.groupby("club")["word_count"].sum().sort_values(ascending=False).reset_index()

# === Plot horizontal bars with 'plasma_r' colormap ===
plt.figure(figsize=(8, 8))
sns.set_style("whitegrid")
sns.barplot(data=word_totals, x="word_count", y="club", palette="plasma")

plt.xlabel("Total Word Count")
plt.ylabel("Club")
plt.tight_layout()

# === Save plot ===
output_path = "data/outputs/total_words_by_club.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path)
plt.close()

print(f"✅ Total word count plot saved to {output_path}")

# Distribution histogram remains unchanged
word_totals["word_count"].hist(bins=15)
plt.axvline(5000, color="red", linestyle="--", label="Suggested Threshold")
plt.title("Distribution of Word Counts Across Clubs")
plt.xlabel("Total Words")
plt.ylabel("Number of Clubs")
plt.legend()
plt.show()
