import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs("data/outputs", exist_ok=True)

# Load normalized club scores (includes word counts and per-1000 normalization)
df = pd.read_csv("data/processed/club_cliche_scores_normalized.csv")

# --- Plot 1: Clichés per 1000 words by club ---
df_sorted = df.sort_values("cliches_per_1000_words", ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x="cliches_per_1000_words", y="club", data=df_sorted, palette="rocket")

plt.title("xGuff – Clichés Per 1000 Words by Club", fontsize=16)
plt.xlabel("Clichés per 1000 Words", fontsize=12)
plt.ylabel("Club", fontsize=12)
plt.tight_layout()
plt.savefig("data/outputs/cliches_per_1000_words_by_club.png")
plt.show()

# --- Plot 2: Top 10 clichés across all clubs (raw counts) ---
# Reload from non-normalized source if needed
df_all = pd.read_csv("data/processed/club_cliche_scores_normalized.csv")

# Get top cliché phrases by total mentions
exclude_cols = ["club", "total_cliche_score", "total_words", "cliches_per_1000_words"]
cliche_cols = [col for col in df_all.columns if col not in exclude_cols]
cliche_totals = df_all[cliche_cols].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=cliche_totals.values, y=cliche_totals.index, palette="magma")

plt.title("Top 10 Most Used Clichés in Press Conferences", fontsize=16)
plt.xlabel("Mentions (Exact + Fuzzy + Semantic)", fontsize=12)
plt.ylabel("Cliché Phrase", fontsize=12)
plt.tight_layout()
plt.savefig("data/outputs/top_cliches_overall.png")
plt.show()

