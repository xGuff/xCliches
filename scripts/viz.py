import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Ensure output directory exists
os.makedirs("data/outputs", exist_ok=True)

# Load normalized club scores (includes word counts and per-1000 normalization)
df = pd.read_csv("data/processed/cliches_by_club.csv")



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

# Get total mentions for all cliché phrases
exclude_cols = ["club", "total_cliche_score", "total_words", "cliches_per_1000_words"]
cliche_cols = [col for col in df_all.columns if col not in exclude_cols]
cliche_totals = df_all[cliche_cols].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=cliche_totals.values, y=cliche_totals.index, palette="viridis")

plt.title("Total Mentions of All Clichés in Press Conferences", fontsize=16)
plt.xlabel("Mentions", fontsize=12)
plt.ylabel("Cliché Phrase", fontsize=12)
plt.tight_layout()
plt.savefig("data/outputs/all_cliches_mentions.png")
plt.show()

# --- Plot 3: Each Club’s Most Used Cliché ---

# Reload normalized cliche breakdown
df = pd.read_csv("data/processed/club_cliche_scores_normalized.csv")

# Columns containing clichés only
exclude_cols = ["club", "total_cliche_score", "total_words", "cliches_per_1000_words"]
cliche_cols = [col for col in df.columns if col not in exclude_cols]

# Get each club's most used cliché
club_top_cliches = []
for _, row in df.iterrows():
    club = row["club"]
    top_cliche = row[cliche_cols].idxmax()
    count = row[top_cliche]
    club_top_cliches.append({
        "club": club,
        "top_cliche": top_cliche,
        "count": count
    })

df_top = pd.DataFrame(club_top_cliches)
df_top = df_top.sort_values("count", ascending=False)

# Plot
plt.figure(figsize=(12, 10))
sns.barplot(x="count", y="club", hue="top_cliche", data=df_top, dodge=False, palette="Set2")

plt.title("xGuff – Most Used Cliché by Club", fontsize=16)
plt.xlabel("Mentions", fontsize=12)
plt.ylabel("Club", fontsize=12)
plt.legend(title="Cliché", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig("data/outputs/most_used_cliche_by_club.png")
plt.show()

from wordcloud import WordCloud
import matplotlib.pyplot as plt

cliche_freq = df[cliche_cols].sum().to_dict()
wc = WordCloud(background_color="white", width=1920, height=1080).generate_from_frequencies(cliche_freq)

plt.figure(figsize=(16, 9))
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()
