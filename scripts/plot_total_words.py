import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import requests
from io import BytesIO
import nltk

# Matplotlib settings
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[cm]{sfmath}\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'cm',
    'font.size': 11,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})
plt.style.use('tableau-colorblind10')

# Download tokenizer if not already available
nltk.download('punkt')

# Paths
transcripts_path = "data/raw/transcripts.csv"
badge_path = "data/raw/club_badges.csv"
output_path = "data/outputs/total_words_by_club.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load data
df = pd.read_csv(transcripts_path)
badge_df = pd.read_csv(badge_path)

# Tokenize and count words
df["word_count"] = df["transcript_text"].apply(lambda t: len(nltk.word_tokenize(str(t))))

# Aggregate word counts by club
word_totals = df.groupby("club")["word_count"].sum().reset_index()
word_totals = word_totals.sort_values("word_count", ascending=False).reset_index(drop=True)
word_totals["rank"] = word_totals.index + 1

# Calculate total word count across all clubs
total_word_count = word_totals["word_count"].sum()
print(f"Total number of words in the entire dataset: {total_word_count}")

# Setup plot
fig, ax = plt.subplots(figsize=(8, 8))
bar_width = 0.6
cmap = plt.get_cmap("plasma")
colors = [cmap(i / len(word_totals)) for i in range(len(word_totals))]

# Plot horizontal bars
bars = ax.barh(word_totals["rank"], word_totals["word_count"], height=bar_width, color=colors)

# Add badge at end of each bar
for club, rank, word_count in zip(word_totals["club"], word_totals["rank"], word_totals["word_count"]):
    badge_url = badge_df.loc[badge_df["club"] == club, "badge_url"].values
    if badge_url.size > 0:
        try:
            response = requests.get(badge_url[0])
            img = mpimg.imread(BytesIO(response.content), format='png')
            imagebox = OffsetImage(img, zoom=0.15)
            ab = AnnotationBbox(imagebox, (word_count, rank), frameon=False, box_alignment=(0, 0.5))
            ax.add_artist(ab)
        except Exception as e:
            print(f"⚠️ Failed to load badge for {club}: {e}")

# Style
ax.set_xlabel("Total Word Count")
ax.set_yticks(word_totals["rank"])
ax.set_yticklabels(word_totals["rank"])  # Optional: replace with club names if you prefer
ax.set_ylabel("Club Ranking by Word Count")
ax.invert_yaxis()
ax.tick_params(axis='y', length=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(axis="x", linestyle="--", alpha=0.6)

# Save
plt.tight_layout()
plt.savefig(output_path, dpi=700)
plt.close()
print(f"✅ Total word count plot saved to {output_path}")
