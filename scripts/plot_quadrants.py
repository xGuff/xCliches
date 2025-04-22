import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests
from io import BytesIO
import os

# Matplotlib settings
plt.rcParams.update({
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage[cm]{sfmath}\usepackage{amsmath}',
    'font.family': 'sans-serif',
    'font.sans-serif': 'cm',
    'font.size': 14,
    'xtick.direction': 'in',
    'ytick.direction': 'in'
})
plt.style.use('tableau-colorblind10')

# === Parameters ===
WORD_COUNT_THRESHOLD = 0

# === Paths ===
cliche_path = "data/processed/cliches_by_club.csv"
badges_path = "data/raw/club_badges.csv"
standings_path = "data/processed/points_per_game.csv"
transcripts_path = "data/raw/transcripts.csv"
output_path = "data/outputs/quadrants.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === Load data ===
cliches_df = pd.read_csv(cliche_path)
badges_df = pd.read_csv(badges_path)
standings_df = pd.read_csv(standings_path)
transcripts_df = pd.read_csv(transcripts_path)

# === Filter clubs by total word count threshold ===
transcripts_df["word_count"] = transcripts_df["transcript_text"].str.split().str.len()
word_totals = transcripts_df.groupby("club")["word_count"].sum()
valid_clubs = word_totals[word_totals >= WORD_COUNT_THRESHOLD].index.tolist()

# === Filter datasets ===
cliches_df = cliches_df[cliches_df["club"].isin(valid_clubs)]
standings_df = standings_df[standings_df["club"].isin(valid_clubs)]
badges_df = badges_df[badges_df["club"].isin(valid_clubs)]

# === Merge ===
merged = pd.merge(cliches_df, standings_df, on="club")
merged = pd.merge(merged, badges_df, on="club")

# === Plot setup ===
plt.figure(figsize=(12, 8))
ax = plt.gca()

# Helper to get badge image
def get_club_badge(url, zoom=0.2):
    try:
        response = requests.get(url)
        img = plt.imread(BytesIO(response.content), format='png')
        return OffsetImage(img, zoom=zoom)
    except Exception as e:
        print(f"⚠️ Failed to load badge: {url} — {e}")
        return None

# === Axes and quadrant setup ===
x = merged["cliches_per_10000_words"]
y = merged["points_per_game"]
mean_x = x.mean()
mean_y = y.mean()

# Draw quadrant lines
ax.axhline(mean_y, color="grey", linestyle="--", linewidth=1)
ax.axvline(mean_x, color="grey", linestyle="--", linewidth=1)

# Set axis limits to create evenly sized quadrants
range_x = max(x.max() - mean_x, mean_x - x.min())
range_y = max(y.max() - mean_y, mean_y - y.min())
ax.set_xlim(mean_x - range_x, mean_x + range_x)
ax.set_ylim(mean_y - range_y, mean_y + range_y)

# Quadrant labels
ax.text(mean_x - range_x * 0.9, mean_y + range_y * 0.9,
        "Low cliché, high performance", fontsize=12, ha='left', va='top', color='gray')
ax.text(mean_x + range_x * 0.1, mean_y + range_y * 0.9,
        "High cliché, high performance", fontsize=12, ha='left', va='top', color='gray')
ax.text(mean_x - range_x * 0.9, mean_y - range_y * 0.9,
        "Low cliché, low performance", fontsize=12, ha='left', va='top', color='gray')
ax.text(mean_x + range_x * 0.1, mean_y - range_y * 0.9,
        "High cliché, low performance", fontsize=12, ha='left', va='top', color='gray')


for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)
# Add club badges
for _, row in merged.iterrows():
    img = get_club_badge(row["badge_url"])
    if img:
        ab = AnnotationBbox(img, (row["cliches_per_10000_words"], row["points_per_game"]),
                            frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

# Final styling
ax.set_xlabel("Clichés per 10,000 Words", fontsize=13)
ax.set_ylabel("Points per Game", fontsize=13)
plt.savefig(output_path, dpi=700)
plt.close()
print(f"✅ Saved plot to {output_path}")
