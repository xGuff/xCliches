import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
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
WORD_COUNT_THRESHOLD = 50000

# === File Paths ===
cliche_counts_path = "data/processed/favourite_cliches.csv"
cliches_per_10k_path = "data/processed/cliches_by_club.csv"
transcripts_path = "data/raw/transcripts.csv"
output_path = "data/outputs/heatmap.png"

os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === Load data ===
df = pd.read_csv(cliche_counts_path)
ranking_df = pd.read_csv(cliches_per_10k_path)
transcripts_df = pd.read_csv(transcripts_path)

# === Compute total word counts per club from transcripts ===
transcripts_df["word_count"] = transcripts_df["transcript_text"].str.split().str.len()
word_totals = transcripts_df.groupby("club")["word_count"].sum()
valid_clubs = word_totals[word_totals >= WORD_COUNT_THRESHOLD].index.tolist()

# === Filter to valid clubs only ===
ranking_df = ranking_df[ranking_df["club"].isin(valid_clubs)]
df = df[df["club"].isin(valid_clubs)]

# === Top 5 clubs by clichés per 10k words ===
top_clubs = ranking_df.sort_values("cliches_per_10000_words", ascending=False)["club"].head(5).tolist()
df = df[df["club"].isin(top_clubs)]

# === Sum counts per (club, cliche) and normalize ===
df = df.groupby(["club", "cliche"])["count"].sum().reset_index()
df["total_words"] = df["club"].map(word_totals)
df["cliches_per_10k_words"] = (df["count"] / df["total_words"]) * 10000

# === Order clichés by total usage across top clubs ===
phrase_order = df.groupby("cliche")["cliches_per_10k_words"].sum().sort_values(ascending=False).index.tolist()

# === Pivot for heatmap ===
pivot = df.pivot(index="cliche", columns="club", values="cliches_per_10k_words").fillna(0)
pivot = pivot.loc[phrase_order]  # y-axis ordered by frequency
pivot = pivot[top_clubs]         # x-axis ordered by top 5

import numpy as np

# Create a mask where values are 0
mask = pivot == 0

# Create annotation labels only for non-zero cells
annotations = pivot.applymap(lambda v: f"{v:.2f}" if v > 0 else "")

# Plot heatmap with mask applied
plt.figure(figsize=(10, 12))
ax = sns.heatmap(
    pivot,
    annot=annotations,
    fmt="",
    cmap="plasma_r",
    mask=mask,
    linewidths=0.5,
    cbar_kws={"label": "Clichés per 10,000 Words"}
)

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image
import requests
from io import BytesIO

# Load badge mapping
badge_df = pd.read_csv("data/raw/club_badges.csv")
badge_map = badge_df.set_index("club")["badge_url"].to_dict()

# Get tick locations
xtick_locs = ax.get_xticks()

# Clear the default tick labels
ax.set_xticklabels([])

# Adjust plot padding for space below badges
plt.subplots_adjust(bottom=0.2)

# Add badge images
for i, club in enumerate(top_clubs):
    badge_url = badge_map.get(club)
    if badge_url:
        try:
            if os.path.isfile(badge_url):
                img = Image.open(badge_url)
            else:
                response = requests.get(badge_url)
                img = Image.open(BytesIO(response.content))
            img = img.convert("RGBA")
            imagebox = OffsetImage(img, zoom=0.2)  # Adjust zoom as needed
            ab = AnnotationBbox(
                imagebox,
                (xtick_locs[i], len(pivot)),
                frameon=False,
                box_alignment=(0.5, 1)
            )
            ax.add_artist(ab)
        except Exception as e:
            print(f"⚠️ Error loading badge for {club}: {e}")
ax.set_xticks([])
ax.tick_params(axis='y', which='both', length=0)
ax.set_xlabel('')
ax.set_ylabel('')
# plt.title("🎙️ Top 5 Clubs by Normalized Cliché Usage", fontsize=16)
plt.tight_layout()

# === Save ===
plt.savefig(output_path, dpi=700)
plt.close()

print(f"✅ Normalized cliché heatmap saved to {output_path}")
