import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import requests
from io import BytesIO

# Paths
data_path = "data/processed/cliches_by_club.csv"  # or by_manager if preferred
badge_path = "data/raw/club_badges.csv"
output_path = "data/outputs/xguff_league_table_bar_chart.png"

# Load and sort data
df = pd.read_csv(data_path)
badge_df = pd.read_csv(badge_path)

# Use only the latest manager if using the manager file
if "manager" in df.columns:
    df = df.sort_values("cliches_per_10000_words", ascending=False).drop_duplicates("club")

# Rank clubs
df = df.sort_values("cliches_per_10000_words", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# Setup figure
fig, ax = plt.subplots(figsize=(10, 12))
bar_width = 0.6

# Plot bars
bars = ax.barh(df["rank"], df["cliches_per_10000_words"], color="skyblue", edgecolor="black", height=bar_width)

# Add club badge next to each bar
for i, (club, rank) in enumerate(zip(df["club"], df["rank"])):
    badge_url = badge_df.loc[badge_df["club"] == club, "badge_url"].values
    if badge_url.size > 0:
        try:
            response = requests.get(badge_url[0])
            img = mpimg.imread(BytesIO(response.content), format='png')
            imagebox = OffsetImage(img, zoom=0.1)
            ab = AnnotationBbox(imagebox, (0, rank), frameon=False, box_alignment=(0.5, 0.5))
            ax.add_artist(ab)
        except:
            print(f"⚠️ Failed to load badge for {club}")

# Style adjustments
ax.set_yticks(df["rank"])
ax.set_yticklabels([])  # Hide text labels since we have badges
ax.invert_yaxis()  # Rank 1 at top
ax.set_xlabel("Clichés per 10,000 Words", fontsize=12)
ax.set_title("🎙️ xGuff League Table – Clichés per 10,000 Words", fontsize=14)
plt.grid(axis="x", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(output_path)
plt.show()
