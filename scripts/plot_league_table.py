import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import os
import requests
from io import BytesIO

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

# Paths
data_path = "data/processed/cliches_by_club.csv"  # or by_manager if preferred
badge_path = "data/raw/club_badges.csv"
output_path = "data/outputs/league_table.png"

# Load and sort data
df = pd.read_csv(data_path)
badge_df = pd.read_csv(badge_path)

# Use only the latest manager if using the manager file
if "manager" in df.columns:
    df = df.sort_values("cliches_per_10000_words", ascending=False).drop_duplicates("club")

# Rank clubs
df = df.sort_values("cliches_per_10000_words", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# Get color map values
cmap = plt.get_cmap("plasma")
colors = [cmap(i / len(df)) for i in range(len(df))]
# Setup figure
fig, ax = plt.subplots(figsize=(8, 8))
bar_width = 0.6

# Plot bars
bars = ax.barh(df["rank"], df["cliches_per_10000_words"], height=bar_width, color=colors)
# Add club badge next to each bar
for i, (club, rank, value) in enumerate(zip(df["club"], df["rank"], df["cliches_per_10000_words"])):
    badge_url = badge_df.loc[badge_df["club"] == club, "badge_url"].values
    if badge_url.size > 0:
        try:
            response = requests.get(badge_url[0])
            img = mpimg.imread(BytesIO(response.content), format='png')
            imagebox = OffsetImage(img, zoom=0.15)
            ab = AnnotationBbox(imagebox, (value, rank), frameon=False, box_alignment=(0, 0.5))
            ax.add_artist(ab)
        except:
            print(f"⚠️ Failed to load badge for {club}")

ax.tick_params(axis='y', length=0)  # Remove y tick marks but keep labels
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# Style adjustments
ax.set_yticks(df["rank"])
ax.set_yticklabels(df["rank"])  # Show league positions on the y-axis
ax.invert_yaxis()  # Rank 1 at top
ax.set_xlabel("Clichés per 10,000 Words", fontsize=12)
ax.set_ylabel("Cliché Ranking", fontsize=12)  # Add label for y-axis
plt.grid(axis="x", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.savefig(output_path)
plt.show()
