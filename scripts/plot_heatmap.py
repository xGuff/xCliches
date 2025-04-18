import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Load data ===
cliche_counts_path = "data/processed/favourite_cliches.csv"
cliches_per_10k_path = "data/processed/cliches_by_club.csv"
output_path = "data/outputs/cliche_heatmap_top5.png"

# Create output directory if needed
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Load cliché counts per club
df = pd.read_csv(cliche_counts_path)

# Drop manager column and sum counts per (club, cliche)
df = df.groupby(["club", "cliche"])["count"].sum().reset_index()

# Load cliché rate to determine club order
ranking_df = pd.read_csv(cliches_per_10k_path)
top_clubs = ranking_df.sort_values("cliches_per_10000_words", ascending=False)["club"].head(5).tolist()

# Filter df to top 5 clubs only
df = df[df["club"].isin(top_clubs)]

# Pivot to wide format
pivot = df.pivot(index="cliche", columns="club", values="count").fillna(0)

# Reorder columns based on cliche rate
pivot = pivot[top_clubs]

# === Plot ===
plt.figure(figsize=(12, 8))
ax = sns.heatmap(
    pivot,
    annot=True,
    fmt=".0f",
    cmap="YlOrBr",
    linewidths=0.5,
    cbar_kws={"label": "Cliché Count"}
)

ax.set_xlabel("Club (Top 5 by clichés per 10,000 words)")
ax.set_ylabel("Cliché")
plt.xticks(rotation=45, ha="right")
plt.title("🎙️ Top 5 Clubs by Cliché Usage", fontsize=16)
plt.tight_layout()

# Save
plt.savefig(output_path)
plt.close()

print(f"✅ Top 5 club cliché heatmap saved to {output_path}")
