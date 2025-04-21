import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import requests
from io import BytesIO
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
import numpy as np

# === Paths ===
cliche_path = "data/processed/cliches_by_club.csv"
badges_path = "data/raw/club_badges.csv"
standings_path = "data/processed/points_per_game.csv"
output_path = "data/outputs/regression_cliches_vs_points.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# === Load data ===
cliches_df = pd.read_csv(cliche_path)
badges_df = pd.read_csv(badges_path)
standings_df = pd.read_csv(standings_path)

# === Merge ===
merged = pd.merge(cliches_df, standings_df, on="club")
merged = pd.merge(merged, badges_df, on="club")

# === Prepare regression data ===
X = merged["cliches_per_10000_words"].values.reshape(-1, 1)
y = merged["points_per_game"].values.reshape(-1, 1)

print(X, y)

# === Fit regression model ===
model = LinearRegression()
model.fit(X, y)
preds = model.predict(X)

print(f"Regression Coefficients: {model.coef_}")
print(f"Regression Intercept: {model.intercept_}")
print(f"R² Score: {model.score(X, y)}")

# === Plot ===
plt.figure(figsize=(12, 8))
ax = plt.gca()

# Plot regression line
x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_vals = model.predict(x_vals)
ax.plot(x_vals, y_vals, color="black", linestyle="--", label="Linear Fit")

# Helper: get club badge
def get_club_badge(url, zoom=0.4):
    try:
        response = requests.get(url)
        img = plt.imread(BytesIO(response.content), format='png')
        return OffsetImage(img, zoom=zoom)
    except Exception as e:
        print(f"⚠️ Failed to load badge: {url} — {e}")
        return None

# Add club badges
for _, row in merged.iterrows():
    img = get_club_badge(row["badge_url"])
    if img:
        ab = AnnotationBbox(img, (row["cliches_per_10000_words"], row["points_per_game"]),
                            frameon=False, box_alignment=(0.5, 0.5))
        ax.add_artist(ab)

# Set custom x and y limits with a 10% margin
x_min, x_max = merged["cliches_per_10000_words"].min(), merged["cliches_per_10000_words"].max()
y_min, y_max = merged["points_per_game"].min(), merged["points_per_game"].max()
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1

ax.set_xlim(x_min - x_margin, x_max + x_margin)
ax.set_ylim(y_min - y_margin, y_max + y_margin)
# Axis labels and style
ax.set_xlabel("Clichés per 10,000 Words", fontsize=13)
ax.set_ylabel("Points per Game", fontsize=13)
plt.title("Regression: Cliché Usage vs. League Performance", fontsize=16)
plt.grid(True, linestyle="--", alpha=0.3)
plt.tight_layout()

# Save
plt.savefig(output_path)
plt.close()
print(f"✅ Saved regression plot to {output_path}")
