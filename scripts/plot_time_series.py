import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Circle
from PIL import Image, ImageDraw
import requests
from io import BytesIO
import yaml

# Load data
df = pd.read_csv("data/processed/cliches_by_week.csv", parse_dates=["publish_date", "week"])
tenure_df = pd.read_csv("data/raw/managers.csv", parse_dates=["start_date", "end_date"])
badge_df = pd.read_csv("data/raw/club_badges.csv")
manager_df = pd.read_csv("data/raw/managers.csv")

with open("data/club_colours.yaml", "r") as f:
    club_colours = yaml.safe_load(f)

# Step 1: Aggregate weekly clichés and word counts
weekly_avg = (
    df.groupby(["club", "week"])
    .agg({"cliche_count": "sum", "word_count": "sum"})
    .reset_index()
)

# Step 2: Ensure full club-week grid
all_clubs = df["club"].unique()
all_weeks = df["week"].sort_values().unique()
full_index = pd.MultiIndex.from_product([all_clubs, all_weeks], names=["club", "week"])
weekly_avg = weekly_avg.set_index(["club", "week"]).reindex(full_index).reset_index()
weekly_avg["cliche_count"] = weekly_avg["cliche_count"].fillna(0)
weekly_avg["word_count"] = weekly_avg["word_count"].fillna(0)
weekly_avg["cliches_per_10000_words"] = weekly_avg.apply(
    lambda row: (row["cliche_count"] / row["word_count"]) * 10000 if row["word_count"] > 0 else 0, axis=1
)

# Step 3: Cumulative sums and ranks
weekly_avg = weekly_avg.sort_values(["club", "week"])
weekly_avg["cum_cliche_count"] = weekly_avg.groupby("club")["cliche_count"].cumsum()
weekly_avg["cum_word_count"] = weekly_avg.groupby("club")["word_count"].cumsum()
weekly_avg["cum_cliches_per_10000_words"] = weekly_avg.apply(
    lambda row: (row["cum_cliche_count"] / row["cum_word_count"]) * 10000 if row["cum_word_count"] > 0 else 0, axis=1
)
weekly_avg = weekly_avg.sort_values(["week", "cum_cliches_per_10000_words", "club"], ascending=[True, False, True])
weekly_avg["rank"] = weekly_avg.groupby("week").cumcount() + 1

# Output setup
output_dir = "data/outputs/club_timeseries"
os.makedirs(output_dir, exist_ok=True)
sns.set_style("whitegrid")

def get_image_from_url(url, zoom=0.05, greyscale=False):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGBA")
        if greyscale:
            image = image.convert("LA").convert("RGBA")
        return OffsetImage(image, zoom=zoom)
    except Exception as e:
        print(f"⚠️ Could not load image: {url} — {e}")
        return None

def add_circular_image_with_border(ax, image, xy, border_color="black", radius=0.6):
    border_circle = Circle(xy, radius=radius, color=border_color, zorder=10)
    ax.add_patch(border_circle)
    ab = AnnotationBbox(image, xy, frameon=False, box_alignment=(0.5, 0.5), zorder=11)
    ax.add_artist(ab)

def get_circular_image_with_border(url, zoom=0.4, border_thickness=6, border_color="black"):
    try:
        response = requests.get(url)
        image = Image.open(BytesIO(response.content)).convert("RGBA")

        # Crop to square
        size = min(image.size)
        image = image.crop((
            (image.width - size) // 2,
            (image.height - size) // 2,
            (image.width + size) // 2,
            (image.height + size) // 2,
        ))

        # Create circular mask
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)

        # Create border background
        bordered_size = size + border_thickness * 2
        background = Image.new("RGBA", (bordered_size, bordered_size), (0, 0, 0, 0))
        border_draw = ImageDraw.Draw(background)
        border_draw.ellipse((0, 0, bordered_size, bordered_size), fill=border_color)

        # Apply circular mask to the image
        image.putalpha(mask)

        # Paste the image onto the border background
        background.paste(image, (border_thickness, border_thickness), mask=image)

        return OffsetImage(background, zoom=zoom)

    except Exception as e:
        print(f"⚠️ Could not load bordered circular image from {url}: {e}")
        return None

# Plot each club individually
for club in all_clubs:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=500)  # Increased DPI for higher resolution

    for other_club in all_clubs:
        group = weekly_avg[weekly_avg["club"] == other_club]
        color = club_colours.get(other_club) if other_club == club else "lightgrey"

        alpha = 1.0 if other_club == club else 0.4
        linewidth = 2.5 if other_club == club else 1
        ax.plot(group["week"], group["rank"], label=other_club if other_club == club else None,
                color=color, alpha=alpha, linewidth=linewidth)

        zoom = 0.3 if other_club == club else 0.1
        # Add badge at end of line
        badge_url = badge_df.loc[badge_df["club"] == other_club, "badge_url"].values
        if badge_url.size > 0:
            last_point = group.dropna(subset=["rank"]).iloc[-1]
            badge_img = get_image_from_url(badge_url[0], zoom=zoom, greyscale=(other_club != club))
            if badge_img:
                ab = AnnotationBbox(badge_img, (last_point["week"], last_point["rank"]),
                                    frameon=False, box_alignment=(0.5, 0.5), zorder=12)
                ax.add_artist(ab)

    # Add manager start markers + circular photo overlays
    starts = tenure_df[(tenure_df["club"] == club) & (tenure_df["start_date"].notna())]

    for _, row in starts.iterrows():
        start_date = pd.to_datetime(row["start_date"])
        manager = row["manager"]

        manager_url = manager_df.loc[manager_df["manager"] == manager, "photo_url"].values
        if manager_url.size > 0:
            img = get_circular_image_with_border(manager_url[0], zoom=0.4, border_color=club_colours.get(club))
            if img:
                future = weekly_avg[(weekly_avg["club"] == club) & (weekly_avg["week"] >= start_date)]
                if not future.empty:
                    week = min(all_weeks, key=lambda d: abs(d - start_date))
                    y_val = future.iloc[0]["rank"]
                    ab = AnnotationBbox(img, (week, y_val), frameon=False, box_alignment=(0.5, 0.5), zorder=11)
                    ax.add_artist(ab)

    ax.invert_yaxis()
    ax.set_yticks(range(1, weekly_avg["rank"].max() + 1))  # Set y-tick labels in increments of 1
    ax.grid(False)  # Remove the background grid
    ax.spines['top'].set_visible(False)  # Turn off top border
    ax.spines['right'].set_visible(False)  # Turn off right border
    ax.spines['left'].set_visible(False)  # Turn off left border
    ax.spines['bottom'].set_visible(False)  # Turn off bottom border
    fig.tight_layout()

    filename = os.path.join(output_dir, f"{club.replace(' ', '_').lower()}.png")
    plt.savefig(filename)
    plt.close()
