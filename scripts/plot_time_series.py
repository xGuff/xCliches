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

# Parameters
WORD_COUNT_THRESHOLD = 50000  # Minimum total words required for a club to be included

# Load data
df = pd.read_csv("data/processed/cliches_by_week.csv", parse_dates=["publish_date", "week"])
tenure_df = pd.read_csv("data/raw/managers.csv", parse_dates=["start_date", "end_date"])
badge_df = pd.read_csv("data/raw/club_badges.csv")
manager_df = pd.read_csv("data/raw/managers.csv")
transcripts_df = pd.read_csv("data/raw/transcripts.csv")

with open("data/club_colours.yaml", "r") as f:
    club_colours = yaml.safe_load(f)

# Calculate total word count per club
transcripts_df["word_count"] = transcripts_df["transcript_text"].str.split().str.len()
club_word_totals = transcripts_df.groupby("club")["word_count"].sum()

# Filter out clubs below threshold
valid_clubs = club_word_totals[club_word_totals >= WORD_COUNT_THRESHOLD].index.tolist()
df = df[df["club"].isin(valid_clubs)]

# Step 1: Weekly aggregation
weekly_avg = (
    df.groupby(["club", "week"])
    .agg({"cliche_count": "sum", "word_count": "sum"})
    .reset_index()
)

# Step 2: Full club-week grid
all_clubs = df["club"].unique()
all_weeks = df["week"].sort_values().unique()
full_index = pd.MultiIndex.from_product([all_clubs, all_weeks], names=["club", "week"])
weekly_avg = weekly_avg.set_index(["club", "week"]).reindex(full_index).reset_index()
weekly_avg["cliche_count"] = weekly_avg["cliche_count"].fillna(0)
weekly_avg["word_count"] = weekly_avg["word_count"].fillna(0)
weekly_avg["cliches_per_10000_words"] = weekly_avg.apply(
    lambda row: (row["cliche_count"] / row["word_count"]) * 10000 if row["word_count"] > 0 else 0, axis=1
)

# Step 3: Cumulative metrics + ranks
weekly_avg = weekly_avg.sort_values(["club", "week"])
weekly_avg["cum_cliche_count"] = weekly_avg.groupby("club")["cliche_count"].cumsum()
weekly_avg["cum_word_count"] = weekly_avg.groupby("club")["word_count"].cumsum()
weekly_avg["cum_cliches_per_10000_words"] = weekly_avg.apply(
    lambda row: (row["cum_cliche_count"] / row["cum_word_count"]) * 10000 if row["cum_word_count"] > 0 else 0,
    axis=1
)
weekly_avg = weekly_avg.sort_values(["week", "cum_cliches_per_10000_words", "club"], ascending=[True, False, True])
weekly_avg["rank"] = weekly_avg.groupby("week").cumcount() + 1

# Output directory
output_dir = "data/outputs/club_timeseries"
os.makedirs(output_dir, exist_ok=True)
sns.set_style("whitegrid")

def get_image_from_url(source, zoom=0.05, greyscale=False):
    try:
        if os.path.isfile(source):
            image = Image.open(source).convert("RGBA")
        else:
            response = requests.get(source)
            image = Image.open(BytesIO(response.content)).convert("RGBA")
        if greyscale:
            image = image.convert("LA").convert("RGBA")
        return OffsetImage(image, zoom=zoom)
    except Exception as e:
        print(f"⚠️ Could not load image: {source} — {e}")
        return None

def get_circular_image_with_border(source, zoom=0.4, border_thickness=6, border_color="black"):
    try:
        if os.path.isfile(source):
            image = Image.open(source).convert("RGBA")
        else:
            response = requests.get(source)
            image = Image.open(BytesIO(response.content)).convert("RGBA")

        standard_size = (128, 128)
        image = image.resize(standard_size, Image.Resampling.LANCZOS)

        size = min(image.size)
        image = image.crop((
            (image.width - size) // 2,
            (image.height - size) // 2,
            (image.width + size) // 2,
            (image.height + size) // 2,
        ))

        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)

        bordered_size = size + border_thickness * 2
        background = Image.new("RGBA", (bordered_size, bordered_size), (0, 0, 0, 0))
        border_draw = ImageDraw.Draw(background)
        border_draw.ellipse((0, 0, bordered_size, bordered_size), fill=border_color)

        image.putalpha(mask)
        background.paste(image, (border_thickness, border_thickness), mask=image)

        return OffsetImage(background, zoom=zoom)

    except Exception as e:
        print(f"⚠️ Could not load bordered circular image from {source}: {e}")
        return None

# Plotting
for club in all_clubs:
    fig, ax = plt.subplots(figsize=(14, 8), dpi=500)

    for other_club in all_clubs:
        group = weekly_avg[weekly_avg["club"] == other_club]
        color = club_colours.get(other_club) if other_club == club else "lightgrey"
        alpha = 1.0 if other_club == club else 0.4
        linewidth = 2.5 if other_club == club else 1
        ax.plot(group["week"], group["rank"], color=color, alpha=alpha, linewidth=linewidth)

        # Club badge at end
        badge_url = badge_df.loc[badge_df["club"] == other_club, "badge_url"].values
        if badge_url.size > 0:
            last_point = group.dropna(subset=["rank"]).iloc[-1]
            badge_img = get_image_from_url(badge_url[0], zoom=0.3 if other_club == club else 0.1, greyscale=(other_club != club))
            if badge_img:
                ab = AnnotationBbox(badge_img, (last_point["week"], last_point["rank"]),
                                    frameon=False, box_alignment=(0.5, 0.5), zorder=12)
                ax.add_artist(ab)

    # Manager start images
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
    ax.set_yticks(range(1, weekly_avg["rank"].max() + 1))
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()

    filename = os.path.join(output_dir, f"{club.replace(' ', '_').lower()}.png")
    plt.savefig(filename)
    plt.close()
