import os
import pandas as pd
import nltk
from datetime import datetime

# Ensure required NLTK resources are downloaded
nltk.download("punkt")

# Paths
MATCH_PATH = "data/processed/cliche_matches.csv"
TRANSCRIPT_PATH = "data/raw/transcripts.csv"
TENURE_PATH = "data/raw/managers.csv"
OUTPUT_DIR = "data/processed"

# Load data
match_df = pd.read_csv(MATCH_PATH)
full_df = pd.read_csv(TRANSCRIPT_PATH)
tenure_df = pd.read_csv(TENURE_PATH)

# Convert date columns
full_df["publish_date"] = pd.to_datetime(full_df["publish_date"])
tenure_df["start_date"] = pd.to_datetime(tenure_df["start_date"])
tenure_df["end_date"] = pd.to_datetime(tenure_df["end_date"])

# Count words per transcript
full_df["word_count"] = full_df["transcript_text"].apply(lambda t: len(nltk.word_tokenize(t)))

# Count total cliche matches per transcript
cliche_counts = match_df.groupby("video_url").size().reset_index(name="cliche_count")
full_df = full_df.merge(cliche_counts, on="video_url", how="left")
full_df["cliche_count"] = full_df["cliche_count"].fillna(0)

# Assign manager at publish date
def assign_manager(row):
    tenures = tenure_df[tenure_df["club"] == row["club"]]
    for _, t in tenures.iterrows():
        if t["start_date"] <= row["publish_date"] and (
            pd.isna(t["end_date"]) or row["publish_date"] <= t["end_date"]
        ):
            return t["manager"]
    return "Unknown"


full_df["manager"] = full_df.apply(assign_manager, axis=1)

# Add week column and normalize cliche rate
full_df["week"] = full_df["publish_date"].dt.to_period("W").dt.start_time
full_df["cliches_per_10000_words"] = (full_df["cliche_count"] / full_df["word_count"]) * 10000

# Save weekly transcript-level data
os.makedirs(OUTPUT_DIR, exist_ok=True)
full_df.to_csv(os.path.join(OUTPUT_DIR, "cliches_by_week.csv"), index=False)

# Cliche usage by club/manager breakdown
match_df = match_df.merge(full_df[["video_url", "club", "manager"]], on="video_url", how="left")
if "club" not in match_df.columns:
    match_df["club"] = match_df["club_y"] if "club_y" in match_df.columns else match_df["club_x"]

breakdown = (
    match_df.groupby(["club", "manager", "cliche"])
    .size()
    .reset_index(name="count")
    .sort_values(["club", "manager", "count"], ascending=[True, True, False])
)
breakdown.to_csv(os.path.join(OUTPUT_DIR, "favourite_cliches.csv"), index=False)


# Manager-level summary
manager_grouped = full_df.groupby(["club", "manager"]).agg({
    "cliche_count": "sum",
    "word_count": "sum"
}).reset_index()
manager_grouped["cliches_per_10000_words"] = (
    manager_grouped["cliche_count"] / manager_grouped["word_count"] * 10000
)
manager_grouped.to_csv(os.path.join(OUTPUT_DIR, "cliches_by_manager.csv"), index=False)

# Club-level summary
club_grouped = full_df.groupby("club").agg({
    "cliche_count": "sum",
    "word_count": "sum"
}).reset_index()
club_grouped["cliches_per_10000_words"] = (
    club_grouped["cliche_count"] / club_grouped["word_count"] * 10000
)
club_grouped.to_csv(os.path.join(OUTPUT_DIR, "cliches_by_club.csv"), index=False)

# Confirm outputs
os.listdir(OUTPUT_DIR)
