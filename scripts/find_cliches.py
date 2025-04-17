import pandas as pd
import yaml
import nltk
from rapidfuzz import fuzz
import os

nltk.download("punkt")

# --- Config ---
TRANSCRIPT_PATH = "data/raw/xguff_pressers_raw.csv"
CLICHE_PATH = "data/cliches.yaml"
OUTPUT_MATCHES = "data/processed/cliche_matches.csv"
FUZZY_THRESHOLD = 95
WINDOW_SIZE = 8  # Use a fixed window size

# --- Load cliché list ---
with open(CLICHE_PATH) as f:
    cliches = yaml.safe_load(f)["cliches"]

# --- Load transcripts ---
df = pd.read_csv(TRANSCRIPT_PATH)

# --- Sliding window generator ---
def generate_windows(tokens, size):
    return [" ".join(tokens[i:i+size]) for i in range(len(tokens) - size + 1)]

# --- Fuzzy matching ---
def match_cliches_in_transcript(text, threshold=FUZZY_THRESHOLD, window_size=WINDOW_SIZE, proximity=10):
    tokens = nltk.word_tokenize(text.lower())
    matches = []

    for i in range(len(tokens) - window_size + 1):
        window_tokens = tokens[i:i + window_size]
        window_text = " ".join(window_tokens)

        for cliche in cliches:
            score = fuzz.partial_ratio(window_text, cliche)
            if score >= threshold:
                matches.append({
                    "cliche": cliche,
                    "matched_text": window_text,
                    "score": score,
                    "position": i
                })

    # Deduplicate based on proximity and score
    kept = []
    for m in matches:
        too_close = False
        for k in kept:
            if (
                m["cliche"] == k["cliche"] and
                abs(m["position"] - k["position"]) < proximity
            ):
                # Keep only the better one
                if m["score"] > k["score"]:
                    kept.remove(k)
                    kept.append(m)
                too_close = True
                break
        if not too_close:
            kept.append(m)

    # Drop 'position' from final output
    for m in kept:
        m.pop("position")

    return kept

# --- Run matching ---
print("🔍 Matching clichés using a fixed window size...")
all_matches = []

for _, row in df.iterrows():
    matches = match_cliches_in_transcript(row["transcript_text"])
    for m in matches:
        m.update({
            "club": row["club"],
            "publish_date": row["publish_date"],
            "video_url": row["video_url"]
        })
        all_matches.append(m)

# --- Save output ---
os.makedirs(os.path.dirname(OUTPUT_MATCHES), exist_ok=True)
pd.DataFrame(all_matches).to_csv(OUTPUT_MATCHES, index=False)

print(f"✅ Done! Saved fuzzy cliché matches to: {OUTPUT_MATCHES}")

import numpy as np
from collections import defaultdict

# Convert to DataFrame
match_df = pd.DataFrame(all_matches)

# Count cliché mentions per club
summary = (
    match_df.groupby(["club", "cliche"])
    .size()
    .reset_index(name="count")
)

# Pivot to wide format
summary_pivot = summary.pivot(index="club", columns="cliche", values="count").fillna(0)

# --- Word counts ---
# Load full transcripts to normalize
full_df = pd.read_csv("data/raw/xguff_pressers_raw.csv")

word_counts = (
    full_df.groupby("club")["transcript_text"]
    .apply(lambda texts: sum(len(nltk.word_tokenize(t)) for t in texts))
    .rename("total_words")
)

# --- Add total cliché counts ---
summary_pivot["total_cliche_score"] = summary_pivot.sum(axis=1)
summary_pivot["total_words"] = word_counts
summary_pivot["cliches_per_1000_words"] = summary_pivot["total_cliche_score"] / summary_pivot["total_words"] * 1000

# Reset index for plotting
summary_pivot = summary_pivot.reset_index()

# Save it
os.makedirs("data/processed", exist_ok=True)
summary_pivot.to_csv("data/processed/club_cliche_scores_normalized.csv", index=False)

print("✅ Saved summarized club cliché scores to data/processed/club_cliche_scores_normalized.csv")
