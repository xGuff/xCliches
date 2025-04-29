import pandas as pd
import yaml
import nltk
from rapidfuzz import fuzz, process
import os

nltk.download("punkt")

# --- Config ---
TRANSCRIPT_PATH = "data/raw/transcripts.csv"
CLICHE_PATH = "data/cliches.yaml"
OUTPUT_MATCHES = "data/processed/cliche_matches.csv"
FUZZY_THRESHOLD = 95
WINDOW_SIZE = 12  # Use a fixed window size

# --- Load cliché list ---
with open(CLICHE_PATH) as f:
    cliches = yaml.safe_load(f)["cliches"]

# --- Load transcripts ---
df = pd.read_csv(TRANSCRIPT_PATH)

# --- Sliding window generator ---
def generate_windows(tokens, size):
    return [" ".join(tokens[i:i+size]) for i in range(len(tokens) - size + 1)]

# --- Fuzzy matching ---
# def match_cliches_in_transcript(text, threshold=FUZZY_THRESHOLD, window_size=WINDOW_SIZE, proximity=10):
#     tokens = nltk.word_tokenize(text.lower())
#     matches = []

#     for i in range(len(tokens) - window_size + 1):
#         window_tokens = tokens[i:i + window_size]
#         window_text = " ".join(window_tokens)

#         for cliche in cliches:
#             score = fuzz.partial_ratio(window_text, cliche)
#             if score >= threshold:
#                 matches.append({
#                     "cliche": cliche,
#                     "matched_text": window_text,
#                     "score": score,
#                     "position": i
#                 })

#     # Deduplicate based on proximity and score
#     kept = []
#     for m in matches:
#         too_close = False
#         for k in kept:
#             if (
#                 m["cliche"] == k["cliche"] and
#                 abs(m["position"] - k["position"]) < proximity
#             ):
#                 # Keep only the better one
#                 if m["score"] > k["score"]:
#                     kept.remove(k)
#                     kept.append(m)
#                 too_close = True
#                 break
#         if not too_close:
#             kept.append(m)

#     # Drop 'position' from final output
#     for m in kept:
#         m.pop("position")

#     return kept
def match_cliches_in_transcript(text, cliches, threshold=95, window_size=12):
    text_lower = text.lower()
    tokens = nltk.word_tokenize(text_lower)

    # Track character spans of each token
    char_offsets = []
    current_pos = 0
    for token in tokens:
        start = text_lower.find(token, current_pos)
        end = start + len(token)
        char_offsets.append((start, end))
        current_pos = end

    matches = []

    # Use a fixed window size
    for i in range(len(tokens) - window_size + 1):
        window_tokens = tokens[i:i + window_size]
        window_text = " ".join(window_tokens)

        result = process.extract(
            window_text,
            cliches,
            scorer=fuzz.partial_ratio,
            score_cutoff=threshold,
            limit=1
        )

        if result:
            matched_cliche, score, _ = result[0]
            char_start = char_offsets[i][0]
            char_end = char_offsets[i + window_size - 1][1]

            matches.append({
                "cliche": matched_cliche,
                "matched_text": window_text,
                "score": score,
                "char_start": char_start,
                "char_end": char_end
            })

    # Sort and deduplicate by removing overlaps
    matches.sort(key=lambda x: (-x["score"], x["char_start"]))
    deduped = []

    for m in matches:
        overlap = False
        for d in deduped:
            if not (m["char_end"] <= d["char_start"] or m["char_start"] >= d["char_end"]):
                overlap = True
                break
        if not overlap:
            deduped.append(m)

    # Remove internal span data if not needed
    for m in deduped:
        m.pop("char_start")
        m.pop("char_end")

    return deduped


# --- Run matching ---
all_matches = []

for _, row in df.iterrows():
    matches = match_cliches_in_transcript(row["transcript_text"], cliches)
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
