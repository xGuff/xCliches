import os
import time
import pandas as pd
import yaml
from pytube import Playlist, YouTube
from youtube_transcript_api import YouTubeTranscriptApi
from datetime import datetime, date

# Paths
CONFIG_PATH = "data/playlists.yaml"
TENURES_PATH = "data/raw/managers.csv"
OUTPUT_PATH = "data/raw/transcripts.csv"

# Season date range
SEASON_START = date(2024, 8, 1)
SEASON_END = date(2025, 6, 30)

# Load playlists config
with open(CONFIG_PATH, "r") as f:
    playlists_config = yaml.safe_load(f)

# Load manager tenures
tenures = pd.read_csv(TENURES_PATH)
tenures["start_date"] = pd.to_datetime(tenures["start_date"])
tenures["end_date"] = pd.to_datetime(tenures["end_date"], errors="coerce")

# Helper: find the manager at a club on a given date
def find_manager(club, published_date):
    published_date = pd.to_datetime(published_date)
    club_tenures = tenures[tenures["club"] == club]

    for _, row in club_tenures.iterrows():
        start = row["start_date"]
        end = row["end_date"] if pd.notna(row["end_date"]) else pd.Timestamp.today()
        if start <= published_date <= end:
            return row["manager"]
    return None

# Main loop
data = []

for club, info in playlists_config.items():
    for playlist_entry in info.get("playlists", []):
        label = playlist_entry.get("label", "Unnamed Playlist")
        url = playlist_entry["url"]

        print(f"🔍 {club} — {label}")
        try:
            playlist = Playlist(url)
        except Exception as e:
            print(f"  ❌ Could not load playlist: {e}")
            continue

        for video_url in playlist.video_urls:
            try:
                yt = YouTube(video_url)
                publish_date = yt.publish_date.date()

                if not (SEASON_START <= publish_date <= SEASON_END):
                    print(f"    ⏩ Skipping (published {publish_date})")
                    continue

                print(f"    ▶️ Fetching transcript for: {video_url}")
                transcript = YouTubeTranscriptApi.get_transcript(yt.video_id)
                full_text = " ".join([t["text"] for t in transcript])

                manager = find_manager(club, publish_date)

                data.append({
                    "club": club,
                    "manager": manager,
                    "playlist_label": label,
                    "video_id": yt.video_id,
                    "video_url": video_url,
                    "publish_date": publish_date.isoformat(),
                    "transcript_text": full_text
                })

                time.sleep(1)

            except Exception as e:
                print(f"    ❌ Failed for {video_url}: {e}")

# Save output
df = pd.DataFrame(data)
os.makedirs("data/raw", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Done! Saved {len(df)} transcripts to {OUTPUT_PATH}")
