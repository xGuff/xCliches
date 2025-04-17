import os
import time
import pandas as pd
import yaml
from pytube import Playlist
from youtube_transcript_api import YouTubeTranscriptApi

# Paths
CONFIG_PATH = "data/playlists.yaml"
OUTPUT_PATH = "data/raw/xguff_pressers_raw.csv"

# Load playlist config
with open(CONFIG_PATH, "r") as f:
    playlists_config = yaml.safe_load(f)

data = []

for club, info in playlists_config.items():
    manager = info["manager"]
    for playlist_entry in info.get("playlists", []):
        label = playlist_entry.get("label", "Unnamed Playlist")
        url = playlist_entry["url"]

        print(f"🔍 {club} ({manager}) — {label}")
        try:
            playlist = Playlist(url)
        except Exception as e:
            print(f"  ❌ Could not load playlist: {e}")
            continue

        for video_url in playlist.video_urls:
            video_id = video_url.split("v=")[-1]

            print(f"    ▶️ Fetching transcript for: {video_id}")
            try:
                transcript = YouTubeTranscriptApi.get_transcript(video_id)
                full_text = " ".join([t["text"] for t in transcript])

                data.append({
                    "club": club,
                    "manager": manager,
                    "playlist_label": label,
                    "video_id": video_id,
                    "video_url": video_url,
                    "transcript_text": full_text
                })

                time.sleep(1)

            except Exception as e:
                print(f"    ❌ Failed: {e}")

# Save output
df = pd.DataFrame(data)
os.makedirs("../data/raw", exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"\n✅ Done! Saved {len(df)} transcripts to {OUTPUT_PATH}")
