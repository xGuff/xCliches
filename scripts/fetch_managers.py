import requests
import pandas as pd
from datetime import datetime, date, timedelta
import os
from dotenv import load_dotenv

# Load API key
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")

headers = {
    'x-rapidapi-host': "v3.football.api-sports.io",
    'x-rapidapi-key': API_KEY
}

# Season boundaries
SEASON_START = date(2024, 8, 1)
SEASON_END = date(2025, 5, 30)
TODAY = date.today()

# Premier League team list
league_id = 39
season = 2024

teams_url = f'https://v3.football.api-sports.io/teams?league={league_id}&season={season}'
teams_res = requests.get(teams_url, headers=headers)
teams = teams_res.json().get("response", [])
team_ids = {team["team"]["id"]: team["team"]["name"] for team in teams}

# Extract and save club badge URLs
club_badges = []
for team in teams:
    team_name = team["team"]["name"]
    logo_url = team["team"]["logo"]
    club_badges.append({
        "club": team_name,
        "badge_url": logo_url
    })

# Save club badges
badges_df = pd.DataFrame(club_badges)
badges_df.to_csv("data/raw/club_badges.csv", index=False)
print(f"✅ Saved {len(badges_df)} club badges to data/raw/club_badges.csv")


# Helper function
def parse_date(d):
    try:
        return datetime.strptime(d, "%Y-%m-%d").date() if d else None
    except:
        return None

tenures = []

for team_id, team_name in team_ids.items():
    print(f"🔍 {team_name}")

    coach_url = f"https://v3.football.api-sports.io/coachs?team={team_id}"
    coach_res = requests.get(coach_url, headers=headers)

    if coach_res.status_code != 200:
        print(f"  ❌ Failed: {coach_res.json()}")
        continue

    for coach in coach_res.json().get("response", []):
        name = coach.get("name", "Unknown")
        photo_url = coach.get("photo", "")

        for job in coach.get("career", []):
            job_team_id = job.get("team", {}).get("id")
            if job_team_id != team_id:
                continue

            start_date = parse_date(job.get("start"))
            end_date = parse_date(job.get("end"))

            if not start_date:
                continue

            # Skip if tenure ended before season started
            if end_date and end_date < SEASON_START:
                continue

            # Skip if tenure started after season ends
            if start_date > SEASON_END:
                continue

            # Skip very short stints (<14 days) unless still ongoing
            if end_date and (end_date - start_date).days < 14:
                continue

            # Clamp start to SEASON_START if earlier
            effective_start = max(start_date, SEASON_START)

            tenures.append({
                "club": team_name,
                "manager": name,
                "start_date": effective_start,
                "end_date": end_date,
                "photo_url": photo_url
            })


# Save full-season manager timeline
df = pd.DataFrame(tenures)
df.sort_values(["club", "start_date"], inplace=True)
os.makedirs("data/raw", exist_ok=True)
df.to_csv("data/raw/managers.csv", index=False)

print(f"\n✅ Saved {len(df)} manager tenures active during 2024/25 season to data/raw/managers.csv")
