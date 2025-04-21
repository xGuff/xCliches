import requests
import pandas as pd
import os
from dotenv import load_dotenv

# === Load API key from .env file ===
load_dotenv()
API_KEY = os.getenv("API_FOOTBALL_KEY")

# === Define league and season ===
LEAGUE_ID = 39  # Premier League
SEASON = 2024

# === API request ===
url = f"https://v3.football.api-sports.io/standings?league={LEAGUE_ID}&season={SEASON}"
headers = {
    "x-rapidapi-host": "v3.football.api-sports.io",
    "x-rapidapi-key": API_KEY
}
response = requests.get(url, headers=headers)
data = response.json()

# === Extract standings info ===
standings = data["response"][0]["league"]["standings"][0]

records = []
for team in standings:
    club = team["team"]["name"]
    played = team["all"]["played"]
    points = team["points"]
    ppg = points / played if played > 0 else 0
    records.append({
        "club": club,
        "points": points,
        "games": played,
        "points_per_game": round(ppg, 3)
    })

# === Save or return as DataFrame ===
df = pd.DataFrame(records)
df.to_csv("data/processed/points_per_game.csv", index=False)

print("✅ Saved latest Premier League points per game data.")
