
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed scores
df = pd.read_csv("data/processed/club_cliche_scores.csv")

# Sort by total score
df_sorted = df.sort_values("total_cliche_score", ascending=False)

print(df_sorted.head())

# Plot bar chart
plt.figure(figsize=(10, 6))
sns.barplot(x="club", y="total_cliche_score", data=df_sorted, palette="rocket")

plt.title("xGuff Total Cliché Score by Club")
plt.xlabel("Total Cliché Score")
plt.ylabel("Club")
plt.tight_layout()
plt.savefig("data/outputs/total_cliche_score_by_club.png")
plt.show()


# Remove summary column
cliche_cols = [col for col in df.select_dtypes(include='number').columns if col != "total_cliche_score"]

# Sum over all clubs
cliche_totals = df[cliche_cols].sum().sort_values(ascending=False).head(10)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x=cliche_totals.values, y=cliche_totals.index, palette="magma")

plt.title("Top 10 Most Used Clichés in Press Conferences")
plt.xlabel("Mentions (Exact + Fuzzy + Semantic)")
plt.ylabel("Cliché Phrase")
plt.tight_layout()
plt.savefig("data/outputs/top_cliches_overall.png")
plt.show()