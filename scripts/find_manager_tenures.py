import pandas as pd
from datetime import datetime

# Constants
CSV_PATH = "data/raw/manager_tenures.csv"

# Blacklist of unwanted managers
BLACKLIST = {
    "J. Tindall",
    "A. Bertoldi",
    "G. Jones",
    "R. Mason",
    "S. Ireland",
    "G. Brazil"
}

def remove_blacklisted_managers(csv_path, blacklist):
    """Remove blacklisted managers from the dataset."""
    df = pd.read_csv(csv_path)
    cleaned_df = df[~df["manager"].isin(blacklist)]
    cleaned_df.to_csv(csv_path, index=False)
    print(f"✅ Removed {len(df) - len(cleaned_df)} blacklisted managers.")
    return cleaned_df

def update_end_dates_based_on_successors(df):
    """Update end dates based on successor start dates."""
    df["start_date"] = pd.to_datetime(df["start_date"])
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    updated_rows = []
    for club, group in df.groupby("club"):
        group_sorted = group.sort_values("start_date").reset_index(drop=True)

        for i in range(len(group_sorted) - 1):
            current = group_sorted.loc[i]
            next_start = group_sorted.loc[i + 1, "start_date"]

            # Update end_date if missing or later than the next manager's start_date
            if pd.isna(current["end_date"]) or current["end_date"] > next_start:
                group_sorted.loc[i, "end_date"] = next_start

        updated_rows.append(group_sorted)

    updated_df = pd.concat(updated_rows).sort_values(["club", "start_date"])
    return updated_df

def main():
    # Step 1: Remove blacklisted managers
    cleaned_df = remove_blacklisted_managers(CSV_PATH, BLACKLIST)

    # Step 2: Update end dates based on successors
    updated_df = update_end_dates_based_on_successors(cleaned_df)

    # Save the updated dataset
    updated_df.to_csv(CSV_PATH, index=False)
    print("✅ Successor-based end dates updated where applicable.")

if __name__ == "__main__":
    main()
