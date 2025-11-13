"""
Data Processing Module
Handles data cleaning, validation, and enrichment
"""

import os
import pandas as pd
from typing import Tuple, List
from dotenv import load_dotenv

try:
    from textblob import TextBlob
except ImportError:
    TextBlob = None

# Load environment variables
load_dotenv()
CSV_PATH = os.getenv("IMDB_CSV_PATH", r"D:\educosys\hackathon\imdb_top_1000.csv")

# In-memory state to avoid CSV dependency
STATE: dict = {"cleaned": None, "fetched": None, "categorized": None, "analyzed": None}


def validate_movie_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Comprehensive data validation with detailed issue reporting"""
    issues = []
    
    # Check for missing critical fields
    critical_cols = ["Series_Title", "IMDb_Rating", "Genre"]
    for col in critical_cols:
        if col not in df.columns:
            issues.append(f"Missing critical column: {col}")
        else:
            # Check for empty values in critical columns
            empty_count = df[col].isna().sum()
            if empty_count > 0:
                issues.append(f"Column '{col}' has {empty_count} empty values")
    
    # Validate rating ranges
    if "IMDb_Rating" in df.columns:
        ratings = pd.to_numeric(df["IMDb_Rating"], errors='coerce')
        invalid_ratings = ratings[~ratings.between(0, 10)].count()
        if invalid_ratings > 0:
            issues.append(f"{invalid_ratings} movies have invalid ratings (outside 0-10 range)")
        
        # Check rating distribution
        rating_stats = ratings.describe()
        if rating_stats['std'] < 0.5:
            issues.append("Warning: Ratings have low variability")
    
    # Validate genre data
    if "Genre" in df.columns:
        empty_genres = df["Genre"].isna().sum()
        if empty_genres > 0:
            issues.append(f"{empty_genres} movies missing genre information")
    
    # Check for data consistency
    if "Series_Title" in df.columns:
        title_lengths = df["Series_Title"].str.len()
        if title_lengths.max() > 100:
            issues.append("Some movie titles are unusually long")
    
    return len(issues) == 0, issues


def enrich_movie_data(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for better analysis"""
    df = df.copy()
    
    # Genre-based features
    df["Genre_Count"] = df["Genre"].str.split(",").str.len()
    df["Has_Multiple_Genres"] = df["Genre_Count"] > 1
    
    # Rating categories
    df["Rating_Category"] = pd.cut(
        df["IMDb_Rating"], 
        bins=[0, 6, 7, 8, 10], 
        labels=["Poor", "Average", "Good", "Excellent"]
    )
    
    # Text-based features
    if "Overview" in df.columns:
        df["Overview_Length"] = df["Overview"].str.len()
        df["Has_Detailed_Overview"] = df["Overview_Length"] > 100
    
    # Sentiment analysis (if TextBlob available)
    if TextBlob is not None and "Overview" in df.columns:
        df["Description_Sentiment"] = df["Overview"].apply(
            lambda x: TextBlob(str(x)).sentiment.polarity if pd.notna(x) else 0
        )
        df["Sentiment_Category"] = pd.cut(
            df["Description_Sentiment"],
            bins=[-1, -0.1, 0.1, 1],
            labels=["Negative", "Neutral", "Positive"]
        )
    
    return df


def clean_and_prepare_data(backup_path: str = None) -> pd.DataFrame:
    """Main data cleaning and preparation function"""
    print("\n🧹 Cleaning and validating IMDb dataset...")
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Detect rating column
    rating_col = next((c for c in df.columns if "imdb" in c.lower() and "rating" in c.lower()), None)
    for cand in ["IMDb_Rating", "IMDB_Rating", "Rating", "IMDB", "imdbScore"]:
        if cand in df.columns:
            rating_col = rating_col or cand
            break
    if not rating_col:
        raise ValueError("No IMDb rating column found.")
    if rating_col != "IMDb_Rating":
        df.rename(columns={rating_col: "IMDb_Rating"}, inplace=True)

    # Ensure essential columns
    for title_col in ["Series_Title", "Title", "Movie_Title", "Name"]:
        if title_col in df.columns:
            df.rename(columns={title_col: "Series_Title"}, inplace=True)
            break
    if "Series_Title" not in df.columns:
        raise ValueError("No title column found.")

    df["Genre"] = df.get("Genre", "")
    df["Overview"] = df.get("Overview", "")
    
    # Enhanced duplicate handling
    initial_count = len(df)
    df.dropna(subset=["IMDb_Rating"], inplace=True)
    df["Series_Title"] = df["Series_Title"].str.strip().str.title()
    
    # More aggressive duplicate removal
    df = df.drop_duplicates(subset=["Series_Title"], keep="first")
    df = df.reset_index(drop=True)
    final_count = len(df)
    
    if initial_count != final_count:
        print(f"⚠️ Removed {initial_count - final_count} duplicate/invalid entries")
    
    # Data validation
    is_valid, issues = validate_movie_data(df)
    if not is_valid:
        print("⚠️ Data quality issues found:")
        for issue in issues:
            print(f"   - {issue}")
    
    # Data enrichment
    df = enrich_movie_data(df)
    
    df["IMDb_Rating"] = pd.to_numeric(df["IMDb_Rating"], errors="coerce").fillna(0)
    STATE["cleaned"] = df

    if backup_path:
        df.to_csv(backup_path, index=False)
        print(f"✅ Backup saved: {backup_path}")
    
    print(f"✅ Data cleaned and enriched: {final_count} unique movies in-memory.")
    return df


def get_state():
    """Get current state"""
    return STATE
