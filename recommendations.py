"""
Recommendations Module
Handles personalized movie recommendations
"""

import pandas as pd
from typing import List
from .data_processing import STATE


def _get_sentiment_weight(df: pd.DataFrame) -> pd.Series:
    """Calculate weight based on description sentiment"""
    if "Description_Sentiment" in df.columns:
        return (df["Description_Sentiment"] + 1) * 5  # Convert -1 to 1 range to 0-10 scale
    return pd.Series([5.0] * len(df))  # Neutral default


def _get_genre_diversity_weight(df: pd.DataFrame) -> pd.Series:
    """Reward movies with multiple genres for diversity"""
    if "Genre_Count" in df.columns:
        return df["Genre_Count"] * 0.5  # Small bonus for genre diversity
    return pd.Series([0] * len(df))


def _apply_genre_preferences(df: pd.DataFrame, preferred_genres: List[str]) -> pd.Series:
    """Apply bonus points for preferred genres"""
    bonus = pd.Series([0.0] * len(df))
    for genre in preferred_genres:
        genre_mask = df["Genre"].str.contains(genre, case=False, na=False)
        bonus[genre_mask] += 1.0
    return bonus


def _diversify_recommendations(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Ensure genre diversity in top recommendations"""
    recommendations = []
    df = df.sort_values("Recommendation_Score", ascending=False)
    
    # Group by primary genre for diversity
    genre_groups = df.groupby("Primary_Genre")
    
    for genre, group in genre_groups:
        top_in_genre = group.head(2)  # Take top 2 from each genre
        recommendations.append(top_in_genre)
    
    # Combine and sort
    diverse_recs = pd.concat(recommendations, ignore_index=True)
    diverse_recs = diverse_recs.sort_values("Recommendation_Score", ascending=False).head(top_n)
    
    return diverse_recs


def enhanced_recommendations(movies: pd.DataFrame, user_preferences: dict = None) -> pd.DataFrame:
    """
    More sophisticated recommendations considering multiple factors
    """
    if user_preferences is None:
        user_preferences = {}
    
    df = movies.copy()
    
    # Calculate base recommendation score
    df["Recommendation_Score"] = (
        df["AI_Score"] * 0.4 + 
        pd.to_numeric(df["IMDb_Rating"], errors='coerce') * 0.3 +
        _get_sentiment_weight(df) * 0.1 +
        _get_genre_diversity_weight(df) * 0.2
    )
    
    # Apply user preferences if provided
    if user_preferences.get("preferred_genres"):
        df["Recommendation_Score"] += _apply_genre_preferences(df, user_preferences["preferred_genres"])
    
    if user_preferences.get("min_rating"):
        min_rating = user_preferences["min_rating"]
        df = df[df["AI_Score"] >= min_rating]
    
    # Ensure diversity in recommendations
    top_recommendations = _diversify_recommendations(df)
    
    return top_recommendations


def recommend_movies(user_preferences: dict = None) -> List[str]:
    """
    Enhanced recommendations with personalization
    """
    print("\n🌟 Generating enhanced recommendations...")
    movies = STATE.get("analyzed")
    if movies is None:
        raise ValueError("Run analysis first.")

    # Use enhanced recommendation engine
    top_movies = enhanced_recommendations(movies, user_preferences).head(5)

    # Format enhanced recommendations
    recommendations = []
    for _, row in top_movies.iterrows():
        rec_text = (
            f"{row['Series_Title']} ({row['AI_Score']}/10) - "
            f"{row.get('Primary_Genre', 'Unknown')} | "
            f"Strengths: {row.get('Strengths', 'N/A')} | "
            f"Audience: {row.get('Target_Audience', 'General')}"
        )
        recommendations.append(rec_text)

    return recommendations
