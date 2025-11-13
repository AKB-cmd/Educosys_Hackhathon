"""
Agent Orchestration Module
Coordinates the entire ETL and analysis pipeline
"""

import os
import pandas as pd
from typing import List
from dotenv import load_dotenv

from .data_processing import clean_and_prepare_data, STATE
from .analysis import analyze_movies, summarize_movies, analyze_genre_trends, create_correlation_analysis
from .recommendations import recommend_movies

# Load config
load_dotenv()
DEFAULT_TOP_N = 50


def fetch_movies(top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
    """Fetch top N movies from cleaned data"""
    print("\n🎬 Fetching top movies...")
    movies = STATE.get("cleaned")
    if movies is None:
        raise ValueError("Run cleaning first.")
    df = movies.head(max(int(top_n), 1))
    STATE["fetched"] = df
    return df


def categorize_movies() -> pd.DataFrame:
    """
    Categorize movies by extracting primary genre.
    Does NOT explode genres to avoid duplicates.
    """
    print("\n🎭 Categorizing movies by genre...")
    movies = STATE.get("fetched")
    if movies is None:
        raise ValueError("Run fetch first.")
    
    df = movies.copy()
    df["Genre"] = df["Genre"].astype(str).str.strip()
    # Extract primary genre (first one in the comma-separated list)
    df["Primary_Genre"] = df["Genre"].str.split(",").str[0].str.strip()
    
    STATE["categorized"] = df
    return df


def run_enhanced_agent(mode: str = "deep", user_role: str = "expert", user_preferences: dict = None):
    """
    Enhanced agent with data validation and improved analysis
    """
    print("\n🚀 Running Enhanced IMDb ETL + AI Pipeline...")
    actions_log: List[str] = []
    final_state = {
        "movies": pd.DataFrame(), 
        "recommendations": [], 
        "summary": "",
        "genre_analysis": pd.DataFrame(),
        "correlation_analysis": None
    }

    try:
        # Step 1: Enhanced Cleaning with Validation
        actions_log.append("🧹 Cleaning and validating data")
        clean_and_prepare_data()
        
        # Step 2: Fetch
        actions_log.append("🎬 Fetching movies")
        fetch_movies()
        
        # Step 3: Categorize
        actions_log.append("🎭 Categorizing movies by genre")
        categorize_movies()
        
        # Step 4: Enhanced Analysis
        actions_log.append("🧠 Enhanced AI analysis")
        analyzed_movies = analyze_movies(mode)
        
        # Step 5: Advanced Analytics
        actions_log.append("📈 Running advanced analytics")
        genre_trends = analyze_genre_trends(analyzed_movies)
        correlation_chart = create_correlation_analysis(analyzed_movies)
        
        # Step 6: Enhanced Recommendations
        actions_log.append("🌟 Personalized recommendations")
        recs = recommend_movies(user_preferences)
        
        # Step 7: Summarize
        actions_log.append("📊 Summarizing insights")
        summary_text = summarize_movies()

        final_state.update({
            "movies": analyzed_movies, 
            "summary": summary_text, 
            "recommendations": recs,
            "genre_analysis": genre_trends,
            "correlation_analysis": correlation_chart
        })
        
        print("✅ Enhanced analysis pipeline completed successfully!")
        
    except Exception as e:
        actions_log.append(f"❌ Pipeline failed: {str(e)}")
        raise
    
    return final_state, actions_log


# Main entry point for standalone execution
if __name__ == "__main__":
    state, log = run_enhanced_agent(
        mode=os.getenv("ANALYSIS_MODE", "deep"), 
        user_role="expert",
        user_preferences={"preferred_genres": ["Drama", "Thriller"]}
    )
    
    print("\n🧠 ENHANCED FINAL SUMMARY:\n", state["summary"])
    print("\n📊 GENRE ANALYSIS:\n", state["genre_analysis"].head())
    print("\n🎯 ENHANCED RECOMMENDATIONS:")
    for i, r in enumerate(state["recommendations"], 1):
        print(f"{i}. {r}")
