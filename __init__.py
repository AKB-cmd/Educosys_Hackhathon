"""
Enhanced Movie Agent - An AI-powered movie analysis and recommendation system
"""

from .data_processing import (
    clean_and_prepare_data,
    enrich_movie_data,
    validate_movie_data
)

from .analysis import (
    analyze_movies,
    summarize_movies,
    analyze_genre_trends,
    create_correlation_analysis
)

from .recommendations import (
    enhanced_recommendations,
    recommend_movies
)

from .run_agent import (
    run_enhanced_agent,
    fetch_movies,
    categorize_movies
)

__version__ = "2.0.0"
__all__ = [
    # Data Processing
    'clean_and_prepare_data',
    'enrich_movie_data',
    'validate_movie_data',
    
    # Analysis
    'analyze_movies',
    'summarize_movies',
    'analyze_genre_trends',
    'create_correlation_analysis',
    
    # Recommendations
    'enhanced_recommendations',
    'recommend_movies',
    
    # Agent Orchestration
    'run_enhanced_agent',
    'fetch_movies',
    'categorize_movies'
]
