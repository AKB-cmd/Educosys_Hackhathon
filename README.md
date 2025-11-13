# 🎬 Enhanced Movie Agent

An advanced AI-powered movie analysis and recommendation system with comprehensive data validation, sentiment analysis, and personalized recommendations.

## 🌟 Features

### 🤖 Advanced AI Analysis
- **Multi-Mode Analysis**: Quick, Normal, and Deep analysis modes
- **Enhanced LLM Prompts**: Structured prompts for consistent AI evaluations
- **Multi-Dimensional Scoring**: Evaluates strengths, weaknesses, and target audience
- **Sentiment Analysis**: AI-powered description sentiment evaluation
- **Robust Fallback**: Intelligent heuristic analysis when LLM unavailable

### 📊 Data Quality & Validation
- **Comprehensive Validation**: Automatic data quality checks
- **Data Enrichment**: Genre counts, rating categories, sentiment scores
- **Duplicate Detection**: Smart duplicate removal
- **Missing Data Handling**: Intelligent filling strategies

### 🎯 Personalized Recommendations
- **Genre Preferences**: Filter by preferred genres
- **Multi-Factor Scoring**: Combines AI score, IMDb rating, sentiment, and diversity
- **Diversity-Aware**: Ensures genre variety in recommendations
- **Minimum Rating Filter**: Set quality thresholds

### 📈 Advanced Analytics
- **Genre Trend Analysis**: Deep dive into genre performance
- **Correlation Analysis**: Visual heatmaps of metric relationships
- **Distribution Analysis**: Score and rating distributions
- **Word Cloud Insights**: Visual analysis of AI reasoning

## 📁 Project Structure

```
enhanced-movie-agent/
│
├── movie_agent/
│   ├── __init__.py                # Package initialization
│   ├── data_processing.py         # Data cleaning, validation, enrichment
│   ├── analysis.py                # AI analysis and insights
│   ├── recommendations.py         # Recommendation engine
│   └── run_agent.py               # Pipeline orchestration
│
├── app.py                         # Streamlit web interface
├── requirements.txt               # Python dependencies
├── README.md                      # This file
├── .env.example                   # Environment variables template
├── .gitignore                     # Git ignore rules
└── assets/
    └── imdb_top_1000.csv         # Sample dataset
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd enhanced-movie-agent

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the project root:

```env
# Required: Path to your IMDb CSV file
IMDB_CSV_PATH=./assets/imdb_top_1000.csv

# Optional: OpenAI API configuration (for enhanced AI analysis)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Optional: Analysis mode
ANALYSIS_MODE=deep
```

### 3. Run the Application

**Web Interface (Recommended):**
```bash
streamlit run app.py
```

**Command Line:**
```bash
python -m movie_agent.run_agent
```

## 📖 Usage Guide

### Web Interface

1. **Configure Analysis**
   - Select analysis depth (Quick/Normal/Deep)
   - Choose number of movies to analyze
   - Set genre preferences
   - Set minimum rating threshold

2. **Run Analysis**
   - Click "Run Enhanced Agent Analysis"
   - Wait for processing to complete

3. **Explore Results**
   - **Overview**: Key metrics and recommendations
   - **Advanced Analytics**: Correlation and genre analysis
   - **Deep Insights**: Visualizations and word clouds
   - **Dataset**: Detailed movie information and reasoning

### Python API

```python
from movie_agent import (
    clean_and_prepare_data,
    fetch_movies,
    categorize_movies,
    analyze_movies,
    recommend_movies,
    run_enhanced_agent
)

# Run the complete pipeline
final_state, actions_log = run_enhanced_agent(
    mode="deep",
    user_preferences={
        "preferred_genres": ["Drama", "Thriller"],
        "min_rating": 7.0
    }
)

# Access results
movies = final_state["movies"]
recommendations = final_state["recommendations"]
genre_analysis = final_state["genre_analysis"]
```

## 🔧 Module Documentation

### `data_processing.py`
- `clean_and_prepare_data()`: Main data cleaning function
- `validate_movie_data()`: Comprehensive data validation
- `enrich_movie_data()`: Add derived features

### `analysis.py`
- `analyze_movies()`: AI-powered movie analysis
- `summarize_movies()`: Generate insights summary
- `analyze_genre_trends()`: Genre performance analysis
- `create_correlation_analysis()`: Correlation visualization

### `recommendations.py`
- `enhanced_recommendations()`: Multi-factor recommendation engine
- `recommend_movies()`: Generate top recommendations

### `run_agent.py`
- `run_enhanced_agent()`: Complete pipeline orchestration
- `fetch_movies()`: Fetch top N movies
- `categorize_movies()`: Genre categorization

## 🎯 Analysis Modes

### Quick Mode
- Fast heuristic-based scoring
- Genre median baseline
- No LLM required

### Normal Mode
- Balanced AI analysis
- Limited to 20 movies for speed
- Moderate prompt complexity

### Deep Mode
- Comprehensive AI evaluation
- Analyzes all movies
- Enhanced prompts with detailed criteria
- Generates strengths, weaknesses, and audience info

## 📊 Output Files

The system generates:
- `enhanced_movie_analysis.csv`: Complete analyzed dataset
- `genre_analysis.csv`: Genre performance metrics
- `ai_score_by_genre.png`: Genre visualization

## 🛠️ Troubleshooting

### Common Issues

**1. CSV Not Found**
- Verify `IMDB_CSV_PATH` in `.env`
- Ensure CSV exists at specified location

**2. LLM Not Working**
- Check `OPENAI_API_KEY` is set correctly
- Verify internet connection
- System falls back to heuristics automatically

**3. Missing Dependencies**
```bash
pip install -r requirements.txt --upgrade
```

**4. Streamlit Port Already in Use**
```bash
streamlit run app.py --server.port 8502
```

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- Built for Educosys Hackathon
- IMDb dataset source
- OpenAI for LLM capabilities
- Streamlit for web framework

---

**Built with ❤️ for intelligent movie analysis**
