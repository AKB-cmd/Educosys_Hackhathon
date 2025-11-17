
# 🎬 Enhanced Movie Agent

An advanced **Agentic AI-powered** movie analysis and recommendation system. It autonomously validates, analyzes, and recommends movies using multi-step reasoning, sentiment evaluation, and personalized preferences.

## 🌟 Features

### 🤖 Advanced AI Analysis

* **Agentic AI Reasoning**: Performs autonomous, multi-step evaluations for intelligent insights.
* **Multi-Mode Analysis**: Quick, Normal, and Deep analysis modes.
* **Enhanced LLM Prompts**: Structured prompts for consistent AI evaluations.
* **Multi-Dimensional Scoring**: Evaluates strengths, weaknesses, and target audience.
* **Sentiment Analysis**: AI-powered description sentiment evaluation.
* **Robust Fallback**: Intelligent heuristic analysis when LLM unavailable.

### 📊 Data Quality & Validation

* **Comprehensive Validation**: Automatic checks for missing, inconsistent, or duplicate data.
* **Data Enrichment**: Adds derived features like genre counts, rating categories, and sentiment scores.
* **Duplicate Detection**: Smart removal of redundant entries.
* **Missing Data Handling**: Intelligent strategies to fill gaps.

### 🎯 Personalized Recommendations

* **Agentic Recommendations**: Autonomously selects top movies using multi-factor reasoning.
* **Genre Preferences**: Filter recommendations based on preferred genres.
* **Multi-Factor Scoring**: Combines AI score, IMDb rating, sentiment, and diversity.
* **Diversity-Aware**: Ensures a variety of genres in recommendations.
* **Minimum Rating Filter**: Filter out low-rated movies.

### 📈 Advanced Analytics

* **Genre Trend Analysis**: Deep dive into genre performance.
* **Correlation Analysis**: Heatmaps showing relationships between metrics.
* **Distribution Analysis**: Visualizations of scores, ratings, and sentiment.
* **Word Cloud Insights**: Visual analysis of AI reasoning.

---

## 📁 Project Structure

```
.
├── __init__.py
├── analysis.py
├── app.py
├── data_processing.py
├── recommendations.py
├── run_agent.py
├── requirements.txt
├── README.md
├── .gitignore
├── .env.example
└── IMDb dataset
```

---

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

# Optional: OpenAI API configuration (for enhanced AI/Agentic analysis)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini

# Optional: Analysis mode (quick, normal, deep)
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

---

## 📖 Usage Guide

### Web Interface

1. **Configure Analysis**

   * Choose analysis depth (Quick / Normal / Deep)
   * Select number of movies to analyze
   * Set genre preferences
   * Set minimum rating threshold

2. **Run Analysis**

   * Click **"Run Enhanced Agent Analysis"**
   * The agent autonomously analyzes, scores, and categorizes movies

3. **Explore Results**

   * **Overview**: Key metrics and recommendations
   * **Advanced Analytics**: Correlation, distribution, and genre insights
   * **Deep Insights**: Visualizations and word clouds
   * **Dataset**: Full analyzed movie data


---

## 🔧 Module Documentation

### `data_processing.py`

* `clean_and_prepare_data()`: Main data cleaning function
* `validate_movie_data()`: Comprehensive validation checks
* `enrich_movie_data()`: Adds derived features for analysis

### `analysis.py`

* `analyze_movies()`: Agentic AI-powered movie analysis
* `summarize_movies()`: Generates insights summary
* `analyze_genre_trends()`: Analyzes genre performance
* `create_correlation_analysis()`: Generates correlation heatmaps

### `recommendations.py`

* `enhanced_recommendations()`: Multi-factor recommendation engine
* `recommend_movies()`: Returns top recommended movies

### `run_agent.py`

* `run_enhanced_agent()`: Full pipeline orchestration
* `fetch_movies()`: Fetches top N movies
* `categorize_movies()`: Categorizes movies by genre

---

## 🎯 Analysis Modes

### Quick Mode

* Fast heuristic scoring
* Genre median baseline
* No LLM required

### Normal Mode

* Balanced AI analysis
* Limited to 20 movies for speed
* Moderate prompt complexity

### Deep Mode

* Comprehensive agentic AI evaluation
* Processes all movies in dataset
* Generates strengths, weaknesses, audience insights

---

## 📊 Output Files

* `enhanced_movie_analysis.csv`: Complete analyzed dataset
* `genre_analysis.csv`: Genre performance metrics
* `ai_score_by_genre.png`: Genre visualization

---

## 🛠️ Troubleshooting

### Common Issues

**1. CSV Not Found**

* Verify `IMDB_CSV_PATH` in `.env`
* Ensure CSV exists at specified location

**2. LLM/Agentic AI Not Working**

* Check `OPENAI_API_KEY` is set
* Verify internet connection
* System falls back to heuristic analysis automatically

**3. Missing Dependencies**

```bash
pip install -r requirements.txt --upgrade
```

**4. Streamlit Port Already in Use**

```bash
streamlit run app.py --server.port 8502
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

* Built for Educosys Hackathon
* IMDb dataset source
* OpenAI for LLM and agentic AI capabilities
* Streamlit for web framework

---

**Built with ❤️ for autonomous and intelligent movie analysis**

---
