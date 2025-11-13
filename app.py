import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import os
import numpy as np

# Import enhanced functions from movie_agent
from movie_agent import (
    clean_and_prepare_data,
    fetch_movies,
    categorize_movies,
    analyze_movies,
    summarize_movies,
    recommend_movies,
    validate_movie_data,
    enrich_movie_data,
    enhanced_recommendations,
    analyze_genre_trends,
    create_correlation_analysis,
    run_enhanced_agent
)

sns.set_theme(style="whitegrid", palette="deep")

# ===================== STREAMLIT SETUP =====================
st.set_page_config(page_title="🎬 Enhanced Movie Agent", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
.main-header { 
    font-size: 3rem; 
    font-weight: 700; 
    background: linear-gradient(120deg, #e63946, #f77f00, #fcbf49);
    -webkit-background-clip: text; 
    -webkit-text-fill-color: transparent; 
    text-align: center; 
    margin-bottom: 1rem; 
}
.subtitle { 
    text-align: center; 
    color: #6c757d; 
    font-size: 1.1rem; 
    margin-bottom: 2rem; 
}
.metric-container { 
    background: white; 
    padding: 1.5rem; 
    border-radius: 10px; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
    text-align: center; 
}
.metric-value { 
    font-size: 2.5rem; 
    font-weight: 700; 
    color: #667eea; 
}
.metric-label { 
    font-size: 1rem; 
    color: #6c757d; 
    text-transform: uppercase; 
    letter-spacing: 1px; 
}
.recommendation-card { 
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); 
    padding: 1.2rem; 
    border-radius: 12px; 
    margin: 0.8rem 0; 
    color: white; 
    box-shadow: 0 4px 12px rgba(0,0,0,0.1); 
}
.workflow-step { 
    background: #f8f9fa; 
    padding: 1rem 1.5rem; 
    border-radius: 8px; 
    margin: 0.5rem 0; 
    border-left: 4px solid #667eea; 
    font-size: 1rem; 
    color: #2c3e50; 
}
.section-header { 
    font-size: 1.8rem; 
    font-weight: 700; 
    color: #2c3e50; 
    margin: 2rem 0 1rem 0; 
    padding-bottom: 0.5rem; 
    border-bottom: 3px solid #667eea; 
}
.enhanced-feature { 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🎥 Enhanced Movie Agent Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered movie analysis with enhanced data validation and personalized recommendations</p>', unsafe_allow_html=True)

# ===================== Enhanced Sidebar =====================
st.sidebar.markdown("<h3 style='text-align:center;'>⚙️ Enhanced Analysis Configuration</h3>", unsafe_allow_html=True)

# Analysis Mode
mode = st.sidebar.selectbox(
    "Choose Analysis Depth",
    ["quick", "normal", "deep"],
    index=1,
    help="Quick: Fast evaluation | Normal: Balanced | Deep: Detailed reasoning with enhanced prompts"
)

# Number of movies
top_n = st.sidebar.slider(
    "Number of Movies to Analyze",
    min_value=10,
    max_value=100,
    value=50,
    step=10,
    help="Select how many top movies to process"
)

# User Preferences for Enhanced Recommendations
st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Recommendation Preferences")

preferred_genres = st.sidebar.multiselect(
    "Preferred Genres",
    ["Action", "Drama", "Comedy", "Thriller", "Romance", "Sci-Fi", "Horror", "Adventure", "Mystery", "Fantasy"],
    help="Select genres you prefer for personalized recommendations"
)

min_rating = st.sidebar.slider(
    "Minimum AI Score",
    min_value=0.0,
    max_value=10.0,
    value=7.0,
    step=0.5,
    help="Filter recommendations by minimum AI score"
)

user_preferences = {
    "preferred_genres": preferred_genres,
    "min_rating": min_rating
}

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Data Quality Inspector")

# ===================== Enhanced Data Validation =====================
@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

if st.sidebar.checkbox("Show Enhanced CSV Diagnostics"):
    try:
        csv_path = os.getenv("IMDB_CSV_PATH", r"D:\educosys\hackathon\imdb_top_1000.csv")
        df_raw = load_csv(csv_path)
        if df_raw is not None:
            st.sidebar.info(f"**Raw CSV rows:** {len(df_raw)}")
            
            # Enhanced validation
            is_valid, issues = validate_movie_data(df_raw)
            if is_valid:
                st.sidebar.success("✅ Data validation passed")
            else:
                st.sidebar.error("❌ Data validation issues:")
                for issue in issues:
                    st.sidebar.write(f"• {issue}")
            
            # Check for duplicate titles
            title_cols = [c for c in df_raw.columns if 'title' in c.lower() or 'name' in c.lower()]
            if title_cols:
                title_col = title_cols[0]
                dupes = df_raw[title_col].duplicated().sum()
                if dupes > 0:
                    st.sidebar.error(f"⚠️ {dupes} duplicate titles in CSV!")
                else:
                    st.sidebar.success("✅ No duplicates in CSV")
        else:
            st.sidebar.error("CSV file not found")
    except Exception as e:
        st.sidebar.error(f"Error: {e}")

# Enhanced help information
st.sidebar.info("""
**Enhanced Features:**
- **Data Validation**: Comprehensive quality checks
- **Sentiment Analysis**: AI-powered description analysis  
- **Personalized Recommendations**: Based on your genre preferences
- **Advanced Analytics**: Correlation analysis and genre trends
""")

# Show current settings
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Current Settings")
if st.session_state.get("movies") is not None and not st.session_state.movies.empty:
    st.sidebar.success(f"✅ {len(st.session_state.movies)} movies analyzed")
    current_settings = f"{mode}_{top_n}_{len(preferred_genres)}"
    if st.session_state.get("last_settings", "") != current_settings:
        st.sidebar.warning("⚠️ Settings changed - click Run to update")
else:
    st.sidebar.warning("⏳ No analysis run yet")

# ===================== Initialize Enhanced Session State =====================
if "movies" not in st.session_state:
    st.session_state.movies = pd.DataFrame()
if "recommendations" not in st.session_state:
    st.session_state.recommendations = []
if "actions_log" not in st.session_state:
    st.session_state.actions_log = []
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""
if "genre_analysis" not in st.session_state:
    st.session_state.genre_analysis = pd.DataFrame()
if "correlation_analysis" not in st.session_state:
    st.session_state.correlation_analysis = None

# ===================== RUN ENHANCED AGENT =====================
settings_key = f"{mode}_{top_n}_{len(preferred_genres)}"
if "last_settings" not in st.session_state:
    st.session_state.last_settings = ""

if st.button("🚀 Run Enhanced Agent Analysis", use_container_width=True, type="primary"):

    # Clear previous results
    st.session_state.actions_log = []
    st.session_state.movies = pd.DataFrame()
    st.session_state.recommendations = []
    st.session_state.summary_text = ""
    st.session_state.genre_analysis = pd.DataFrame()
    st.session_state.correlation_analysis = None
    st.session_state.last_settings = settings_key

    try:
        with st.spinner(f"🤖 Running enhanced agent in **{mode}** mode with {top_n} movies..."):
            
            # Use the enhanced agent pipeline
            final_state, actions_log = run_enhanced_agent(
                mode=mode, 
                user_role="expert",
                user_preferences=user_preferences
            )
            
            # Store all results in session state
            st.session_state.actions_log = actions_log
            st.session_state.movies = final_state["movies"]
            st.session_state.recommendations = final_state["recommendations"]
            st.session_state.summary_text = final_state["summary"]
            st.session_state.genre_analysis = final_state["genre_analysis"]
            st.session_state.correlation_analysis = final_state["correlation_analysis"]

        st.success("✅ Enhanced agent workflow completed successfully!")
        
        # Show enhanced features used
        with st.expander("🔧 Enhanced Features Used", expanded=True):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown('<div class="enhanced-feature">🎯 Personalized Recommendations</div>', unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="enhanced-feature">📊 Data Validation</div>', unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="enhanced-feature">🧠 Enhanced AI Analysis</div>', unsafe_allow_html=True)

    except FileNotFoundError as e:
        st.error(f"❌ Input file missing: {e}")
    except Exception as e:
        st.error(f"❌ An error occurred: {str(e)}")
        st.exception(e)

# ===================== DISPLAY ENHANCED RESULTS =====================
movies = st.session_state.movies
recs = st.session_state.recommendations
actions_log = st.session_state.actions_log
genre_analysis = st.session_state.genre_analysis
correlation_chart = st.session_state.correlation_analysis

if not movies.empty:
    # Ensure consistent column naming and handle enhanced columns
    if "AI_Score" not in movies.columns and "AI Score" in movies.columns:
        movies = movies.rename(columns={"AI Score": "AI_Score"})
    if "IMDb_Rating" not in movies.columns and "IMDb Rating" in movies.columns:
        movies = movies.rename(columns={"IMDb Rating": "IMDb_Rating"})

    # Ensure Primary_Genre exists
    if "Primary_Genre" not in movies.columns:
        movies["Primary_Genre"] = (
            movies["Genre"]
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
        )

    # Create enhanced tabs
    overview_tab, analytics_tab, insights_tab, dataset_tab = st.tabs(
        ["📊 Overview", "📈 Advanced Analytics", "🔍 Deep Insights", "📋 Dataset & Reasoning"]
    )

    # -------------------- Enhanced Overview Tab --------------------
    with overview_tab:
        st.markdown('<p class="section-header">🔄 Enhanced Agent Workflow</p>', unsafe_allow_html=True)
        if actions_log:
            progress_bar = st.progress(0.0)
            for idx, entry in enumerate(actions_log, 1):
                st.markdown(f'<div class="workflow-step">{entry}</div>', unsafe_allow_html=True)
                progress_bar.progress(min(idx / len(actions_log), 1.0))
        else:
            st.info("No workflow actions recorded. Click 'Run Enhanced Agent Analysis' to begin.")

        st.markdown('<p class="section-header">📊 Enhanced Key Insights</p>', unsafe_allow_html=True)
        
        # Enhanced metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Movies Analyzed", len(movies))
        with col2:
            avg_ai_score = movies['AI_Score'].mean()
            st.metric("Avg AI Score", f"{avg_ai_score:.2f}/10")
        with col3:
            avg_imdb = movies['IMDb_Rating'].mean()
            st.metric("Avg IMDb Rating", f"{avg_imdb:.2f}/10")
        with col4:
            unique_genres = movies["Genre"].astype(str).str.split(",").explode().nunique()
            st.metric("Unique Genres", unique_genres)

        # Additional enhanced metrics
        col5, col6, col7, col8 = st.columns(4)
        with col5:
            avg_delta = movies.get('AI_Delta', pd.Series([0] * len(movies))).mean()
            st.metric("Avg Score Delta", f"{avg_delta:+.2f}")
        with col6:
            if "Genre_Count" in movies.columns:
                avg_genres = movies['Genre_Count'].mean()
                st.metric("Avg Genres per Movie", f"{avg_genres:.1f}")
            else:
                st.metric("Multi-Genre Movies", "N/A")
        with col7:
            if "Description_Sentiment" in movies.columns:
                avg_sentiment = movies['Description_Sentiment'].mean()
                st.metric("Avg Sentiment", f"{avg_sentiment:+.2f}")
            else:
                st.metric("Sentiment Analysis", "N/A")
        with col8:
            high_quality = len(movies[movies['AI_Score'] >= 8.0])
            st.metric("High Quality (8.0+)", high_quality)

        st.markdown('<p class="section-header">🎯 Enhanced Movie Recommendations</p>', unsafe_allow_html=True)
        if recs:
            for i, rec in enumerate(recs, 1):
                st.markdown(
                    f'<div class="recommendation-card"><strong>#{i}</strong> {rec}</div>', 
                    unsafe_allow_html=True
                )
            
            # Show recommendation rationale
            with st.expander("🤔 Why these recommendations?"):
                st.info("""
                These recommendations are generated using:
                - **AI Score** (40% weight): Our enhanced AI evaluation
                - **IMDb Rating** (30% weight): Community consensus
                - **Genre Preferences** (20% weight): Your selected genres
                - **Diversity** (10% weight): Ensuring genre variety
                """)
        else:
            st.info("Run the agent to generate enhanced recommendations.")

    # -------------------- Enhanced Analytics Tab --------------------
    with analytics_tab:
        st.markdown('<p class="section-header">📈 Correlation Analysis</p>', unsafe_allow_html=True)
        
        if correlation_chart is not None:
            st.pyplot(correlation_chart)
            st.caption("Heatmap showing relationships between different movie metrics")
        else:
            st.info("Correlation analysis not available. Run analysis to generate.")
        
        st.markdown('<p class="section-header">📊 Genre Performance Trends</p>', unsafe_allow_html=True)
        
        if not genre_analysis.empty:
            # Display top performing genres
            st.dataframe(genre_analysis.head(10), use_container_width=True)
            
            # Genre performance visualization
            col1, col2 = st.columns(2)
            
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                top_genres = genre_analysis.head(10)
                ax.barh(top_genres.index, top_genres['AI_Score_mean'])
                ax.set_xlabel('Average AI Score')
                ax.set_title('Top Genres by AI Score')
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(genre_analysis['AI_Score_mean'], genre_analysis['Movie_Count'])
                ax.set_xlabel('Average AI Score')
                ax.set_ylabel('Number of Movies')
                ax.set_title('Genre Popularity vs Quality')
                # Add genre labels for top genres
                for i, genre in enumerate(genre_analysis.head(8).index):
                    ax.annotate(genre, (genre_analysis.loc[genre, 'AI_Score_mean'], genre_analysis.loc[genre, 'Movie_Count']))
                st.pyplot(fig)
        else:
            st.info("Genre trend analysis not available. Run analysis to generate.")

    # -------------------- Enhanced Insights Tab --------------------
    with insights_tab:
        st.markdown('<p class="section-header">🎭 Genre Distribution Analysis</p>', unsafe_allow_html=True)
        
        col_scores, col_scatter = st.columns(2)

        with col_scores:
            exploded = movies.assign(Genre=movies["Genre"].astype(str).str.split(",")).explode("Genre")
            exploded["Genre"] = exploded["Genre"].str.strip()
            avg_scores = exploded.groupby("Genre")["AI_Score"].mean().sort_values(ascending=False).head(15)

            if avg_scores.empty:
                st.info("Not enough genre data to plot.")
            else:
                fig, ax = plt.subplots(figsize=(10, 8))
                sns.barplot(x=avg_scores.values, y=avg_scores.index, ax=ax, palette="viridis")
                ax.set_xlabel("Average AI Score")
                ax.set_ylabel("Genre")
                ax.set_title("Top Genres by Average AI Score")
                st.pyplot(fig)

        with col_scatter:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = sns.scatterplot(
                x=pd.to_numeric(movies["IMDb_Rating"], errors="coerce"),
                y=movies["AI_Score"],
                hue=movies["Primary_Genre"],
                palette="tab20",
                alpha=0.7,
                ax=ax,
                s=60,
            )
            ax.set_xlabel("IMDb Rating")
            ax.set_ylabel("AI Score")
            ax.set_title("IMDb Rating vs AI Score by Genre")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            st.pyplot(fig)

        st.markdown('<p class="section-header">☁️ AI Reasoning Word Cloud</p>', unsafe_allow_html=True)
        all_reasons = " ".join(movies["Reason"].astype(str))
        if all_reasons.strip():
            stopwords = {"movie", "film", "plot", "story", "character", "the", "and", "a", "of", "to", "in", "is", "it", "with", "for", "on", "as", "at", "by", "an"}
            wordcloud = WordCloud(
                width=1000,
                height=500,
                background_color="white",
                colormap="plasma",
                stopwords=stopwords,
                max_words=100
            ).generate(all_reasons)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wordcloud, interpolation="bilinear")
            ax.axis("off")
            ax.set_title("Most Common Words in AI Reasoning")
            st.pyplot(fig)
        else:
            st.info("No reasoning text available for the word cloud.")

        st.markdown('<p class="section-header">📊 Score Distribution Analysis</p>', unsafe_allow_html=True)
        col_hist1, col_hist2 = st.columns(2)

        with col_hist1:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(movies["AI_Score"], bins=20, color="#667eea", alpha=0.7, edgecolor="black")
            ax.set_xlabel("AI Score")
            ax.set_ylabel("Frequency")
            ax.set_title("AI Score Distribution")
            ax.axvline(movies["AI_Score"].mean(), color='red', linestyle='--', label=f'Mean: {movies["AI_Score"].mean():.2f}')
            ax.legend()
            st.pyplot(fig)

        with col_hist2:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(movies["IMDb_Rating"], bins=20, color="#f5576c", alpha=0.7, edgecolor="black")
            ax.set_xlabel("IMDb Rating")
            ax.set_ylabel("Frequency")
            ax.set_title("IMDb Rating Distribution")
            ax.axvline(movies["IMDb_Rating"].mean(), color='red', linestyle='--', label=f'Mean: {movies["IMDb_Rating"].mean():.2f}')
            ax.legend()
            st.pyplot(fig)

    # -------------------- Enhanced Dataset Tab --------------------
    with dataset_tab:
        st.markdown('<p class="section-header">🎬 Enhanced Movie Dataset</p>', unsafe_allow_html=True)
        
        # Enhanced display dataframe
        display_columns = ["Series_Title", "Primary_Genre", "IMDb_Rating", "AI_Score", "AI_Delta"]
        if "Genre_Count" in movies.columns:
            display_columns.append("Genre_Count")
        if "Description_Sentiment" in movies.columns:
            display_columns.append("Description_Sentiment")
            
        display_df = movies[display_columns].copy()
        display_df = display_df.rename(columns={
            "Series_Title": "Title",
            "Primary_Genre": "Genre",
            "IMDb_Rating": "IMDb",
            "AI_Score": "AI Score",
            "AI_Delta": "Δ Score",
            "Genre_Count": "Genres Count",
            "Description_Sentiment": "Sentiment"
        })
        
        st.dataframe(display_df, use_container_width=True, height=400)

        # Enhanced download options
        col1, col2 = st.columns(2)
        with col1:
            csv_bytes = movies.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Full Dataset",
                data=csv_bytes,
                file_name="enhanced_movie_analysis.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            if not genre_analysis.empty:
                genre_csv = genre_analysis.to_csv(index=True).encode("utf-8")
                st.download_button(
                    "⬇️ Download Genre Analysis",
                    data=genre_csv,
                    file_name="genre_analysis.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        st.markdown('<p class="section-header">🧩 Detailed AI Analysis per Movie</p>', unsafe_allow_html=True)

        search_term = st.text_input("🔍 Search movies by title:", "")
        filtered_movies = movies[
            movies["Series_Title"].str.contains(search_term, case=False, na=False)
        ] if search_term else movies

        st.info(f"Showing {len(filtered_movies)} of {len(movies)} movies")

        for _, row in filtered_movies.iterrows():
            with st.expander(f"🎬 {row['Series_Title']} - {row['AI_Score']}/10"):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.metric("IMDb Rating", f"{row['IMDb_Rating']}/10")
                    st.metric("AI Score", f"{row['AI_Score']}/10")
                    delta = row.get("AI_Delta", 0)
                    st.metric("Score Difference", f"{delta:+.2f}")
                    st.markdown(f"**Genre:** {row['Genre']}")
                    st.markdown(f"**Primary Genre:** {row.get('Primary_Genre', 'N/A')}")
                    
                    # Enhanced metrics
                    if "Genre_Count" in row:
                        st.markdown(f"**Genre Count:** {row['Genre_Count']}")
                    if "Description_Sentiment" in row:
                        st.markdown(f"**Description Sentiment:** {row['Description_Sentiment']:+.2f}")
                    if "Target_Audience" in row:
                        st.markdown(f"**Target Audience:** {row['Target_Audience']}")
                    
                with col_b:
                    st.markdown("**🧠 AI Reasoning:**")
                    st.markdown(
                        f"""
                        <div style="
                            background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%);
                            color: white;
                            padding: 1rem;
                            border-radius: 12px;
                            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
                            font-weight: 500;
                            line-height: 1.6;
                            margin-bottom: 1rem;
                            ">
                            {row['Reason']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    
                    # Enhanced analysis sections
                    col_c, col_d = st.columns(2)
                    with col_c:
                        if "Strengths" in row and pd.notna(row["Strengths"]) and row["Strengths"].strip():
                            st.markdown("**✅ Strengths:**")
                            st.info(row["Strengths"])
                    with col_d:
                        if "Weaknesses" in row and pd.notna(row["Weaknesses"]) and row["Weaknesses"].strip():
                            st.markdown("**⚠️ Weaknesses:**")
                            st.warning(row["Weaknesses"])

# ===================== ENHANCED WELCOME SCREEN =====================
else:
    st.markdown("""
    <div style="text-align:center; padding:3rem; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
     border-radius:15px; color:white; margin-top: 2rem;">
        <h2>🚀 Welcome to Enhanced Movie Agent</h2>
        <p style="font-size: 1.1rem; margin-top: 1rem;">
            Next-generation intelligent movie analysis system with advanced AI reasoning,<br>
            data validation, and personalized recommendations.
        </p>
        <p style="margin-top: 1.5rem; font-size: 1rem;">
            📌 Configure your analysis settings in the sidebar<br>
            📌 Set your genre preferences for personalized recommendations<br>
            📌 Click "Run Enhanced Agent Analysis" to begin<br>
            📌 Explore advanced analytics and deep insights
        </p>
    </div>
    """ , unsafe_allow_html=True)

    st.markdown("---")
    
    # Enhanced feature highlights
    st.markdown('<p class="section-header">🆕 Enhanced Features</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 🤖 Advanced AI Analysis
        - Enhanced LLM prompts for detailed reasoning
        - Multi-dimensional evaluation (strengths, weaknesses, audience)
        - Sentiment analysis of movie descriptions
        - Robust fallback mechanisms
        """)
    with col2:
        st.markdown("""
        ### 📊 Data Quality & Validation
        - Comprehensive data validation checks
        - Automated data enrichment
        - Genre trend analysis
        - Correlation matrix visualization
        """)
    with col3:
        st.markdown("""
        ### 🎯 Personalized Recommendations
        - Genre preference-based filtering
        - Multi-factor scoring algorithm
        - Diversity-aware suggestions
        - Minimum rating thresholds
        """)

    st.markdown("---")
    
    # Quick start guide
    with st.expander("🚀 Quick Start Guide", expanded=True):
        st.markdown("""
        1. **Configure Analysis** in the sidebar:
           - Choose analysis depth (Quick/Normal/Deep)
           - Select number of movies to analyze
           - Set your preferred genres for personalized recommendations
        
        2. **Run Analysis**:
           - Click the "Run Enhanced Agent Analysis" button
           - Wait for the AI to process and analyze the movies
        
        3. **Explore Results**:
           - View overview metrics and workflow
           - Check advanced analytics and correlations
           - Examine detailed AI reasoning per movie
           - Download enhanced datasets
        
        4. **Get Personalized Recommendations**:
           - See top recommendations based on your preferences
           - Understand the rationale behind each suggestion
        """)

st.markdown("---")
st.caption("Built for Educosys Hackathon • Enhanced Agentic AI Application • © 2025 Akhil Kumar B")
