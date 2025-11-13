"""
Analysis Module
Handles AI-powered movie analysis, scoring, and insights generation
"""

import os
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Optional
from tqdm import tqdm
from dotenv import load_dotenv

try:
    from langchain_openai import ChatOpenAI
except ImportError:
    ChatOpenAI = None

try:
    import seaborn as sns
except ImportError:
    sns = None

from .data_processing import STATE

# Load config
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def _get_llm(model: str = OPENAI_MODEL, temperature: float = 0.3):
    """Initialize LLM if available"""
    if ChatOpenAI is None:
        return None
    try:
        if not (os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY_PATH")):
            return None
        return ChatOpenAI(model=model, temperature=temperature)
    except Exception:
        return None


def _to_float_safe(value, default: float = 0.0) -> float:
    """Safely convert value to float"""
    try:
        if isinstance(value, (list, tuple)):
            candidate = value[0] if value else default
        elif isinstance(value, (pd.Series, pd.Index)):
            candidate = value.iloc[0] if len(value) > 0 else default
        else:
            candidate = value
        num = pd.to_numeric(candidate, errors="coerce")
        return float(num) if not pd.isna(num) else default
    except Exception:
        return default


def _compute_genre_median_baseline(movies: pd.DataFrame) -> Tuple[dict, float]:
    """Compute genre-based median ratings"""
    temp = movies.copy()
    temp = temp.assign(Genre=temp["Genre"].str.split(",")).explode("Genre")
    temp["Genre"] = temp["Genre"].str.strip()
    temp["IMDb_Rating"] = pd.to_numeric(temp["IMDb_Rating"], errors="coerce")
    genre_median = temp.groupby("Genre")["IMDb_Rating"].median().to_dict()
    global_median = float(temp["IMDb_Rating"].median()) if not temp["IMDb_Rating"].dropna().empty else 6.5
    return genre_median, global_median


def _build_enhanced_prompt(title: str, genre: str, desc: str, rating: float, mode: str) -> str:
    """More structured prompts for consistent LLM responses"""
    rating_text = f"{rating}/10" if rating > 0 else "unavailable"
    
    analysis_depth = {
        "quick": "Focus on basic plot quality and entertainment value.",
        "normal": "Consider acting, direction, pacing, and audience appeal.",
        "deep": "Analyze cinematography, thematic depth, cultural impact, technical execution, and rewatch value."
    }
    
    return f"""
    As an expert film critic, analyze this movie and provide a comprehensive assessment.
    
    MOVIE: {title}
    PRIMARY GENRE: {genre}
    IMDb RATING: {rating_text}
    DESCRIPTION: {desc}
    
    ANALYSIS FOCUS: {analysis_depth.get(mode, analysis_depth['normal'])}
    
    EVALUATION CRITERIA:
    - Story originality and narrative execution
    - Technical aspects (cinematography, editing, sound)
    - Performance and character development  
    - Genre conventions and innovation
    - Cultural impact and rewatch value
    - Pacing and audience engagement
    
    RESPONSE FORMAT (JSON only):
    {{
        "AI_Score": <float between 0-10, be critical and use the full scale>,
        "Reason": "<concise reasoning, 150-300 characters>",
        "Strengths": ["<key strength 1>", "<key strength 2>"],
        "Weaknesses": ["<key weakness 1>", "<key weakness 2>"],
        "Audience": "<main target audience>"
    }}
    
    Important: Be honest and critical. A 10/10 should be exceptionally rare.
    """


def _analyze_with_enhanced_llm(movies: pd.DataFrame, llm, limit: Optional[int] = None, mode: str = "normal") -> Tuple[pd.DataFrame, List[str]]:
    """Enhanced LLM analysis with better response parsing"""
    logs: List[str] = []
    df = movies.copy()
    
    # Initialize enhanced columns
    df["AI_Score"], df["Reason"] = 0.0, ""
    df["Strengths"], df["Weaknesses"] = "", ""
    df["Target_Audience"] = ""
    
    indices = list(df.index)[:limit] if limit else list(df.index)
    
    prompts = [
        _build_enhanced_prompt(
            df.loc[i]["Series_Title"], 
            df.loc[i].get("Primary_Genre", df.loc[i]["Genre"].split(",")[0].strip()), 
            str(df.loc[i].get("Overview", ""))[:400],
            _to_float_safe(df.loc[i]["IMDb_Rating"]), 
            mode
        ) 
        for i in indices
    ]

    responses: List[str] = []
    try:
        for p in tqdm(prompts, desc="AI analysis"):
            r = llm.invoke(p) if hasattr(llm, "invoke") else str(p)
            content = getattr(r, "content", None)
            responses.append(content if isinstance(content, str) else str(r))
    except Exception as e:
        logs.append(f"⚠️ LLM failed: {e}. Using heuristics.")
        return _fallback_analysis(df), logs

    # Process enhanced responses
    for i, idx in enumerate(indices):
        text = (responses[i] or "").strip()
        if text.startswith("```"):
            text = text.split("```")[-1].strip()
        
        # Default values
        score = _to_float_safe(df.loc[idx, "IMDb_Rating"])
        reason = ""
        strengths = []
        weaknesses = []
        audience = "General"
        
        try:
            # Try to parse enhanced JSON response
            data = json.loads(text)
            score = _to_float_safe(data.get("AI_Score", score))
            reason = str(data.get("Reason", "")).strip()
            strengths = data.get("Strengths", [])
            weaknesses = data.get("Weaknesses", [])
            audience = data.get("Audience", "General")
            
        except Exception:
            # Fallback to basic parsing
            match = re.search(r"\{[\s\S]*\}", text)
            if match:
                try:
                    data = json.loads(match.group(0))
                    score = _to_float_safe(data.get("AI_Score", score))
                    reason = str(data.get("Reason", "")).strip()
                except:
                    reason = text[:200]
            else:
                reason = text[:200]

        # Scale conversion check
        combined_text = f"{reason.lower()} {text.lower()}"
        if (
            score <= 5.0
            and "out of 10" not in combined_text
            and "/10" not in combined_text
            and any(token in combined_text for token in ["/5", "out of 5", "5-point", "five-point"])
        ):
            score = score * 2.0
            reason = (reason + " (Converted from 5-point scale)")[:300]

        # Apply bounds and store
        score = max(0.1, min(10.0, score))
        df.loc[idx, "AI_Score"] = round(score, 2)
        df.loc[idx, "Reason"] = reason[:300]
        df.loc[idx, "Strengths"] = ", ".join(strengths)[:200] if isinstance(strengths, list) else str(strengths)[:200]
        df.loc[idx, "Weaknesses"] = ", ".join(weaknesses)[:200] if isinstance(weaknesses, list) else str(weaknesses)[:200]
        df.loc[idx, "Target_Audience"] = audience[:100]

    # Calculate additional metrics
    df["AI_Delta"] = df["AI_Score"] - pd.to_numeric(df["IMDb_Rating"], errors="coerce")
    df["Reason_Length"] = df["Reason"].astype(str).str.len()
    df["Analysis_Complexity"] = df["Reason_Length"] / df["Reason_Length"].max() if df["Reason_Length"].max() > 0 else 0
    
    return df, logs


def _fallback_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced fallback analysis when LLM fails"""
    df["AI_Score"] = pd.to_numeric(df["IMDb_Rating"], errors="coerce").fillna(0)
    
    # More sophisticated fallback reasoning
    conditions = [
        (df["AI_Score"] >= 8.5),
        (df["AI_Score"] >= 7.0),
        (df["AI_Score"] >= 5.5),
        (df["AI_Score"] < 5.5)
    ]
    choices = [
        "Exceptional film with critical acclaim and audience praise",
        "Solid entertainment with good production values and engaging story", 
        "Average film with some enjoyable elements but notable flaws",
        "Below average with significant issues in execution"
    ]
    
    df["Reason"] = np.select(conditions, choices, default="Heuristic scoring based on IMDb rating")
    df["Strengths"] = "Heuristic analysis"
    df["Weaknesses"] = "Limited analysis available"
    df["Target_Audience"] = "General"
    df["AI_Delta"] = 0
    df["Reason_Length"] = df["Reason"].str.len()
    df["Analysis_Complexity"] = 0.3
    
    return df


def analyze_movies(mode: str = "deep", llm_model: str = OPENAI_MODEL) -> pd.DataFrame:
    """
    Enhanced movie analysis with multiple modes and fallbacks
    """
    print("\n🧠 Analyzing movies with enhanced AI...")
    movies = STATE.get("categorized")
    if movies is None:
        raise ValueError("Run categorization first.")
    
    mode = mode.lower().strip()
    df = movies.copy()

    # ---------------- Quick Mode ----------------
    if mode == "quick":
        ratings = pd.to_numeric(df["IMDb_Rating"], errors="coerce")
        genre_median, global_median = _compute_genre_median_baseline(df)
        missing_mask = ratings.isna() | (ratings <= 0)
        if missing_mask.any():
            genres_primary = df.get("Primary_Genre", df["Genre"].str.split(",").str[0].str.strip())
            mapped = genres_primary.map(lambda g: genre_median.get(g, global_median))
            ratings[missing_mask] = mapped[missing_mask].fillna(global_median)
        df["AI_Score"] = ratings.clip(0.1, 10).round(2)
        df["Reason"] = "Quick heuristic scoring based on IMDb and genre median."
        df["AI_Delta"] = df["AI_Score"] - pd.to_numeric(df["IMDb_Rating"], errors="coerce")
        df["Reason_Length"] = df["Reason"].astype(str).str.len()
        STATE["analyzed"] = df
        return df

    # ---------------- LLM Mode (Normal / Deep) ----------------
    llm = _get_llm(llm_model, temperature=0.2 if mode == "deep" else 0.4)
    if llm is None:
        print("⚠️ LLM not available, using enhanced fallback analysis")
        df = _fallback_analysis(df)
        STATE["analyzed"] = df
        return df

    # ---------------- Enhanced LLM Analysis ----------------
    limit = min(20, len(df)) if mode == "normal" else None
    analyzed, logs = _analyze_with_enhanced_llm(df, llm, limit=limit, mode=mode)

    # Ensure data quality
    analyzed["AI_Score"] = analyzed["AI_Score"].apply(lambda x: max(0.1, min(10.0, x)))
    analyzed["Reason"] = analyzed["Reason"].fillna("No reasoning provided")

    for log in logs:
        print(log)

    # Final cleanup
    analyzed = analyzed.drop_duplicates(subset=["Series_Title"], keep="first")
    analyzed = analyzed.reset_index(drop=True)
    
    print(f"✅ Enhanced analysis complete: {len(analyzed)} unique movies")
    STATE["analyzed"] = analyzed
    return analyzed


def summarize_movies() -> str:
    """Generate summary of movie analysis"""
    print("\n📊 Summarizing AI insights...")
    movies = STATE.get("analyzed")
    if movies is None:
        raise ValueError("Run analysis first.")
    
    df = movies.copy()
    # Explode ONLY for genre summarization
    df = df.assign(Genre=df["Genre"].astype(str).str.split(",")).explode("Genre")
    df["Genre"] = df["Genre"].str.strip()
    genre_scores = df.groupby("Genre")["AI_Score"].mean().reset_index()
    
    if genre_scores.empty:
        return "No genre data."
    
    top = genre_scores.loc[genre_scores["AI_Score"].idxmax()]
    plt.figure(figsize=(10, 6))
    plt.barh(genre_scores["Genre"], genre_scores["AI_Score"])
    plt.title("Average AI Score by Genre")
    plt.xlabel("AI Score")
    plt.tight_layout()
    chart_path = os.path.join(os.getcwd(), "ai_score_by_genre.png")
    plt.savefig(chart_path)
    plt.close()
    
    return f"🎬 Top Genre: {top['Genre']} (Avg AI Score: {top['AI_Score']:.2f})\n{genre_scores.to_string(index=False)}"


def analyze_genre_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Deep analysis of genre performance trends"""
    # Explode genres for analysis
    exploded = df.assign(Genre=df["Genre"].astype(str).str.split(",")).explode("Genre")
    exploded["Genre"] = exploded["Genre"].str.strip()
    
    genre_analysis = exploded.groupby("Genre").agg({
        "AI_Score": ["count", "mean", "std", "min", "max"],
        "IMDb_Rating": ["mean", "std"],
        "AI_Delta": "mean"
    }).round(2)
    
    # Flatten column names
    genre_analysis.columns = ['_'.join(col).strip() for col in genre_analysis.columns.values]
    genre_analysis = genre_analysis.rename(columns={"AI_Score_count": "Movie_Count"})
    
    return genre_analysis.sort_values("AI_Score_mean", ascending=False)


def create_correlation_analysis(df: pd.DataFrame):
    """Show relationships between different metrics"""
    if sns is None:
        return None
        
    numeric_cols = ["AI_Score", "IMDb_Rating", "AI_Delta"]
    
    # Add sentiment if available
    if "Description_Sentiment" in df.columns:
        numeric_cols.append("Description_Sentiment")
    
    if "Genre_Count" in df.columns:
        numeric_cols.append("Genre_Count")
    
    # Filter to only numeric columns that exist
    existing_numeric = [col for col in numeric_cols if col in df.columns]
    
    if len(existing_numeric) < 2:
        return None
    
    correlation_matrix = df[existing_numeric].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0, fmt=".2f")
    plt.title("Metric Correlation Analysis")
    plt.tight_layout()
    
    return plt.gcf()
