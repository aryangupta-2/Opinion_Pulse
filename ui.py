import streamlit as st
import pandas as pd

from scraper import scraping
from dataframe import preprocessing_training
from Insights import insight

st.set_page_config(
    page_title="OpinionPulse",
    layout="wide"
)

st.sidebar.title(" OpinionPulse")
st.sidebar.caption("Amazon Review Intelligence")

product_url = st.sidebar.text_input(
    "Enter Amazon Product URL",
    placeholder="https://www.amazon.in/..."
)

run_btn = st.sidebar.button(" Run Full Analysis")

st.title("OpinionPulse")
st.subheader("Amazon Review Sentiment & Insight Dashboard")
st.caption("BERT-based Sentiment • Pros & Cons • Trends")

st.divider()

if run_btn:

    if product_url.strip() == "":
        st.error("Please enter a product URL")
        st.stop()

    with st.spinner("Scraping reviews from Amazon..."):
        scraping(product_url)

    with st.spinner("Cleaning reviews & running sentiment model..."):
        df = preprocessing_training()

    with st.spinner("Generating insights..."):
        (
            summary,
            trend,
            pros,
            cons,
            polarization,
    
        ) = insight(df)

    st.success("Analysis complete!")

    st.subheader("Overall Sentiment Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Sentiment Score", summary["overall_sentiment_score"])
    c2.metric("Positive Reviews", f"{summary['positive_ratio']*100:.1f}%")
    c3.metric("Negative Reviews", f"{summary['negative_ratio']*100:.1f}%")
    c4.metric("Total Reviews", summary["total_reviews"])

    st.divider()

    

    st.subheader("Advanced Signals")

    a1, a2 = st.columns(2)

    a1.metric(
        "Polarization Index",
        polarization,
        help="Higher = more divided opinions"
    )

    
    

    st.divider()

    st.subheader("Sentiment Trend Over Time")

    trend_df = trend.set_index("review_date")
    st.line_chart(trend_df[["avg_sentiment", "positive_ratio"]])

    st.subheader("Pros &  Cons")

    p_col, c_col = st.columns(2)

    with p_col:
        st.markdown("### What users like")
        for p, count in pros:
            st.success(f"{p} ({count})")

    with c_col:
        st.markdown("### What users dislike")
        for c, count in cons:
            st.error(f"{c} ({count})")

    st.divider()
   
    

else:
    st.info("Enter a product URL and click **Run Full Analysis**")
