import pandas as pd
import numpy as np
from collections import Counter
import re
from dataframe import get_df, train_and_save_bert_sentiment_model,predict_bert_sentiment

df = get_df()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text


def normalize_sentiment(x):
    """
    Converts sentiment into +1 (positive) or -1 (negative)
    """
    if isinstance(x, str):
        return 1 if x.lower().strip() == "positive" else -1
    else:
        return 1 if x == 1 else -1

def overall_sentiment_score(df):
    df = df.copy()
    df["sentiment_norm"] = df["bert_label"].apply(normalize_sentiment)

    return {
        "overall_sentiment_score": round(df["sentiment_norm"].mean(), 3),
        "positive_ratio": round((df["sentiment_norm"] == 1).mean(), 3),
        "negative_ratio": round((df["sentiment_norm"] == -1).mean(), 3),
        "total_reviews": len(df)
    }


def sentiment_trend_over_time(df, freq="M"):
    """
    freq = 'D' (daily), 'W' (weekly), 'M' (monthly)
    """
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["sentiment_norm"] = df["bert_label"].apply(normalize_sentiment)

    trend = (
        df
        .groupby(pd.Grouper(key="review_date", freq=freq))
        .agg(
            avg_sentiment=("sentiment_norm", "mean"),
            positive_ratio=("sentiment_norm", lambda x: (x == 1).mean()),
            review_count=("sentiment_norm", "count")
        )
        .reset_index()
    )

    return trend

def top_pros(df, top_n=5):
    positive_reviews = df[df["bert_label"].apply(normalize_sentiment) == 1]

    text = " ".join(
        positive_reviews["review_title"].fillna("") + " " +
        positive_reviews["review_text"].fillna("")
    )

    words = clean_text(text).split()
    common = Counter(words)

    return common.most_common(top_n)

def top_cons(df, top_n=5):
    negative_reviews = df[df["bert_label"].apply(normalize_sentiment) == -1]

    text = " ".join(
        negative_reviews["review_title"].fillna("") + " " +
        negative_reviews["review_text"].fillna("")
    )

    words = clean_text(text).split()
    common = Counter(words)

    return common.most_common(top_n)

def sentiment_polarization_index(df):
    df = df.copy()
    df["sentiment_norm"] = df["bert_label"].apply(normalize_sentiment)

    pos = (df["sentiment_norm"] == 1).mean()
    neg = (df["sentiment_norm"] == -1).mean()

    polarization = 1 - abs(pos - neg)

    return round(polarization, 3)


def review_momentum(df, freq="M"):
    df = df.copy()
    df["review_date"] = pd.to_datetime(df["review_date"])

    counts = (
        df
        .groupby(pd.Grouper(key="review_date", freq=freq))
        .size()
        .reset_index(name="review_count")
    )

    counts["growth_rate"] = counts["review_count"].pct_change()

    return counts

def generate_review_insights(df):
    pipe = train_and_save_bert_sentiment_model()
    df = predict_bert_sentiment(df,pipe)
    return {
        "overall_sentiment": overall_sentiment_score(df),
        "sentiment_trend": sentiment_trend_over_time(df),
        "top_5_pros": top_pros(df),
        "top_5_cons": top_cons(df),
        "sentiment_polarization_index": sentiment_polarization_index(df),
        "review_momentum": review_momentum(df)
    }



insights = generate_review_insights(df)

print(insights.head())
