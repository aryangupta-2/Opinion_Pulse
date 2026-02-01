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

import spacy
from collections import Counter

nlp = spacy.load("en_core_web_sm")

USELESS_WORDS = {
    "product", "item", "thing", "amazon", "buy", "purchase",
    "quality", "price", "value"
}

def extract_noun_phrases(texts):
    phrases = []

    for doc in nlp.pipe(texts, disable=["ner", "parser"]):
        for token in doc:
            if token.pos_ in {"NOUN", "PROPN"}:
                phrase = token.lemma_.lower()
                if phrase not in USELESS_WORDS and len(phrase) > 2:
                    phrases.append(phrase)

    return phrases



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
        "overall_sentiment_score":float( round(df["sentiment_norm"].mean(), 3)),
        "positive_ratio": float(round((df["sentiment_norm"] == 1).mean(), 3)),
        "negative_ratio": float(round((df["sentiment_norm"] == -1).mean(), 3)),
        "total_reviews": int(len(df))
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

    trend = trend[trend["review_count"] > 0]
    return trend


def top_pros(df, n=5):

    texts = df[df["sentiment_norm"] == 1]["review_text"].dropna().tolist()

    features = extract_noun_phrases(texts)

    return Counter(features).most_common(n)

def top_cons(df, n=5):

    texts = df[df["sentiment_norm"] == -1]["review_text"].dropna().tolist()

    features = extract_noun_phrases(texts)

    return Counter(features).most_common(n)


def sentiment_polarization_index(df):
    df = df.copy()
    df["sentiment_norm"] = df["bert_label"].apply(normalize_sentiment)

    pos = (df["sentiment_norm"] == 1).mean()
    neg = (df["sentiment_norm"] == -1).mean()

    polarization = 1 - abs(pos - neg)

    return float(round(polarization, 3))


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
  
    return (
         overall_sentiment_score(df),
         sentiment_trend_over_time(df, freq="ME"),
         top_pros(df),
         top_cons(df),
         sentiment_polarization_index(df),
         review_momentum(df, freq="ME")
    )

df["sentiment_norm"] = df["bert_label"].apply(normalize_sentiment)
insights = generate_review_insights(df)

print(insights)
