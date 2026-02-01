def insight(df):
    import pandas as pd
    import numpy as np
    from collections import Counter
    import re
    import re
    import spacy
    from collections import Counter
    from rapidfuzz import fuzz



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

        for doc in nlp.pipe(texts):
            for chunk in doc.noun_chunks:
                phrase = chunk.text.lower().strip()

                phrase = re.sub(r"[^a-z\s]", "", phrase)
                phrase = re.sub(r"\s+", " ", phrase)

                if (
                    len(phrase.split()) >= 2 and
                    phrase not in USELESS_WORDS
                ):
                    phrases.append(phrase)

        return phrases
    

    def merge_similar_phrases(phrases, similarity_threshold=85):
        merged = {}

        for phrase in phrases:
            found = False
            for key in merged:
                if fuzz.ratio(phrase, key) >= similarity_threshold:
                    merged[key] += 1
                    found = True
                    break
            if not found:
                merged[phrase] = 1

        return merged



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

        phrases = extract_noun_phrases(texts)

        merged_phrases = merge_similar_phrases(phrases)

        return Counter(merged_phrases).most_common(n)


    def top_cons(df, n=5):
        texts = df[df["sentiment_norm"] == -1]["review_text"].dropna().tolist()

        phrases = extract_noun_phrases(texts)

        merged_phrases = merge_similar_phrases(phrases)

        return Counter(merged_phrases).most_common(n)


    def sentiment_polarization_index(df):
        df = df.copy()
        df["sentiment_norm"] = df["bert_label"].apply(normalize_sentiment)

        pos = (df["sentiment_norm"] == 1).mean()
        neg = (df["sentiment_norm"] == -1).mean()

        polarization = 1 - abs(pos - neg)

        return float(round(polarization, 3))


    def generate_review_insights(df):
    
        return (
            overall_sentiment_score(df),
            sentiment_trend_over_time(df, freq="ME"),
            top_pros(df),
            top_cons(df),
            sentiment_polarization_index(df),
        )

    df["sentiment_norm"] = df["bert_label"].apply(normalize_sentiment)
    insights = generate_review_insights(df)

    return insights
