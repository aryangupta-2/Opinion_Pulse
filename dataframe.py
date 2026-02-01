import pandas as pd
def csv_to_dataframe(file_path):
    return pd.read_csv(file_path)

df = csv_to_dataframe("amazon_reviews.csv")
import re
import string
import emoji
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_review(text):
    if not isinstance(text, str):
        return ""


    text = BeautifulSoup(text, "html.parser").get_text()

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = emoji.replace_emoji(text, replace="")

    text = text.translate(str.maketrans("", "", string.punctuation))

    tokens = text.split()

    tokens = [t for t in tokens if t not in stop_words]

    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)






_df = None

def get_df():
    global _df
    if _df is None:
        _df = pd.read_csv("amazon_reviews.csv")
    return _df

from transformers import pipeline
import os

MODEL_PATH = "sentiment-bert"

def load_trained_sentiment_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Trained model not found. Run train_model.py once."
        )

    return pipeline(
        "sentiment-analysis",
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        device=-1  
    )



def train_and_save_bert_sentiment_model():
    import os
    import kagglehub
    import pandas as pd
    import torch
    import numpy as np
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        pipeline
    )

    path = kagglehub.dataset_download("kritanjalijain/amazon-reviews")
    print("Path to dataset files:", path)

    train_path = os.path.join(path, "train.csv")
    test_path  = os.path.join(path, "test.csv")

    train_df = pd.read_csv(train_path, header=None)
    test_df  = pd.read_csv(test_path, header=None)

    train_df.columns = ["polarity", "title", "text"]
    test_df.columns  = ["polarity", "title", "text"]

    for df in [train_df, test_df]:
        df["label"] = df["polarity"].map({1: 0, 2: 1})
        df["review"] = df["title"].fillna("") + " " + df["text"].fillna("")
   

    bert_train = train_df.sample(20, random_state=42)
    bert_val   = test_df.sample(20, random_state=42)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize(texts):
        return tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=256
        )

    class ReviewDataset(torch.utils.data.Dataset):
        def __init__(self, texts, labels):
            self.encodings = tokenize(texts)
            self.labels = labels.tolist()

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
            item["labels"] = torch.tensor(self.labels[idx])
            return item

    train_ds = ReviewDataset(bert_train["review"], bert_train["label"])
    val_ds   = ReviewDataset(bert_val["review"], bert_val["label"])


    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=2
    )

    training_args = TrainingArguments(
        output_dir="./bert",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        weight_decay=0.01,
        logging_steps=100,
        save_total_limit=1
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    trainer.train()


    trainer.save_model("sentiment-bert")
    tokenizer.save_pretrained("sentiment-bert")

    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="sentiment-bert",
        tokenizer="sentiment-bert",
        device=0 if torch.cuda.is_available() else -1
    )

    return sentiment_pipe


def predict_bert_sentiment(
    df,
    sentiment_pipe,
    batch_size=32,
    max_length=256
):
    """
    Appends a bert_label column to df:
    1 -> positive
    0 -> negative
    """

    df["text"] = (
        "TITLE: " + df["review_title"].fillna("") +
        " [SEP] REVIEW: " + df["review_text"].fillna("")
    )


    outputs = sentiment_pipe(
        df["text"].tolist(),
        batch_size=batch_size,
        truncation=True,
        max_length=max_length
    )


    df["bert_label"] = [
        1 if o["label"] == "LABEL_1" else 0
        for o in outputs
    ]

    return df


pipe = load_trained_sentiment_model()

df = predict_bert_sentiment(get_df(), pipe)

