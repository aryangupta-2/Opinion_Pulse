# Opinion_Pulse
## Amazon Review Mining for Sentiment Classification & Trend Analysis

##  Project Description

Customer reviews contain valuable feedback, but are unstructured and noisy.
Manually analyzing thousands of reviews is infeasible.

This project implements an **end-to-end AI-powered review intelligence pipeline** that transforms raw Amazon product reviews into **business-ready sentiment and trend insights** using web scraping, NLP, and deep learning.

The system automatically:

- Scrapes customer reviews from Amazon product pages  
- Cleans and preprocesses noisy review text  
- Classifies each review as **Positive** or **Negative** using a fine-tuned **DistilBERT** model  
- Performs **aspect-based trend analysis** to identify recurring product themes  
- Generates a **structured insight report** highlighting sentiment patterns and product trends  


---

##  Problem Statement

Customer reviews contain valuable feedback but are:

- Unstructured  
- Noisy  
- Extremely large in volume  

Manual analysis is impractical.

### Goal

Design a system that converts unstructured Amazon reviews into:

- Overall sentiment trends  
- Aspect-level product trend analysis  
- Temporal sentiment movement  
- Business-ready summarized insights  

---

## Project Objectives

✔ Scrape Amazon product reviews  
✔ Build a robust text preprocessing pipeline  
✔ Train and deploy a sentiment classification model  
✔ Extract aspect-level trends from reviews  
✔ Generate structured, interpretable insights  

---

## Implemented Features

###  1. Review Scraping Module

- Built using **Selenium**
- Handles:
  - Login/session persistence
  - Dynamic content loading
  - Pagination
- Extracts:
  - Review title
  - Review text
  - Review date
- Stores data in `amazon_reviews.csv`

---

###  2. Text Preprocessing Pipeline

Applied transformations include:

- HTML removal (BeautifulSoup)
- Lowercasing
- URL removal
- Emoji removal
- Punctuation removal
- Stop-word removal (NLTK)
- Lemmatization (WordNet)

This ensures clean, normalized input for downstream NLP models.

---

###  3. Sentiment Classification Engine

- Fine-tuned **DistilBERT** on an Amazon Reviews dataset  
- Binary sentiment classification:
  - `1` → Positive
  - `0` → Negative
- Model is:
  - Trained once
  - Saved locally in `sentiment-bert/`
  - Reused for inference without retraining  

---

###  4. Aspect-Based Trend Analysis Engine

- Uses **spaCy** for noun-phrase extraction
- Identifies meaningful multi-word product aspects such as:
  - battery life
  - build quality
  - charging speed
- Applies **RapidFuzz** to cluster semantically similar aspects
- Associates sentiment polarity with each aspect

This enables detection of **recurring positive and negative product trends** instead of isolated opinions.

---

###  5. Product Insight Generator

Generates high-level business intelligence including:

- Overall sentiment score
- Positive vs Negative sentiment ratio
- Sentiment trend over time
- Top positively trending product aspects
- Top negatively trending product aspects
- Sentiment polarization index
- Review growth momentum

---

##  System Architecture
<img width="286" height="512" alt="image" src="https://github.com/user-attachments/assets/dd488732-72f8-4ab6-9083-0b6b87416b4c" />

---
## Demo 

https://drive.google.com/drive/u/0/folders/1UesQQiGNEZcZFN93e-wFEZBELeLRWX3Y

## Installation Guide

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/aryangupta-2/Opinion_Pulse

python train.py

streamlit run ui.py


