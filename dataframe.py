import pandas as pd
def csv_to_dataframe(file_path):
    return pd.read_csv(file_path)

    
df = csv_to_dataframe("amazon_reviews.csv")
print(df)



_df = None

def get_df():
    global _df
    if _df is None:
        _df = pd.read_csv("amazon_reviews.csv")
    return _df