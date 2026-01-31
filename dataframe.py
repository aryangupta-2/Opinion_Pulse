import pandas as pd
def csv_to_dataframe(file_path):
    return pd.read_csv(file_path)

    
df = csv_to_dataframe("amazon_reviews.csv")
print(df)
