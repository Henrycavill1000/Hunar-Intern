import pandas as pd
import numpy as np

df = pd.read_csv("food_coded.csv")
df_cleaned = df.copy()
df_cleaned['GPA'] = pd.to_numeric(df_cleaned['GPA'], errors='coerce')
df_cleaned['weight'] = df_cleaned['weight'].astype(str).str.extract(r'(\d+\.?\d*)')[0]
df_cleaned['weight'] = pd.to_numeric(df_cleaned['weight'], errors='coerce')

if 'comfort_food_reasons_coded.1' in df_cleaned.columns and df_cleaned['comfort_food_reasons_coded'].equals(df_cleaned['comfort_food_reasons_coded.1']):
    df_cleaned.drop(columns=['comfort_food_reasons_coded.1'], inplace=True)

for col in df_cleaned.columns:
    if df_cleaned[col].isnull().sum() > 0:
        if df_cleaned[col].dtype in [np.float64, np.int64]:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        else:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)

df_cleaned.drop_duplicates(inplace=True)

print(df_cleaned.info())
print("Missing values left:", df_cleaned.isnull().sum().sum())
print("Data cleaning completed .")