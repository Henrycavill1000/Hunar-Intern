# Hunar-Intern
Project :
As part of my machine learning internship,given a raw dataset named food_coded.csv which contains survey responses related to food habits, dietary preferences, calories, and lifestyle factors of students.
My main task was to analyze and clean this dataset before applying any machine learning models. This involved handling missing values, fixing incorrect data formats, and removing duplicate entries.

1. Loaded the Dataset
I used Pandas to load the CSV file into a DataFrame:
import pandas as pd
import numpy as np
df = pd.read_csv("food_coded.csv")

3. Copied the Original Data
To be safe, I worked on a copy of the original dataset:

df_cleaned = df.copy()

3. Fixed Column Data Types
Some columns like GPA and weight were in the wrong format (e.g., weight had values like “Not sure, 240” or “I’m not answering”). So, I converted them to numbers:

df_cleaned['GPA'] = pd.to_numeric(df_cleaned['GPA'], errors='coerce')

# For weight, I extracted numbers from strings
df_cleaned['weight'] = df_cleaned['weight'].astype(str).str.extract(r'(\d+\.?\d*)')[0]
df_cleaned['weight'] = pd.to_numeric(df_cleaned['weight'], errors='coerce')

4. Removed Duplicate Columns
There was a column comfort_food_reasons_coded.1 which was exactly the same as another column, so I removed it:

df_cleaned.drop(columns=['comfort_food_reasons_coded.1'], inplace=True)

5. Filled Missing Values
For missing values:

I used median for numerical columns to avoid impact of outliers.
For categorical columns, I used the most frequent value (mode).

for col in df_cleaned.columns:
    if df_cleaned[col].isnull().sum() > 0:
        if df_cleaned[col].dtype in [np.float64, np.int64]:
            df_cleaned[col].fillna(df_cleaned[col].median(), inplace=True)
        else:
            df_cleaned[col].fillna(df_cleaned[col].mode()[0], inplace=True)
            
6. Removed Duplicate Rows
Finally, I removed any repeated rows to make sure the data is clean:

df_cleaned.drop_duplicates(inplace=True)

7. Checked the Final Dataset
To confirm everything worked, I printed the summary:

print(df_cleaned.info())
print("Missing values left:", df_cleaned.isnull().sum().sum())

Final Output:
--All columns were cleaned properly.
--No missing values remained.
--Dataset was ready for ML modeling or analysis.

I done this work by honest, thankyou.
