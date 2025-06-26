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

I learned a lot from this task, especially about real-world data issues like messy formats, mixed types, and null values. It also gave me hands-on experience using Pandas, which I now feel more confident with.
This preprocessing step is super important in any ML pipeline, and I'm glad I got to do it as part of my internship.
I done this work by honest, thankyou.


Task 2: House price prediction 

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
df = pd.read_csv("house price data.csv")

# Step 3: Initial analysis
print("Initial shape of dataset:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Step 4: Drop irrelevant columns
df.drop(['date', 'street', 'city', 'statezip', 'country'], axis=1, inplace=True)

# Step 5: Handle missing values (if any)
df.dropna(inplace=True)

# Step 6: Remove duplicates
df.drop_duplicates(inplace=True)

# Step 7: Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Step 8: Define features and target
X = df.drop('price', axis=1)
y = df['price']

# Step 9: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 11: Make predictions
y_pred = model.predict(X_test)

# Step 12: Evaluate the model
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Step 13: Plot actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='green')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', lw=2)
plt.show()

# Step 14: Predict on new sample (optional)
sample = X_test.iloc[0].values.reshape(1, -1)
predicted_price = model.predict(sample)
print("Predicted price for test sample:", predicted_price[0])
