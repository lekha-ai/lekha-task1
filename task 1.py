import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset using the correct path
df = pd.read_csv(r"C:\Users\lekha\Downloads\Titanic-Dataset.csv")

# Strip any extra spaces from column names
df.columns = df.columns.str.strip()

# Display basic information and summary statistics
print(df.info())
print(df.describe())
print("Missing values before cleaning:\n", df.isnull().sum())

# Handle missing values
df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Encode categorical variables
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Identify numeric columns for scaling
num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
print("Missing values in numeric columns before scaling:\n", df[num_cols].isnull().sum())

# Standardize numeric features
scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# Visualize outliers using boxplots
for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()

# Preview processed data
print("Preview of processed data:\n", df.head())
print("Missing values after processing:\n", df.isnull().sum())
