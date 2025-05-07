import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\lekha\Downloads\Titanic-Dataset.csv")
df.columns = df.columns.str.strip()
print(df.info())
print(df.describe())
print("Missing values before cleaning:\n", df.isnull().sum())

df['Age'] = df['Age'].fillna(df['Age'].median())
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

num_cols = ['Age', 'Fare', 'SibSp', 'Parch']
print("Missing values in numeric columns before scaling:\n", df[num_cols].isnull().sum())

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

for col in num_cols:
    plt.figure(figsize=(6, 3))
    sns.boxplot(x=df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()
print("Preview of processed data:\n", df.head())
print("Missing values after processing:\n", df.isnull().sum())
