import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules

# Load the dataset
data = pd.read_csv("C:/Users/malik/OneDrive/Desktop/python program/Online Retail Dataset.csv")
print(data.head())

# Data Preprocessing
print(data.isnull().sum())
print(data.dtypes)

# Drop rows with missing CustomerID (since it's important for analysis)
data.dropna(subset=['CustomerID'], inplace=True)

# Convert InvoiceDate to datetime
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'], dayfirst=True)

# Filter out cancelled orders (InvoiceNo starts with 'C')
data = data[~data['InvoiceNo'].str.startswith('C')]

print(data.head())

# Create a transaction dataset (CustomerID, StockCode)
basket = data.groupby(['CustomerID', 'Description'])['Quantity'].sum().unstack().reset_index().fillna(0).set_index('CustomerID')

# Convert quantity greater than 0 to 1 (indicating purchase)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

# Box Plot (Price Distribution by Country)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Country', y='UnitPrice', data=data)
plt.xticks(rotation=90)
plt.title('Price Distribution by Country')
plt.show()

# Histogram (Quantity Distribution)
plt.figure(figsize=(10, 6))
sns.histplot(data['Quantity'], kde=True)
plt.title('Quantity Distribution')
plt.show()

# Scatter Plot (Quantity vs UnitPrice)
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Quantity', y='UnitPrice', data=data)
plt.title('Quantity vs UnitPrice')
plt.show()

# Pearson Correlation: UnitPrice vs Quantity
corr = data[['Quantity', 'UnitPrice']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
