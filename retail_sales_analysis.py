import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("retail_sales.csv")

print("\nFIRST 5 RECORDS")
print(df.head())

# ==========================
# Data Information
# ==========================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Info:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicates:")
print(df.duplicated().sum())

# ==========================
# Statistical Summary
# ==========================

print("\nSummary Statistics")
print(df.describe())

# ==========================
# Data Visualization
# ==========================

sns.set_style("whitegrid")

# Sales Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Sales'], bins=10)
plt.title("Sales Distribution")
plt.savefig("sales_distribution.png")
plt.show()

# Category Wise Sales
plt.figure(figsize=(8,5))
sns.barplot(x='Category', y='Sales', data=df)
plt.title("Average Sales by Category")
plt.savefig("category_sales.png")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(8,6))
numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# ==========================
# Machine Learning
# Sales Prediction
# ==========================

X = df[['Quantity', 'Price', 'Discount']]
y = df['Sales']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ==========================
# Evaluation
# ==========================

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print("MAE:", round(mae,2))
print("R² Score:", round(r2,2))

# ==========================
# Prediction Visualization
# ==========================

plt.figure(figsize=(8,5))
plt.scatter(y_test, predictions)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")

plt.savefig("sales_prediction.png")
plt.show()

# ==========================
# Business Insights
# ==========================

print("\nBUSINESS INSIGHTS")

print("\nTotal Revenue:")
print(df["Sales"].sum())

print("\nAverage Revenue:")
print(df["Sales"].mean())

print("\nCategory Sales:")
print(df.groupby("Category")["Sales"].mean())

print("\nTop Sale:")
print(df.loc[df["Sales"].idxmax()])

print("\nProject Completed Successfully")