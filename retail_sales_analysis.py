import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# ==========================================
# CREATE OUTPUT FOLDERS
# ==========================================

os.makedirs("output", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)

# ==========================================
# LOAD DATASET
# ==========================================

df = pd.read_csv("data/retail_store_sales_dataset_pakistan.csv")

print("=" * 50)
print("ORIGINAL DATASET INFO")
print("=" * 50)
print(df.info())

print("\nShape:", df.shape)

# ==========================================
# DATA CLEANING
# ==========================================

print("\nCleaning Dataset...")

# Remove duplicates
duplicates = df.duplicated().sum()
print(f"Duplicate Records Found: {duplicates}")

df = df.drop_duplicates()

# Standardize Category Names
df["Product_Category"] = (
    df["Product_Category"]
    .astype(str)
    .str.strip()
    .str.title()
)

# Fill Missing Values

categorical_cols = [
    "Customer_City",
    "Payment_Method",
    "Sales_Channel",
    "Customer_Gender"
]

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

numeric_cols = [
    "Quantity",
    "Unit_Price",
    "Discount_Percent",
    "Customer_Age"
]

for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Convert Date

df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])

# Recalculate Total Amount

df["Total_Amount"] = (
    df["Quantity"]
    * df["Unit_Price"]
    * (1 - df["Discount_Percent"] / 100)
)

# Save Cleaned Dataset

df.to_csv(
    "data/retail_store_sales_cleaned.csv",
    index=False
)

print("Cleaning Completed")
print("New Shape:", df.shape)

# ==========================================
# FEATURE ENGINEERING
# ==========================================

df["Month"] = df["Purchase_Date"].dt.strftime("%Y-%m")

df["Age_Group"] = pd.cut(
    df["Customer_Age"],
    bins=[18, 25, 35, 45, 55, 65],
    labels=[
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65"
    ]
)

# ==========================================
# CUSTOMER SEGMENTATION
# ==========================================

customer_summary = (
    df.groupby("Customer_ID")
    .agg(
        Total_Spending=("Total_Amount", "sum"),
        Total_Orders=("Transaction_ID", "count"),
        Average_Order_Value=("Total_Amount", "mean")
    )
    .reset_index()
)

# Features for clustering
features = customer_summary[
    [
        "Total_Spending",
        "Total_Orders",
        "Average_Order_Value"
    ]
]

scaler = StandardScaler()

scaled_features = scaler.fit_transform(features)

kmeans = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

customer_summary["Cluster"] = kmeans.fit_predict(
    scaled_features
)

cluster_stats = (
    customer_summary
    .groupby("Cluster")["Total_Spending"]
    .mean()
    .sort_values()
)

cluster_order = cluster_stats.index.tolist()

cluster_labels = {
    cluster_order[0]: "Budget Customers",
    cluster_order[1]: "Regular Customers",
    cluster_order[2]: "High Value Customers"
}

customer_summary["Customer_Segment"] = (
    customer_summary["Cluster"]
    .map(cluster_labels)
)

# ==========================================
# ELBOW METHOD
# ==========================================

wcss = []

for k in range(1, 11):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    model.fit(scaled_features)

    wcss.append(model.inertia_)
    
plt.figure(figsize=(8,5))

plt.plot(
    range(1,11),
    wcss,
    marker="o"
)

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "charts/elbow_method.png"
)

plt.close()

# ==========================================
# SALES FORECASTING
# ==========================================

# Monthly Revenue
monthly_forecast = (
    df.groupby("Month")["Total_Amount"]
    .sum()
    .reset_index()
)

# Create numeric month index
monthly_forecast["Month_Index"] = range(
    1,
    len(monthly_forecast) + 1
)

X = monthly_forecast[["Month_Index"]]
y = monthly_forecast["Total_Amount"]

# Train Linear Regression
forecast_model = LinearRegression()
forecast_model.fit(X, y)

# Predict next 3 months
future_months = pd.DataFrame({
    "Month_Index": [
        len(monthly_forecast) + 1,
        len(monthly_forecast) + 2,
        len(monthly_forecast) + 3
    ]
})

future_months["Forecast_Revenue"] = (
    forecast_model.predict(future_months)
)

future_months["Month"] = [
    "2026-01",
    "2026-02",
    "2026-03"
]

# ==========================================
# KPI CALCULATIONS
# ==========================================

total_revenue = df["Total_Amount"].sum()
total_orders = len(df)
average_order_value = df["Total_Amount"].mean()

print("\n")
print("=" * 50)
print("KEY PERFORMANCE INDICATORS")
print("=" * 50)

print(f"Total Revenue: {total_revenue:,.2f}")
print(f"Total Orders: {total_orders}")
print(f"Average Order Value: {average_order_value:,.2f}")

# ==========================================
# MONTHLY SALES TREND
# ==========================================

monthly_sales = (
    df.groupby("Month")["Total_Amount"]
    .sum()
    .sort_index()
)

plt.figure(figsize=(10,5))
monthly_sales.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("charts/monthly_sales_trend.png")
plt.close()

# ==========================================
# TOP PRODUCTS
# ==========================================

top_products = (
    df.groupby("Product_Name")["Total_Amount"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,6))
top_products.plot(kind="bar")
plt.title("Top 10 Products by Revenue")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("charts/top_products.png")
plt.close()

# ==========================================
# CATEGORY PERFORMANCE
# ==========================================

category_sales = (
    df.groupby("Product_Category")["Total_Amount"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(8,5))
category_sales.plot(kind="bar")
plt.title("Revenue by Category")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("charts/category_performance.png")
plt.close()

# ==========================================
# REVENUE BY CITY
# ==========================================

city_sales = (
    df.groupby("Customer_City")["Total_Amount"]
    .sum()
    .sort_values(ascending=False)
)

plt.figure(figsize=(10,5))
city_sales.plot(kind="bar")
plt.title("Revenue by City")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("charts/revenue_by_city.png")
plt.close()

# ==========================================
# PAYMENT METHODS
# ==========================================

payment_dist = (
    df["Payment_Method"]
    .value_counts()
)

plt.figure(figsize=(6,6))
payment_dist.plot(kind="pie", autopct="%1.1f%%")
plt.ylabel("")
plt.title("Payment Method Distribution")
plt.tight_layout()
plt.savefig("charts/payment_method_distribution.png")
plt.close()

# ==========================================
# SALES CHANNEL
# ==========================================

sales_channel = (
    df["Sales_Channel"]
    .value_counts()
)

plt.figure(figsize=(6,5))
sales_channel.plot(kind="bar")
plt.title("Sales Channel Distribution")
plt.tight_layout()
plt.savefig("charts/sales_channel_distribution.png")
plt.close()

# ==========================================
# CUSTOMER AGE GROUPS
# ==========================================

age_groups = (
    df["Age_Group"]
    .value_counts()
    .sort_index()
)

plt.figure(figsize=(8,5))
age_groups.plot(kind="bar")
plt.title("Customer Age Groups")
plt.tight_layout()
plt.savefig("charts/customer_age_groups.png")
plt.close()

# ==========================================
# BUSINESS INSIGHTS
# ==========================================

best_product = top_products.index[0]
best_category = category_sales.index[0]
best_city = city_sales.index[0]
popular_payment = payment_dist.index[0]

print("\n")
print("=" * 50)
print("BUSINESS INSIGHTS")
print("=" * 50)

print(f"Best Performing Product: {best_product}")
print(f"Top Revenue Category: {best_category}")
print(f"Highest Revenue City: {best_city}")
print(f"Most Used Payment Method: {popular_payment}")

print("\nAnalysis Completed Successfully")

# ==========================================
# SAVE SUMMARY REPORT
# ==========================================

summary = pd.DataFrame({
    "Metric": [
        "Total Revenue",
        "Total Orders",
        "Average Order Value",
        "Best Product",
        "Top Category",
        "Top City",
        "Popular Payment Method"
    ],
    "Value": [
        total_revenue,
        total_orders,
        average_order_value,
        best_product,
        best_category,
        best_city,
        popular_payment
    ]
})

summary.to_csv(
    "output/project_summary.csv",
    index=False
)

customer_summary.to_csv(
    "output/customer_segments.csv",
    index=False
)

future_months.to_csv(
    "output/sales_forecast.csv",
    index=False
)

print("\n")
print("=" * 50)
print("CUSTOMER SEGMENTATION")
print("=" * 50)

print(
    customer_summary[
        "Customer_Segment"
    ].value_counts()
)
print("\n")
print("=" * 50)
print("SALES FORECAST")
print("=" * 50)

print(
    future_months[
        ["Month", "Forecast_Revenue"]
    ]
)
print("\nFiles Generated:")
print("1. output/retail_store_sales_cleaned.csv")
print("2. output/project_summary.csv")
print("3. charts/monthly_sales_trend.png")
print("4. charts/top_products.png")
print("5. charts/category_performance.png")
print("6. charts/revenue_by_city.png")
print("7. charts/payment_method_distribution.png")
print("8. charts/sales_channel_distribution.png")
print("9. charts/customer_age_groups.png")

print("\nPROJECT COMPLETED")