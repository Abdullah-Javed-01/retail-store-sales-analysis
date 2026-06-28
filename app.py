# ==========================================================
# RETAIL BUSINESS INTELLIGENCE DASHBOARD
# Developed by Abdullah Javed
# ==========================================================

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from PIL import Image


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Retail Business Intelligence Dashboard",
    page_icon="📊",
    layout="wide"
)


# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "data/retail_store_sales_cleaned.csv"
    )

    df["Purchase_Date"] = pd.to_datetime(
        df["Purchase_Date"]
    )

    df["Month"] = df["Purchase_Date"].dt.strftime("%Y-%m")

    df["Product_Category"] = (
        df["Product_Category"]
        .fillna("Unknown")
        .replace("nan", "Unknown")
    )

    return df


df = load_data()


# ==========================================================
# LOAD FORECAST
# ==========================================================

@st.cache_data
def load_forecast():

    return pd.read_csv(
        "output/sales_forecast.csv"
    )


forecast_df = load_forecast()


# ==========================================================
# LOAD ELBOW CHART
# ==========================================================

elbow_chart = Image.open(
    "charts/elbow_method.png"
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📊 Retail BI Dashboard")


page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Executive Overview",

        "📈 Sales Analytics",

        "🛍 Product Analytics",

        "🏙 Geographic Analytics",

        "👥 Customer Analytics",

        "📅 Revenue Forecast",

        "💡 Business Insights",

        "🔍 Data Explorer"

    ]

)


st.sidebar.markdown("---")

st.sidebar.header("Filters")


cities = st.sidebar.multiselect(

    "City",

    options=sorted(
        df["Customer_City"].unique()
    ),

    default=sorted(
        df["Customer_City"].unique()
    )

)


categories = st.sidebar.multiselect(

    "Category",

    options=sorted(
        df["Product_Category"].unique()
    ),

    default=sorted(
        df["Product_Category"].unique()
    )

)


payments = st.sidebar.multiselect(

    "Payment Method",

    options=sorted(
        df["Payment_Method"].unique()
    ),

    default=sorted(
        df["Payment_Method"].unique()
    )

)


filtered_df = df[

    (df["Customer_City"].isin(cities))

    &

    (df["Product_Category"].isin(categories))

    &

    (df["Payment_Method"].isin(payments))

].copy()


# ==========================================================
# FUNCTIONS
# ==========================================================

def calculate_kpis(data):

    revenue = data["Total_Amount"].sum()

    orders = len(data)

    avg_order = data["Total_Amount"].mean()

    return revenue, orders, avg_order


def monthly_sales(data):

    return (

        data.groupby("Month")["Total_Amount"]

        .sum()

        .reset_index()

    )


def category_sales(data):

    return (

        data.groupby("Product_Category")["Total_Amount"]

        .sum()

        .reset_index()

    )


def city_sales(data):

    return (

        data.groupby("Customer_City")["Total_Amount"]

        .sum()

        .reset_index()

    )


def top_products(data):

    return (

        data.groupby("Product_Name")["Total_Amount"]

        .sum()

        .sort_values(ascending=False)

        .head(10)

        .reset_index()

    )


def payment_distribution(data):

    payment = (

        data["Payment_Method"]

        .value_counts()

        .reset_index()

    )

    payment.columns = [

        "Payment_Method",

        "Count"

    ]

    return payment

# ==========================================================
# REUSABLE SECTION HEADER
# ==========================================================

def section_header(title, description):
    st.title(title)
    st.caption(description)
    st.divider()


# ==========================================================
# EXECUTIVE OVERVIEW
# ==========================================================

if page == "🏠 Executive Overview":

    st.info(
    """
    ## 📖 About This Dashboard

    This Retail Business Intelligence Dashboard provides an interactive
    analysis of retail sales performance, customer behavior, revenue trends,
    customer segmentation, and future revenue projections.

    The dashboard enables decision-makers to monitor business performance,
    identify customer segments, evaluate sales trends, and support
    data-driven strategic decisions.
    """
    )
    
    section_header(
        "🏠 Executive Overview",
        "High-level summary of retail sales performance."
    )
    

    revenue, orders, avg_order = calculate_kpis(filtered_df)

    best_product = (
        filtered_df.groupby("Product_Name")["Total_Amount"]
        .sum()
        .idxmax()
    )

    best_category = (
        filtered_df.groupby("Product_Category")["Total_Amount"]
        .sum()
        .idxmax()
    )

    best_city = (
        filtered_df.groupby("Customer_City")["Total_Amount"]
        .sum()
        .idxmax()
    )

    peak_month = (
        monthly_sales(filtered_df)
        .sort_values("Total_Amount", ascending=False)
        .iloc[0]["Month"]
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "💰 Total Revenue",
        f"PKR {revenue:,.0f}"
    )

    c2.metric(
        "🛒 Total Orders",
        f"{orders:,}"
    )

    c3.metric(
        "💳 Average Order Value",
        f"PKR {avg_order:,.0f}"
    )

    st.divider()

    c4, c5, c6, c7 = st.columns(4)

    c4.metric(
        "🏆 Best Product",
        best_product
    )

    c5.metric(
        "📦 Top Category",
        best_category
    )

    c6.metric(
        "🏙 Best City",
        best_city
    )

    c7.metric(
        "📅 Peak Month",
        peak_month
    )

    st.divider()

    monthly = monthly_sales(filtered_df)

    fig = px.line(
        monthly,
        x="Month",
        y="Total_Amount",
        markers=True,
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ==========================================================
# SALES ANALYTICS
# ==========================================================

elif page == "📈 Sales Analytics":

    section_header(
        "📈 Sales Analytics",
        "Monthly sales performance and revenue trends."
    )

    monthly = monthly_sales(filtered_df)

    fig = px.line(
        monthly,
        x="Month",
        y="Total_Amount",
        markers=True,
        title="Monthly Revenue Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    col1, col2 = st.columns(2)

    category = category_sales(filtered_df)

    fig2 = px.bar(
        category,
        x="Product_Category",
        y="Total_Amount",
        title="Revenue by Category"
    )

    col1.plotly_chart(
        fig2,
        use_container_width=True
    )

    payment = payment_distribution(filtered_df)

    fig3 = px.pie(
        payment,
        names="Payment_Method",
        values="Count",
        title="Payment Distribution"
    )

    col2.plotly_chart(
        fig3,
        use_container_width=True
    )


# ==========================================================
# PRODUCT ANALYTICS
# ==========================================================

elif page == "🛍 Product Analytics":

    section_header(
        "🛍 Product Analytics",
        "Product performance and revenue contribution."
    )

    top = top_products(filtered_df)

    fig = px.bar(
        top,
        x="Product_Name",
        y="Total_Amount",
        title="Top 10 Products"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("📋 Product Revenue")

    product_table = (
        filtered_df.groupby("Product_Name")["Total_Amount"]
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )

    product_table.columns = [
        "Product",
        "Revenue (PKR)"
    ]

    product_table["Revenue (PKR)"] = (
        product_table["Revenue (PKR)"]
        .map(lambda x: f"{x:,.0f}")
    )

    st.dataframe(
        product_table,
        use_container_width=True,
        hide_index=True
    )
    
# ==========================================================
# GEOGRAPHIC ANALYTICS
# ==========================================================

elif page == "🏙 Geographic Analytics":

    section_header(
        "🏙 Geographic Analytics",
        "Sales performance across different cities."
    )

    city = city_sales(filtered_df)

    fig = px.bar(
        city,
        x="Customer_City",
        y="Total_Amount",
        title="Revenue by City",
        text_auto=".2s"
    )

    fig.update_layout(
        xaxis_title="City",
        yaxis_title="Revenue (PKR)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("🏆 City Revenue Ranking")

    city_table = city.copy()

    city_table.columns = [
        "City",
        "Revenue (PKR)"
    ]

    city_table["Revenue (PKR)"] = (
        city_table["Revenue (PKR)"]
        .map(lambda x: f"{x:,.0f}")
    )

    st.dataframe(
        city_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    best_city = city.sort_values(
        "Total_Amount",
        ascending=False
    ).iloc[0]["Customer_City"]

    st.success(
        f"🏙 **{best_city}** generated the highest revenue among all cities."
    )

# ==========================================================
# CUSTOMER ANALYTICS
# ==========================================================

elif page == "👥 Customer Analytics":

    section_header(
        "👥 Customer Analytics",
        "Customer segmentation and behavioral analysis."
    )

    customer_summary = (
        filtered_df.groupby("Customer_ID")
        .agg(
            Total_Spending=("Total_Amount", "sum"),
            Total_Orders=("Transaction_ID", "count"),
            Average_Order_Value=("Total_Amount", "mean")
        )
        .reset_index()
    )

    if len(customer_summary) >= 3:

        features = customer_summary[
            [
                "Total_Spending",
                "Total_Orders",
                "Average_Order_Value"
            ]
        ]

        scaler = StandardScaler()

        scaled = scaler.fit_transform(features)

        model = KMeans(
            n_clusters=3,
            random_state=42,
            n_init=10
        )

        customer_summary["Cluster"] = model.fit_predict(scaled)

        cluster_avg = (
            customer_summary
            .groupby("Cluster")["Total_Spending"]
            .mean()
            .sort_values()
        )

        order = cluster_avg.index.tolist()

        labels = {

            order[0]: "Budget Customers",

            order[1]: "Regular Customers",

            order[2]: "High Value Customers"

        }

        customer_summary["Customer_Segment"] = (
            customer_summary["Cluster"]
            .map(labels)
        )

        st.subheader("📊 Customer Segment KPIs")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "💎 High Value",
            (
                customer_summary["Customer_Segment"]
                == "High Value Customers"
            ).sum()
        )

        c2.metric(
            "👥 Regular",
            (
                customer_summary["Customer_Segment"]
                == "Regular Customers"
            ).sum()
        )

        c3.metric(
            "💰 Budget",
            (
                customer_summary["Customer_Segment"]
                == "Budget Customers"
            ).sum()
        )

        st.divider()

        st.subheader("📈 Customer Segmentation")

        fig = px.scatter(

            customer_summary,

            x="Average_Order_Value",

            y="Total_Spending",

            color="Customer_Segment",

            hover_data=[
                "Customer_ID",
                "Total_Orders"
            ]

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("📊 Segment Distribution")

        segment = (
            customer_summary["Customer_Segment"]
            .value_counts()
            .reset_index()
        )

        segment.columns = [
            "Customer Segment",
            "Customers"
        ]

        fig = px.bar(

            segment,

            x="Customer Segment",

            y="Customers"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.divider()

        st.subheader("📊 Cluster Validation")

        st.image(
            elbow_chart,
            use_container_width=True
        )

        st.info(
            """
The Elbow Method measures the Within-Cluster Sum of Squares (WCSS).

The bend in the curve appears around **3 clusters**, indicating that
three customer segments provide a good balance between simplicity and
meaningful grouping.
"""
        )

        st.divider()

        st.subheader("📋 Customer Segment Details")

        display = customer_summary.drop(
            columns=["Cluster"]
        )

        st.dataframe(

            display,

            use_container_width=True,

            hide_index=True

        )

        st.divider()

        st.subheader("💡 Recommendations")

        st.success("""
💎 High Value Customers

• VIP rewards

• Early product access

• Premium promotions
""")

        st.info("""
👥 Regular Customers

• Upselling

• Cross-selling

• Loyalty program
""")

        st.warning("""
💰 Budget Customers

• Coupons

• Bundle offers

• Seasonal discounts
""")

    else:

        st.warning(
            "Not enough customers for segmentation."
        )
        
# ==========================================================
# REVENUE FORECAST
# ==========================================================

elif page == "📅 Revenue Forecast":

    section_header(
        "📅 Revenue Forecast",
        "Trend-based revenue projection using Linear Regression."
    )

    # =====================================================
    # MODEL INFORMATION
    # =====================================================

    st.info(
        """
### 🤖 Forecast Model

**Model Used:** Trend-Based Linear Regression

This forecast is generated using historical monthly revenue.
It provides a business projection for the next three months
and should be interpreted as a planning tool rather than
an exact prediction.
"""
    )

    # =====================================================
    # KPI CALCULATIONS
    # =====================================================

    expected_revenue = forecast_df["Forecast_Revenue"].sum()

    highest = forecast_df.loc[
        forecast_df["Forecast_Revenue"].idxmax()
    ]

    highest_month = highest["Month"]

    average_forecast = (
        forecast_df["Forecast_Revenue"]
        .mean()
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "📈 Expected Revenue",
        f"PKR {expected_revenue:,.0f}"
    )

    c2.metric(
        "🏆 Highest Forecast Month",
        highest_month
    )

    c3.metric(
        "📊 Average Monthly Revenue",
        f"PKR {average_forecast:,.0f}"
    )

    st.divider()

    # =====================================================
    # COMBINE HISTORICAL + FORECAST
    # =====================================================

    history = monthly_sales(filtered_df)

    history.columns = [
        "Month",
        "Revenue"
    ]

    forecast = forecast_df.rename(
        columns={
            "Forecast_Revenue": "Revenue"
        }
    )

    history["Type"] = "Historical"

    forecast["Type"] = "Forecast"

    combined = pd.concat(
        [
            history,
            forecast[
                [
                    "Month",
                    "Revenue",
                    "Type"
                ]
            ]
        ],
        ignore_index=True
    )

    # =====================================================
    # CHART
    # =====================================================

    st.subheader(
        "📈 Historical Revenue vs Future Projection"
    )

    fig = px.line(

        combined,

        x="Month",

        y="Revenue",

        color="Type",

        markers=True

    )

    fig.update_layout(

        xaxis_title="Month",

        yaxis_title="Revenue (PKR)",

        legend_title=""

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =====================================================
    # FORECAST TABLE
    # =====================================================

    st.subheader("📋 Revenue Forecast")

    forecast_table = forecast_df.copy()

    trend = []

    values = forecast_table["Forecast_Revenue"].tolist()

    for i in range(len(values)):

        if i == 0:
            trend.append("▶ Stable")

        elif values[i] >= values[i-1]:
            trend.append("▲ Increasing")

        else:
            trend.append("▼ Decreasing")

    forecast_table["Trend"] = trend

    forecast_table = forecast_table[
        [
            "Month",
            "Forecast_Revenue",
            "Trend"
        ]
    ]

    forecast_table.columns = [

        "Forecast Month",

        "Projected Revenue (PKR)",

        "Trend"

    ]

    forecast_table[
        "Projected Revenue (PKR)"
    ] = forecast_table[
        "Projected Revenue (PKR)"
    ].map(
        lambda x: f"{x:,.0f}"
    )

    st.dataframe(

        forecast_table,

        use_container_width=True,

        hide_index=True

    )

    st.divider()

    # =====================================================
    # INTERPRETATION
    # =====================================================

    growth = (
        forecast_df["Forecast_Revenue"]
        .iloc[-1]
        -
        forecast_df["Forecast_Revenue"]
        .iloc[0]
    )

    st.subheader("📖 Forecast Interpretation")

    if growth > 0:

        st.success(
            """
Revenue is projected to increase gradually over the
next three months.

This indicates stable demand and suggests that
inventory planning should prioritize
high-performing products while maintaining
sufficient stock levels.
"""
        )

    elif growth < 0:

        st.warning(
            """
Revenue is projected to decline over the
forecast period.

Management should investigate seasonal
effects, promotional strategy and
customer retention initiatives.
"""
        )

    else:

        st.info(
            """
Revenue is expected to remain relatively stable.

Focus should remain on customer retention
and operational efficiency.
"""
        )

    st.divider()

    # =====================================================
    # RECOMMENDATIONS
    # =====================================================

    st.subheader("💡 Recommended Actions")

    st.success(
        """
✅ Increase inventory for top-performing products.

✅ Continue promotions during historically strong sales periods.

✅ Monitor actual revenue against projected revenue each month.

✅ Focus marketing efforts on high-performing cities.
"""
    )      
    
# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

elif page == "💡 Business Insights":

    section_header(
        "💡 Executive Business Insights",
        "Key findings and strategic recommendations generated from the analysis."
    )

    # ------------------------------------------------------
    # Calculate Insights
    # ------------------------------------------------------

    revenue, orders, avg_order = calculate_kpis(filtered_df)

    best_product = (
        filtered_df.groupby("Product_Name")["Total_Amount"]
        .sum()
        .idxmax()
    )

    best_category = (
        filtered_df.groupby("Product_Category")["Total_Amount"]
        .sum()
        .idxmax()
    )

    best_city = (
        filtered_df.groupby("Customer_City")["Total_Amount"]
        .sum()
        .idxmax()
    )

    popular_payment = (
        filtered_df["Payment_Method"]
        .mode()[0]
    )

    peak_month = (
        monthly_sales(filtered_df)
        .sort_values("Total_Amount", ascending=False)
        .iloc[0]["Month"]
    )

    # ------------------------------------------------------
    # Executive Summary
    # ------------------------------------------------------

    st.subheader("📊 Executive Summary")

    c1, c2 = st.columns(2)

    with c1:

        st.success(f"🏆 Best Performing Product\n\n**{best_product}**")

        st.success(f"📦 Top Revenue Category\n\n**{best_category}**")

        st.success(f"🏙 Highest Revenue City\n\n**{best_city}**")

        st.success(f"💳 Most Popular Payment Method\n\n**{popular_payment}**")

    with c2:

        st.info(f"📅 Peak Sales Month\n\n**{peak_month}**")

        st.info(f"💰 Total Revenue\n\n**PKR {revenue:,.0f}**")

        st.info(f"🛒 Total Orders\n\n**{orders:,}**")

        st.info(f"💳 Average Order Value\n\n**PKR {avg_order:,.0f}**")

    st.divider()

    # ------------------------------------------------------
    # Forecast Outlook
    # ------------------------------------------------------

    st.subheader("📈 Forecast Outlook")

    forecast_growth = (
        forecast_df["Forecast_Revenue"].iloc[-1]
        -
        forecast_df["Forecast_Revenue"].iloc[0]
    )

    if forecast_growth > 0:

        st.success(
            """
### Positive Outlook

Revenue is projected to grow steadily over the
next three months.

Current sales trends indicate healthy business
performance with continued growth potential.
"""
        )

    elif forecast_growth < 0:

        st.warning(
            """
### Declining Outlook

Revenue is projected to decrease over the
forecast period.

Additional marketing efforts and customer
retention strategies are recommended.
"""
        )

    else:

        st.info(
            """
### Stable Outlook

Revenue is expected to remain relatively
consistent over the next three months.
"""
        )

    st.divider()

    # ------------------------------------------------------
    # Strategic Recommendations
    # ------------------------------------------------------

    st.subheader("🎯 Strategic Recommendations")

    rec1, rec2 = st.columns(2)

    with rec1:

        st.success("""
### 📦 Product Strategy

• Expand inventory for top products

• Prioritize Electronics

• Bundle slow-moving products
""")

        st.info("""
### 💳 Payment Strategy

• Encourage digital payments

• Offer cashback promotions

• Improve checkout experience
""")

    with rec2:

        st.warning("""
### 👥 Customer Strategy

• Reward High Value Customers

• Launch loyalty programs

• Target Budget Customers with discounts
""")

        st.success("""
### 🏙 Growth Strategy

• Increase investment in top-performing cities

• Replicate successful campaigns

• Monitor regional demand
""")

    st.divider()

    # ------------------------------------------------------
    # Final Conclusion
    # ------------------------------------------------------

    st.subheader("📌 Executive Conclusion")

    st.markdown(
        f"""
The retail business generated **PKR {revenue:,.0f}**
from **{orders:,}** completed orders.

**{best_category}** is the strongest revenue-driving
category, while **{best_product}** remains the
highest-performing product.

Customer segmentation identifies clear opportunities
for targeted marketing, and the revenue forecast
suggests a stable business outlook over the coming
months.

Overall, the business demonstrates healthy sales
performance with opportunities for further growth
through customer retention, inventory optimization,
and strategic marketing.
"""
    )

# ==========================================================
# DATA EXPLORER
# ==========================================================

elif page == "🔍 Data Explorer":

    section_header(
        "🔍 Data Explorer",
        "Explore and download the filtered transaction data."
    )

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        f"{len(filtered_df):,}"
    )

    c2.metric(
        "Columns",
        filtered_df.shape[1]
    )

    c3.metric(
        "Missing Values",
        int(filtered_df.isna().sum().sum())
    )

    st.divider()

    search = st.text_input(
        "🔎 Search Product"
    )

    data = filtered_df.copy()

    if search:

        data = data[
            data["Product_Name"]
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    csv = data.to_csv(index=False)

    st.download_button(
        label="⬇ Download Filtered Dataset",
        data=csv,
        file_name="filtered_transactions.csv",
        mime="text/csv"
    )