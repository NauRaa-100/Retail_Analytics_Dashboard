import gradio as gr
import pandas as pd
import plotly.express as px
from joblib import load
import numpy as np

# --------------------------------------------------
# Load Models & Data
# --------------------------------------------------
kmeans_pipeline = load("kmeans_rfm_pipeline.pkl")
clv_pipeline = load("clv_pipeline.pkl")

rfm = pd.read_csv("rfm_segments.csv")
df_sales = pd.read_csv("new_Retail.csv")
basket_rules = pd.read_csv("basket_rules.csv")

# Fix dates
df_sales["InvoiceDate"] = pd.to_datetime(df_sales["InvoiceDate"], errors="ignore")
df_sales["Weekday"] = df_sales["InvoiceDate"].dt.dayofweek
df_sales["Month"] = df_sales["InvoiceDate"].dt.month

# Profit column
if "Cost" in df_sales.columns:
    df_sales["Profit"] = df_sales["TotalPrice"] - df_sales["Cost"]
else:
    df_sales["Profit"] = df_sales["TotalPrice"] * 0.2

customer_ids = sorted(rfm["CustomerID"].unique())
product_names = sorted(df_sales["Description"].unique())


# --------------------------------------------------
# Functions
# --------------------------------------------------
def rfm_kmeans(customer_id):
    cust = rfm[rfm["CustomerID"] == customer_id]

    if cust.empty:
        return "CustomerID not found", None, None, None, None

    segment = str(cust["Segment"].values[0])
    cluster = int(
        kmeans_pipeline.predict(
            cust[["Recency", "Frequency", "Monetary"]]
        )[0]
    )

    figs = []

    # Heatmap
    heatmap_data = (
        rfm.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
        .mean()
        .reset_index()
    )
    fig1 = px.imshow(
        heatmap_data.set_index("Segment"),
        text_auto=True,
        color_continuous_scale="YlGnBu",
        title="RFM Heatmap",
    )
    figs.append(fig1)

    # Pie segmentation
    pie_data = rfm["Segment"].value_counts().reset_index()
    pie_data.columns = ["Segment", "Count"]
    fig2 = px.pie(
        pie_data, names="Segment", values="Count", title="Customer Segment Distribution"
    )
    figs.append(fig2)

    # Top 10
    top10 = (
        df_sales.groupby("Description")["TotalPrice"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig3 = px.bar(top10, x="Description", y="TotalPrice", title="Top 10 Products")
    figs.append(fig3)

    # Bottom 10
    bottom10 = (
        df_sales.groupby("Description")["TotalPrice"]
        .sum()
        .sort_values()
        .head(10)
        .reset_index()
    )
    fig4 = px.bar(bottom10, x="Description", y="TotalPrice", title="Bottom 10 Products")
    figs.append(fig4)

    return (
        f"Segment: {segment} | Cluster: {cluster}",
        figs[0],
        figs[1],
        figs[2],
        figs[3],
    )


def sales_trends():
    figs = []

    monthly_sales = df_sales.groupby("Month")["TotalPrice"].sum().reset_index()
    figs.append(px.line(monthly_sales, x="Month", y="TotalPrice", markers=True, title="Monthly Sales"))

    weekly_sales = df_sales.groupby("Weekday")["TotalPrice"].sum().reset_index()
    figs.append(px.line(weekly_sales, x="Weekday", y="TotalPrice", markers=True, title="Weekly Sales"))

    return figs


def profit_analysis():
    figs = []

    prof_m = df_sales.groupby("Month")["Profit"].sum().reset_index()
    figs.append(px.bar(prof_m, x="Month", y="Profit", title="Monthly Profit"))

    prof_p = (
        df_sales.groupby("Description")["Profit"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    figs.append(px.bar(prof_p, x="Description", y="Profit", title="Top Profit Products"))

    merge_seg = df_sales.merge(rfm[["CustomerID", "Segment"]], on="CustomerID")
    prof_s = merge_seg.groupby("Segment")["Profit"].sum().reset_index()
    figs.append(px.pie(prof_s, names="Segment", values="Profit", title="Profit by Segment"))

    return figs


def clv_predict(customer_id):
    cust = rfm[rfm["CustomerID"] == customer_id]
    if cust.empty:
        return "CustomerID not found"
    clv_value = float(
        clv_pipeline.predict(
            cust[["Recency", "Frequency", "Monetary"]]
        )[0]
    )
    return f"Predicted CLV: {clv_value:.2f}"


def apriori_suggestions(product_name):
    rules = basket_rules[
        basket_rules["antecedents"].apply(lambda x: product_name in eval(str(x)))
    ]

    if rules.empty:
        return "No suggestions found."

    out = set()
    for c in rules["consequents"]:
        out.update(eval(str(c)))

    return ", ".join(list(out))


# --------------------------------------------------
# UI (ALL inside Blocks)
# --------------------------------------------------
with gr.Blocks() as demo:

    gr.Markdown("# 🌟 Retail Analytics Dashboard")
    gr.Markdown("**⚠️ Note:** Models work only on the provided Retail dataset.")

    # ==================== RFM =======================
    with gr.Tab("RFM + KMeans"):

        customer_input = gr.Dropdown(choices=customer_ids, label="CustomerID")
        rfm_output = gr.Textbox(label="Result")

        with gr.Row():
            heatmap_plot = gr.Plot()
            pie_plot = gr.Plot()

        with gr.Row():
            top_plot = gr.Plot()
            bottom_plot = gr.Plot()

        customer_input.change(
            rfm_kmeans,
            inputs=customer_input,
            outputs=[rfm_output, heatmap_plot, pie_plot, top_plot, bottom_plot],
        )

    # ================== Trends ======================
    with gr.Tab("Sales Trends"):
        figs = sales_trends()
        gr.Plot(value=figs[0])
        gr.Plot(value=figs[1])

    # ================== Profit ======================
    with gr.Tab("Profit Analysis"):
        figs = profit_analysis()
        for f in figs:
            gr.Plot(value=f)

    # ================== CLV =========================
    with gr.Tab("CLV Prediction"):
        customer_in = gr.Dropdown(choices=customer_ids, label="CustomerID")
        out_clv = gr.Textbox()
        customer_in.change(clv_predict, inputs=customer_in, outputs=out_clv)

    # ================== Basket ======================
    with gr.Tab("Basket Analysis"):
        product_input = gr.Dropdown(choices=product_names, label="Product Name")
        out_rules = gr.Textbox()
        product_input.change(apriori_suggestions, inputs=product_input, outputs=out_rules)

# HuggingFace Fix → no SSR + enable share
demo.launch(share=True, ssr_mode=False)
