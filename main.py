# 💸 AI-Powered Personal Finance Tracker

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import re

# Streamlit config
st.set_page_config(page_title="AI Finance Tracker", page_icon="💸", layout="wide")

# Load and save categories
category_file = "categories.json"
if "categories" not in st.session_state:
    st.session_state.categories = {"Uncategorized": []}
if os.path.exists(category_file):
    with open(category_file, "r") as f:
        st.session_state.categories = json.load(f)

def save_categories():
    with open(category_file, "w") as f:
        json.dump(st.session_state.categories, f)

def categorize_transactions(df):
    df["Category"] = "Uncategorized"
    for category, keywords in st.session_state.categories.items():
        for idx, row in df.iterrows():
            if any(kw.lower() in row["Details"].lower() for kw in keywords):
                df.at[idx, "Category"] = category
    return df

def detect_currency_symbol(amount_column):
    symbols = {
        "AED": "د.إ", "USD": "$", "EUR": "€", "INR": "₹",
        "GBP": "£", "JPY": "¥", "CNY": "¥", "CAD": "C$", "AUD": "A$"
    }

    for amount in amount_column:
        if isinstance(amount, str):
            for code, symbol in symbols.items():
                if symbol in amount or re.match(rf"{code}", amount, re.IGNORECASE):
                    return code, symbol
    return "USD", "$"  # default fallback

def load_transactions(file):
    try:
        df = pd.read_csv(file)
        df.columns = [col.strip() for col in df.columns]

        currency_code, currency_sym = detect_currency_symbol(df["Amount"])
        st.session_state.currency = currency_code
        st.session_state.currency_symbol = currency_sym

        df["Amount"] = df["Amount"].astype(str).str.replace(r"[^\d\.-]", "", regex=True).astype(float)
        df["Date"] = pd.to_datetime(df["Date"], format="%d %b %Y")
        return categorize_transactions(df)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

def predict_spending(df):
    df = df[df["Debit/Credit"] == "Debit"]
    df["Month"] = df["Date"].dt.to_period("M").astype(str)
    monthly_totals = df.groupby("Month")["Amount"].sum().reset_index()

    X = np.arange(len(monthly_totals)).reshape(-1, 1)
    y = monthly_totals["Amount"].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    pred = model.predict([[len(X)]])[0][0]
    return pred

def detect_anomalies(df):
    debit_df = df[df["Debit/Credit"] == "Debit"].copy()
    model = IsolationForest(contamination=0.05)
    debit_df["Anomaly"] = model.fit_predict(debit_df[["Amount"]])
    return debit_df[debit_df["Anomaly"] == -1]

def cluster_expenses(df):
    debit_df = df[df["Debit/Credit"] == "Debit"].copy()
    if len(debit_df) > 5:
        kmeans = KMeans(n_clusters=3, random_state=42)
        debit_df["Cluster"] = kmeans.fit_predict(debit_df[["Amount"]])
        fig = px.scatter(debit_df, x="Date", y="Amount", color="Cluster", hover_data=["Details", "Category"])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Need at least 6 transactions to show clusters.")

def suggest_categories(df):
    tfidf = TfidfVectorizer()
    debit_df = df[df["Debit/Credit"] == "Debit"].copy()
    corpus = [d for cat in st.session_state.categories.values() for d in cat] + debit_df["Details"].tolist()
    tfidf_matrix = tfidf.fit_transform(corpus)
    similarity = cosine_similarity(tfidf_matrix[-len(debit_df):], tfidf_matrix[:-len(debit_df)])
    suggestions = []
    for i, sim in enumerate(similarity):
        best_idx = np.argmax(sim)
        if sim[best_idx] > 0.3:
            for cat, keywords in st.session_state.categories.items():
                if corpus[best_idx] in keywords:
                    suggestions.append(cat)
                    break
        else:
            suggestions.append("Uncategorized")
    debit_df["Suggested"] = suggestions
    return debit_df

def main():
    st.title("💰 AI Personal Finance Tracker")
    uploaded_file = st.file_uploader("📁 Upload your transaction CSV", type="csv")

    if uploaded_file:
        df = load_transactions(uploaded_file)
        if df is not None:
            currency = st.session_state.get("currency", "USD")
            currency_symbol = st.session_state.get("currency_symbol", "$")

            st.session_state.debits_df = df[df["Debit/Credit"] == "Debit"].copy()
            credits_df = df[df["Debit/Credit"] == "Credit"].copy()
            pred = predict_spending(df)
            anomalies = detect_anomalies(df)
            suggestions_df = suggest_categories(df)

            st.subheader("📊 Dashboard Overview")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🗖️ Total Spent", f"{st.session_state.debits_df['Amount'].sum():,.2f} {currency_symbol}")
            with col2:
                st.metric("💳 Total Received", f"{credits_df['Amount'].sum():,.2f} {currency_symbol}")
            with col3:
                st.metric("🔮 Next Month Spend Est.", f"{pred:,.2f} {currency_symbol}")

            st.markdown("---")

            summary = st.session_state.debits_df.groupby("Category")["Amount"].sum().reset_index()
            summary = summary.sort_values("Amount", ascending=False)
            col1, col2 = st.columns(2)
            with col1:
                st.dataframe(summary, use_container_width=True)
            with col2:
                pie_chart = px.pie(summary, values="Amount", names="Category", title="Expense Breakdown")
                st.plotly_chart(pie_chart, use_container_width=True)

            st.markdown("---")
            st.subheader("🚨 Anomalies & 🧪 Suggestions")
            col1, col2 = st.columns(2)
            with col1:
                if anomalies.empty:
                    st.success("No anomalies detected.")
                else:
                    st.warning("Suspicious transactions:")
                    st.dataframe(anomalies[["Date", "Details", "Amount", "Category"]])
            with col2:
                st.dataframe(suggestions_df[["Date", "Details", "Amount", "Category", "Suggested"]], use_container_width=True)

            st.markdown("---")
            st.subheader("🤖 Spending Pattern Clustering")
            cluster_expenses(df)

            st.markdown("---")
            with st.expander("🛠️ Manage Categories"):
                new_cat = st.text_input("New Category Name")
                if st.button("➕ Add Category") and new_cat:
                    if new_cat not in st.session_state.categories:
                        st.session_state.categories[new_cat] = []
                        save_categories()
                        st.rerun()
                edited_df = st.data_editor(
                    st.session_state.debits_df,
                    column_config={
                        "Category": st.column_config.SelectboxColumn("Category", options=list(st.session_state.categories.keys())),
                        "Date": st.column_config.DateColumn("Date", format="DD/MM/YYYY"),
                        "Amount": st.column_config.NumberColumn("Amount", format=f"%.2f {currency_symbol}")
                    },
                    use_container_width=True,
                    hide_index=True,
                    key="category_editor"
                )
                if st.button("✅ Save Categories"):
                    for idx, row in edited_df.iterrows():
                        cat = row["Category"]
                        st.session_state.debits_df.at[idx, "Category"] = cat
                        st.session_state.categories[cat] = list(set(st.session_state.categories.get(cat, []) + [row["Details"]]))
                    save_categories()

main()
