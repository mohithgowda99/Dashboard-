import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# ------------------ CONFIGURATION ------------------

HISTORICAL_FOLDER = "data/historical"
DAILY_FOLDER = "data/daily"
MASTER_FILE = "data/master.xlsx"
SPECIALTY_THRESHOLD = 999
ANOMALY_THRESHOLD = 0.2  # 20%

# ------------------ DATA FUNCTIONS ------------------

def load_all_data():
    all_files = []
    # Load historical files
    if os.path.exists(HISTORICAL_FOLDER):
        for filename in sorted(os.listdir(HISTORICAL_FOLDER)):
            if filename.endswith('.xlsx'):
                path = os.path.join(HISTORICAL_FOLDER, filename)
                df = pd.read_excel(path)
                all_files.append(df)
    # Load daily files
    if os.path.exists(DAILY_FOLDER):
        for filename in sorted(os.listdir(DAILY_FOLDER)):
            if filename.endswith('.xlsx'):
                path = os.path.join(DAILY_FOLDER, filename)
                df = pd.read_excel(path)
                all_files.append(df)
    if all_files:
        return pd.concat(all_files, ignore_index=True)
    else:
        return pd.DataFrame()

def remove_duplicates(df):
    if 'Bill ID' in df.columns:
        return df.drop_duplicates(subset=['Bill ID'], keep='last')
    return df

def save_master(df):
    os.makedirs(os.path.dirname(MASTER_FILE), exist_ok=True)
    df.to_excel(MASTER_FILE, index=False)

def load_master():
    if os.path.exists(MASTER_FILE):
        return pd.read_excel(MASTER_FILE)
    return pd.DataFrame()

def prepare_data(df):
    if 'InvoiceDate' not in df.columns and 'Date' in df.columns:
        df['InvoiceDate'] = pd.to_datetime(df['Date'], errors='coerce')
    if 'NetRevenue' not in df.columns and 'Gross' in df.columns:
        df['NetRevenue'] = pd.to_numeric(df['Gross'], errors='coerce').fillna(0)
    if 'Branch' not in df.columns and 'Branch Name' in df.columns:
        df['Branch'] = df['Branch Name']
    if 'ClientName' not in df.columns and 'Organisation' in df.columns:
        df['ClientName'] = df['Organisation']
    df['Month'] = pd.to_datetime(df['InvoiceDate']).dt.to_period('M').dt.start_time
    df['Tests_999_Flag'] = df['NetRevenue'] >= SPECIALTY_THRESHOLD
    return df

def revenue_trend(df, by_column, periods=6):
    df['Month'] = pd.to_datetime(df['Month'])
    grouped = df.groupby([by_column, 'Month'])['NetRevenue'].sum().reset_index()
    recent_months = grouped['Month'].drop_duplicates().sort_values().tail(periods)
    return grouped[grouped['Month'].isin(recent_months)]

def detect_anomalies(df, by_column, threshold=ANOMALY_THRESHOLD):
    trends = revenue_trend(df, by_column)
    anomalies = []
    for group in trends[by_column].unique():
        sub = trends[trends[by_column] == group].sort_values('Month')
        if len(sub) >= 2:
            prev = sub.iloc[-2]['NetRevenue']
            current = sub.iloc[-1]['NetRevenue']
            if prev > 0 and abs(current - prev) / prev >= threshold:
                anomalies.append({
                    'Group': group,
                    'Previous': prev,
                    'Current': current,
                    'Change %': round((current - prev) / prev * 100, 2)
                })
    return anomalies

def update_master_with_upload(uploaded_file):
    df_new = pd.read_excel(uploaded_file)
    df_master = load_master()
    df_combined = pd.concat([df_master, df_new], ignore_index=True)
    df_combined = remove_duplicates(df_combined)
    df_combined = prepare_data(df_combined)
    save_master(df_combined)
    return df_combined

# ------------------ UI ------------------

st.set_page_config(page_title="Truemedix Enhanced Dashboard", layout="wide")

st.title("📊 Truemedix Enhanced Analytics Dashboard")

# Load or initialize data
if 'df_master' not in st.session_state:
    df_all = load_all_data()
    df_all = remove_duplicates(df_all)
    df_all = prepare_data(df_all)
    save_master(df_all)
    st.session_state.df_master = df_all

df_master = load_master()

# Sidebar: upload daily data
with st.sidebar:
    st.header("📥 Upload Today's Data")
    uploaded_file = st.file_uploader("Choose file", type=["xlsx"])
    if uploaded_file is not None:
        st.session_state.df_master = update_master_with_upload(uploaded_file)
        st.success("✅ File uploaded and master dataset updated!")

# Main UI Tabs
tabs = st.tabs(["Overview", "Trends", "Alerts", "Upload Data"])

# ---------- Overview Tab ----------
with tabs[0]:
    st.header("📈 Overview")
    df = st.session_state.df_master
    total_revenue = df['NetRevenue'].sum()
    total_tests = df.shape[0]
    high_value_tests = df['Tests_999_Flag'].sum()
    st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    st.metric("Total Tests", f"{total_tests:,}")
    st.metric("High Value Tests (≥₹999)", f"{high_value_tests:,}")

# ---------- Trends Tab ----------
with tabs[1]:
    st.header("📊 Revenue Trends")
    df = st.session_state.df_master
    dimension = st.selectbox("Select dimension for trends", ["Branch", "ClientName", "Salesperson"])
    trends = revenue_trend(df, dimension)
    fig = px.line(trends, x="Month", y="NetRevenue", color=dimension, title=f"Revenue Trend by {dimension}")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Alerts Tab ----------
with tabs[2]:
    st.header("⚠ Alerts")
    df = st.session_state.df_master
    dimension = st.selectbox("Select dimension for anomaly detection", ["Branch", "ClientName", "Salesperson"], index=0)
    anomalies = detect_anomalies(df, dimension)
    if anomalies:
        for alert in anomalies:
            st.warning(f"{dimension}: {alert['Group']} → Change: {alert['Change %']}%")
    else:
        st.info("No significant anomalies detected.")

# ---------- Upload Data Tab ----------
with tabs[3]:
    st.header("📥 Upload Data")
    st.write("Use the sidebar to upload today's data. It will be merged with the master dataset automatically.")
    st.write("Files in the historical and daily folders are processed on app load.")

# ------------------ END ------------------