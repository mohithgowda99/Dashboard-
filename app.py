import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import os
import glob
from datetime import datetime
from collections import defaultdict

# ---- CONFIG ----
HISTORICAL = "historical"
DAILY = "daily"
REQUIRED_COLUMNS = [
    'Date','Branch','Salesperson','ClientName','TestName',
    'Specialty','TestMRP','BilledAmount','NetRevenue','InvoiceID','ClientAddedDate'
]

# ---- Modern UI CSS ----
st.set_page_config(page_title="Truemedix Advanced Analytics", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    body, .stApp {background-color: #191c24 !important;}
    .block-container {padding-top:2.1rem;}
    .stMetric, .css-1kyxreq, .element-container {
        background: #23263b !important; border-radius:14px !important;
        padding:1.1rem 1.2rem!important; color:white!important;
    }
    .stPlotlyChart {background:#23263b !important;}
    .stDataFrame {background: #23263b !important; color: #fff !important;}
    .css-1cpxqw2 a, .st-af {color: #00c9a7;}
    </style>
""", unsafe_allow_html=True)

# ---- Utilities ----
def ensure_dirs():
    Path(HISTORICAL).mkdir(exist_ok=True)
    Path(DAILY).mkdir(exist_ok=True)

def robust_read_excel(fp):
    for engine in [None, "openpyxl"]:
        try:
            return pd.read_excel(fp, engine=engine)
        except Exception:
            continue
    raise Exception(f"Could not read Excel file '{fp}' with any engine.")

def check_required_columns(df):
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]

def merge_files():
    all_files = glob.glob(f"{HISTORICAL}/*") + glob.glob(f"{DAILY}/*")
    frames = []
    seen_ids = set()
    error_log = defaultdict(list)
    for fp in all_files:
        ext = fp.split(".")[-1]
        try:
            if ext in ("xlsx", "xls"):
                df = robust_read_excel(fp)
            elif ext == "csv":
                df = pd.read_csv(fp)
            else:
                st.warning(f"File ignored (unsupported type): {fp}")
                continue
            df.columns = [c.strip() for c in df.columns]
        except Exception as e:
            error_log["failed_files"].append(f"{fp}: {e}")
            continue

        missing = check_required_columns(df)
        if missing:
            error_log["missing_columns"].append(f"{fp}: {', '.join(missing)}")
            continue

        # Deduplicate by InvoiceID within all files
        df = df[~df['InvoiceID'].astype(str).isin(seen_ids)]
        seen_ids.update(df['InvoiceID'].astype(str))
        frames.append(df)
    if not frames:
        return pd.DataFrame(), error_log
    data = pd.concat(frames, ignore_index=True)
    data['Date'] = pd.to_datetime(data['Date'], errors='coerce')
    return data, error_log

def analyze_and_transform(df):
    # Add derived columns
    df['Month'] = df['Date'].dt.strftime('%b')
    # Variance and Receivables
    df["Receivables"] = df['BilledAmount'] - df['NetRevenue']
    return df

def anomaly_detect(df):
    # Monthly revenue anomalies detection (simple z-score check)
    month_rev = df.groupby(df['Date'].dt.to_period('M'))['NetRevenue'].sum()
    mean = month_rev.mean()
    std = month_rev.std(ddof=0)
    anomalies = []
    if std > 0:
        for i, val in month_rev.items():
            z = (val - mean)/std
            if abs(z) > 2:
                anomalies.append({"month": str(i), "revenue": int(val), "z_score": float(z)})
    return anomalies

# ---- Sidebar: File Upload & Management ----
ensure_dirs()
st.sidebar.header("📤 Upload Files")
with st.sidebar:
    upload_type = st.radio("Save uploaded files to:", ["Daily", "Historical"], horizontal=True)
    file_area = st.file_uploader(
        f"Upload Excel/CSV ({upload_type})", 
        type=['csv','xlsx','xls'],
        accept_multiple_files=True
    )
    if file_area:
        for file_obj in file_area:
            folder = DAILY if upload_type == "Daily" else HISTORICAL
            path = Path(folder) / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file_obj.name}"
            with open(path, "wb") as wfile:
                wfile.write(file_obj.getbuffer())
        st.success(f"Uploaded {len(file_area)} file(s) to /{folder}")

    # File listing & deletion
    for label, folder in [("Historic files", HISTORICAL), ("Daily files", DAILY)]:
        st.markdown(f"**{label}:**")
        files = sorted(list(Path(folder).glob("*")))
        for f in files:
            c1, c2 = st.columns([5,1])
            c1.write(f.name)
            if c2.button("🗑", key=f"del_{folder}_{f.name}"):
                f.unlink()
                st.experimental_rerun()

    st.divider()
    st.caption("⬆️ Choose file type, upload, see files above. Use 'Refresh All Data' below when ready.")

# ---- Process All Data ----
st.sidebar.divider()
st.sidebar.header("🚀 Data Processing")
if st.sidebar.button("Refresh All Data", use_container_width=True):
    data, log = merge_files()
    if len(data):
        data = analyze_and_transform(data)
    st.session_state.data = data
    st.session_state.error_log = log
    st.experimental_rerun()

# ---- MAIN DASHBOARD ----

data = st.session_state.get("data")
errlog = st.session_state.get("error_log", {})
if data is None or data.empty:
    st.info("Upload and refresh your files using the sidebar.")
    st.write("""
    #### Expected file columns
    - Date, Branch, Salesperson, ClientName, TestName, Specialty, TestMRP, BilledAmount, NetRevenue, InvoiceID, ClientAddedDate
    """)
    if errlog:
        if errlog.get("missing_columns"):
            st.error(f"Files missing columns:\n" + "\n".join(errlog["missing_columns"]))
        if errlog.get("failed_files"):
            st.error(f"Files failed to load:\n" + "\n".join(errlog["failed_files"]))
    st.stop()

# --- Metrics
current_month = data['Date'].max().month
month_data = data[data['Date'].dt.month == current_month]
total_rev = int(data['NetRevenue'].sum())
unique_clients = data['ClientName'].nunique()
monthly_rev = int(month_data['NetRevenue'].sum())
receivables = int(data["Receivables"].sum())
monthly_added = data.groupby('Month')['ClientName'].nunique().reindex(
    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']).fillna(0).astype(int)

# --- Chart: Revenue Trend
monthly_group = data.groupby('Month')['NetRevenue'].sum().reindex(
    ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']).fillna(0)
fig_line = go.Figure()
fig_line.add_trace(go.Scatter(
    y=monthly_group,
    x=monthly_group.index,
    fill='tozeroy',
    line=dict(width=5, color='#ffaa4c'),
    name="Net Revenue"
))
fig_line.update_layout(
    paper_bgcolor='#23263b', 
    plot_bgcolor='#23263b', 
    font_color="#fff",
    showlegend=False, 
    height=320,
    margin=dict(l=22, r=22, t=30, b=0)
)

specialty_group = data.groupby('Specialty')['NetRevenue'].sum().sort_values(ascending=False).head(5)
fig_pie = go.Figure()
fig_pie.add_trace(go.Pie(
    labels=specialty_group.index,
    values=specialty_group.values,
    hole=.7,
    marker=dict(colors=['#ffaa4c', '#00c9a7', '#5b6dde', '#f66d9b', '#ae66fa'])
))
fig_pie.update_layout(paper_bgcolor='#23263b', font_color="#fff", showlegend=True, height=230, margin=dict(t=15, b=0))

st.plotly_chart(fig_line, use_container_width=True)
col1, col2, col3, col4 = st.columns(4)
col1.metric("Variance to Target", f"${total_rev:,.0f}")
col2.metric("Clients add men", f"${unique_clients:,.0f}")
col3.metric("Monthly landing", f"${monthly_rev:,.0f}")
col4.metric("Receivables\nPending out", f"${receivables:,.0f}")

c1, c2 = st.columns([1, 2])
with c1:
    st.markdown("#### Specialty")
    st.plotly_chart(fig_pie, use_container_width=True)
with c2:
    st.markdown("#### Monthly added")
    st.bar_chart(monthly_added)
    st.markdown("#### Latest 12 Entries")
    st.dataframe(
        data[['Date','ClientName','NetRevenue','Specialty']].sort_values('Date',ascending=False).head(12),
        use_container_width=True,
        hide_index=True
    )

# ---- Anomalies ---
anomalies = anomaly_detect(data)
if anomalies:
    st.error("Revenue Anomalies Detected (z-score >2):")
    st.dataframe(pd.DataFrame(anomalies))

# ---- Expandable: debug info
with st.expander("Show advanced column info"):
    st.write(data.head(1).T)
    st.write("Files attempted:", glob.glob(f"{HISTORICAL}/*") + glob.glob(f"{DAILY}/*"))
