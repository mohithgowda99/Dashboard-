# Enhanced Diagnostics Laboratory Analytics Dashboard
# Keeps original UI layout; adds robust file handling + target tracking

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import json
import os
import logging
import calendar
import re
from pathlib import Path

# -----------------------------------------------------------------------------
# Config & Logging
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Enhanced Lab Analytics Dashboard",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CONFIG_FILE = "enhanced_dashboard_config.json"

REQUIRED_COLUMNS = [
    "InvoiceDate", "Branch", "Salesperson", "ClientName", "TestName",
    "Specialty", "TestMRP", "BilledAmount", "NetRevenue", "InvoiceID", "ClientAddedDate"
]

COLUMN_MAPPING = {
    "invoice_date": ["InvoiceDate", "Invoice Date", "Date", "Transaction Date", "Bill Date"],
    "branch": ["Branch", "Location", "Centre", "Lab"],
    "salesperson": ["Salesperson", "Sales Person", "Executive", "Agent", "Rep"],
    "client_name": ["ClientName", "Client Name", "Customer", "Hospital", "Patient"],
    "test_name": ["TestName", "Test Name", "Service", "Investigation"],
    "specialty": ["Specialty", "Department", "Category", "Type"],
    "test_mrp": ["TestMRP", "Test MRP", "MRP", "List Price", "Rate"],
    "billed_amount": ["BilledAmount", "Billed Amount", "Bill Amount", "Gross Amount", "Total"],
    "net_revenue": ["NetRevenue", "Net Revenue", "Net Amount", "Final Amount", "Collected"],
    "invoice_id": ["InvoiceID", "Invoice ID", "Bill No", "Receipt No", "Transaction ID"],
    "client_added_date": ["ClientAddedDate", "Client Added Date", "Registration Date", "Join Date"]
}

DATE_COLUMNS = ["InvoiceDate", "ClientAddedDate"]
NUMERIC_COLUMNS = ["TestMRP", "BilledAmount", "NetRevenue"]

DEFAULT_CONFIG = {
    "monthly_target": 10000000.0,  # 1 Cr default
    "targets": {
        1: 10000000, 2: 9500000, 3: 11000000, 4: 10500000, 5: 11500000, 6: 12000000,
        7: 13000000, 8: 12500000, 9: 11000000, 10: 14000000, 11: 13500000, 12: 15000000
    },
    "last_file_path": None
}

# -----------------------------------------------------------------------------
# Helpers: Config
# -----------------------------------------------------------------------------
@st.cache_data
def load_config() -> dict:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                cfg = json.load(f)
            for k, v in DEFAULT_CONFIG.items():
                if k not in cfg:
                    cfg[k] = v
            return cfg
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

def save_config(cfg: dict):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save config: {e}")

# -----------------------------------------------------------------------------
# Robust File Processing
# -----------------------------------------------------------------------------
def detect_encoding(file_bytes: bytes) -> str:
    try:
        import chardet
        result = chardet.detect(file_bytes)
        return result.get("encoding") or "utf-8"
    except Exception:
        return "utf-8"

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df_norm = df.copy()
    original_cols = df_norm.columns.tolist()
    col_map = {}

    for std_key, candidates in COLUMN_MAPPING.items():
        primary = candidates
        for orig in original_cols:
            o_clean = re.sub(r"\s+", " ", str(orig).strip().lower())
            for cand in candidates:
                c_clean = re.sub(r"\s+", " ", cand.strip().lower())
                if o_clean == c_clean or c_clean in o_clean:
                    col_map[orig] = primary
                    break
            if orig in col_map:
                continue

    if col_map:
        df_norm.rename(columns=col_map, inplace=True)
        logger.info(f"Column mapping applied: {col_map}")
    return df_norm

def clean_excel_data(df: pd.DataFrame):
    notes = []
    original_rows = len(df)

    # Remove fully empty rows
    df = df.dropna(how="all")

    # Remove summary rows
    summary_tokens = [
        "total", "subtotal", "grand total", "sum", "summary", "aggregate",
        "overall", "combined", "consolidated", "final", "end"
    ]
    mask_keep = pd.Series(True, index=df.index)
    for col in df.select_dtypes(include=["object"]).columns:
        col_mask = ~df[col].astype(str).str.lower().str.contains("|".join(summary_tokens), regex=True, na=False)
        mask_keep &= col_mask
    df = df[mask_keep]
    if len(df) < original_rows:
        notes.append(f"Removed {original_rows - len(df)} summary/total rows")

    # Numeric cleaning
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        try:
            s = df[col].astype(str).str.replace("₹", "", regex=False)\
                                   .str.replace("Rs.", "", regex=False)\
                                   .str.replace(",", "", regex=False)\
                                   .str.replace("$", "", regex=False).str.strip()
            s = s.replace(["nan", "null", "none", "", "N/A", "n/a"], np.nan)
            s = pd.to_numeric(s, errors="coerce")
            nan_count = s.isna().sum()
            if nan_count > 0:
                notes.append(f"{nan_count} invalid values in {col} replaced with 0")
            s = s.fillna(0)
            if col in ["NetRevenue", "BilledAmount", "TestMRP"]:
                neg = (s < 0).sum()
                if neg > 0:
                    notes.append(f"{neg} negative values in {col} converted to absolute")
                s = s.abs()
            df[col] = s
        except Exception as e:
            notes.append(f"Error processing numeric column {col}: {e}")

    # Date parsing
    for col in [c for c in DATE_COLUMNS if c in df.columns]:
        try:
            d = pd.to_datetime(df[col].astype(str), errors="coerce", dayfirst=True, infer_datetime_format=True)
            invalid = d.isna().sum()
            if invalid > 0:
                notes.append(f"{invalid} invalid dates in {col}")
            if col == "InvoiceDate":
                d = d.fillna(pd.Timestamp.now())
                notes.append("Filled missing InvoiceDate with current date")
            elif col == "ClientAddedDate":
                if "InvoiceDate" in df.columns:
                    d = d.fillna(pd.to_datetime(df["InvoiceDate"], errors="coerce"))
            df[col] = d
        except Exception as e:
            notes.append(f"Error parsing date column {col}: {e}")

    # Critical filters
    before = len(df)
    if "InvoiceDate" in df.columns:
        df = df[df["InvoiceDate"].notna()]
    rev_cols = [c for c in ["NetRevenue", "BilledAmount", "TestMRP"] if c in df.columns]
    if rev_cols:
        df = df[df[rev_cols].gt(0).any(axis=1)]
    if len(df) < before:
        notes.append(f"Removed {before - len(df)} rows with missing date or zero revenue")

    final_rows = len(df)
    if final_rows == 0:
        notes.append("ERROR: No valid data after cleaning")
    else:
        notes.append(f"Cleaning complete: {final_rows}/{original_rows} rows retained ({final_rows/original_rows*100:.1f}%)")

    return df, notes

@st.cache_data
def load_and_process_file(file_bytes: bytes, filename: str):
    warnings = []
    errors = []
    df = None
    ext = Path(filename).suffix.lower()

    try:
        if ext == ".csv":
            enc = detect_encoding(file_bytes)
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            except Exception:
                for fallback in ["utf-8", "latin1", "cp1252"]:
                    try:
                        df = pd.read_csv(io.BytesIO(file_bytes), encoding=fallback)
                        warnings.append(f"Used {fallback} encoding for CSV")
                        break
                    except Exception:
                        continue
                if df is None:
                    raise ValueError("Could not parse CSV with common encodings")
        elif ext in [".xlsx", ".xls"]:
            try:
                xl = pd.ExcelFile(io.BytesIO(file_bytes))
                df = None
                for sheet in xl.sheet_names:
                    temp = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
                    if len(temp) > 0 and len(temp.columns) > 5:
                        df = temp
                        if sheet != xl.sheet_names:
                            warnings.append(f"Using sheet '{sheet}'")
                        break
                if df is None:
                    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
            except Exception as e:
                errors.append(f"Error reading Excel: {e}")
                return None, warnings, errors
        else:
            errors.append(f"Unsupported file type: {ext}")
            return None, warnings, errors

        if df is None or len(df) == 0:
            errors.append("No rows found in file")
            return None, warnings, errors

        # Normalize and clean
        df = normalize_column_names(df)
        df, cleaning_notes = clean_excel_data(df)
        warnings.extend(cleaning_notes)

        # Validate required columns
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {', '.join(missing)}")

        return df, warnings, errors

    except Exception as e:
        logger.exception("Unexpected file processing error")
        errors.append(f"Unexpected error: {e}")
        return None, warnings, errors

# -----------------------------------------------------------------------------
# Transforms & Filters
# -----------------------------------------------------------------------------
@st.cache_data
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    if "InvoiceDate" in d.columns:
        d["Month"] = d["InvoiceDate"].dt.to_period("M").dt.start_time
        d["InvoiceDay"] = d["InvoiceDate"].dt.day
        d["WeekOfYear"] = d["InvoiceDate"].dt.isocalendar().week
        d["DayOfWeek"] = d["InvoiceDate"].dt.day_name()
    if "ClientAddedDate" in d.columns:
        d["ClientAddedMonth"] = d["ClientAddedDate"].dt.to_period("M").dt.start_time
    if "NetRevenue" in d.columns:
        d["Tests_999_Flag"] = d["NetRevenue"] >= 999
    if "Specialty" in d.columns and "TestMRP" in d.columns:
        d["SpecialtyTest"] = (~d["Specialty"].isna()) & (d["TestMRP"] >= 999)
    return d

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    out = df.copy()
    try:
        if filters.get("date_range"):
            start_date, end_date = filters["date_range"]
            out = out[(out["InvoiceDate"] >= pd.Timestamp(start_date)) &
                      (out["InvoiceDate"] <= pd.Timestamp(end_date))]
        for col_ui, key in [("Branch", "branch"), ("Salesperson", "salesperson"),
                            ("ClientName", "clientname"), ("Specialty", "specialty")]:
            if key in filters and filters[key] and col_ui in out.columns:
                out = out[out[col_ui].isin(filters[key])]
        if filters.get("high_value_only") and "Tests_999_Flag" in out.columns:
            out = out[out["Tests_999_Flag"]]
        if filters.get("test_search") and "TestName" in out.columns:
            term = str(filters["test_search"]).lower().strip()
            if term:
                out = out[out["TestName"].astype(str).str.lower().str.contains(term, na=False)]
    except Exception as e:
        logger.warning(f"Filter application error: {e}")
    return out

# -----------------------------------------------------------------------------
# Target Tracking
# -----------------------------------------------------------------------------
def compute_target_metrics(df: pd.DataFrame, monthly_target: float) -> dict:
    if df is None or df.empty:
        return dict(
            daily_target=0, days_in_month=30, days_elapsed=0, days_remaining=30,
            expected_revenue=0, actual_revenue=0, variance_amount=0, variance_percent=0,
            monthly_projection=0, completion_percent=0, daily_average=0,
            remaining_target=monthly_target, required_daily_run_rate=0
        )

    today = datetime.now()
    y, m, d = today.year, today.month, today.day
    days_in_month = calendar.monthrange(y, m)[6]
    days_elapsed = min(d, days_in_month)
    days_remaining = max(0, days_in_month - d)

    daily_target = monthly_target / days_in_month

    month_start = pd.Timestamp(datetime(y, m, 1))
    mtd = df[(df["InvoiceDate"] >= month_start) & (df["InvoiceDate"] <= pd.Timestamp(today.date()))]
    actual = float(mtd["NetRevenue"].sum()) if "NetRevenue" in df.columns else 0.0

    expected = daily_target * days_elapsed
    variance_amount = actual - expected
    variance_percent = (variance_amount / expected * 100.0) if expected > 0 else 0.0
    daily_avg = (actual / days_elapsed) if days_elapsed > 0 else 0.0
    monthly_projection = daily_avg * days_in_month
    completion_percent = (days_elapsed / days_in_month) * 100.0
    remaining_target = max(0.0, monthly_target - actual)
    required_daily_run_rate = (remaining_target / days_remaining) if days_remaining > 0 else 0.0

    return dict(
        daily_target=daily_target,
        days_in_month=days_in_month,
        days_elapsed=days_elapsed,
        days_remaining=days_remaining,
        expected_revenue=expected,
        actual_revenue=actual,
        variance_amount=variance_amount,
        variance_percent=variance_percent,
        monthly_projection=monthly_projection,
        completion_percent=completion_percent,
        daily_average=daily_avg,
        remaining_target=remaining_target,
        required_daily_run_rate=required_daily_run_rate
    )

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("🧪 Enhanced Diagnostics Laboratory Analytics Dashboard")
st.markdown("_Production-ready with robust Excel handling and intelligent target tracking_")

config = load_config()

# Session init
if "df" not in st.session_state:
    st.session_state.df = None
if "warnings" not in st.session_state:
    st.session_state.warnings = []
if "monthly_target" not in st.session_state:
    mon = datetime.now().month
    st.session_state.monthly_target = float(config.get("targets", {}).get(mon, config.get("monthly_target", 10000000.0)))

with st.sidebar:
    st.header("📁 Data Upload & Processing")
    upl = st.file_uploader("Upload Excel/CSV", type=["csv", "xlsx", "xls"])
    if upl is not None:
        with st.spinner("Processing file..."):
            file_bytes = upl.read()
            df_raw, notes, errs = load_and_process_file(file_bytes, upl.name)
            if errs:
                st.error("❌ File processing errors:")
                for e in errs:
                    st.error(f"• {e}")
            if notes:
                with st.expander("📋 Processing Details", expanded=len(errs) == 0):
                    for n in notes:
                        if n.startswith("ERROR"):
                            st.error(n)
                        else:
                            st.info(n)
                st.session_state.warnings = notes
            if df_raw is not None and len(df_raw) > 0:
                st.session_state.df = transform_data(df_raw)
                config["last_file_path"] = upl.name
                save_config(config)
                if not errs:
                    st.success(f"✅ Loaded {len(st.session_state.df):,} records")
                else:
                    st.warning(f"⚠️ Loaded {len(st.session_state.df):,} records with issues")

    if st.session_state.df is not None:
        st.divider()
        st.subheader("📊 Current Dataset")
        df_info = st.session_state.df
        st.metric("Total Records", f"{len(df_info):,}")
        if "InvoiceDate" in df_info.columns:
            st.metric("Date Range", f"{df_info['InvoiceDate'].min().date()} to {df_info['InvoiceDate'].max().date()}")
        if "NetRevenue" in df_info.columns:
            st.metric("Total Revenue", f"₹{df_info['NetRevenue'].sum():,.0f}")

    st.divider()
    st.header("🎯 Target Configuration")
    current_month = datetime.now().month
    cur_default = float(config.get("targets", {}).get(current_month, config.get("monthly_target", 10000000.0)))
    val = st.number_input(
        f"Monthly Target (₹) - {calendar.month_name[current_month]}",
        value=float(st.session_state.monthly_target or cur_default),
        min_value=0.0,
        step=100000.0,
        format="%.0f"
    )
    st.session_state.monthly_target = float(val)

    cols_p = st.columns(4)
    with cols_p:
        if st.button("₹50L"):
            st.session_state.monthly_target = 5_000_000.0
            st.rerun()
    with cols_p[6]:
        if st.button("₹1Cr"):
            st.session_state.monthly_target = 10_000_000.0
            st.rerun()
    with cols_p[7]:
        if st.button("₹1.5Cr"):
            st.session_state.monthly_target = 15_000_000.0
            st.rerun()
    with cols_p[8]:
        if st.button("₹2Cr"):
            st.session_state.monthly_target = 20_000_000.0
            st.rerun()

    if st.button("💾 Save Target"):
        config["monthly_target"] = float(st.session_state.monthly_target)
        config["targets"][current_month] = float(st.session_state.monthly_target)
        save_config(config)
        st.success("Target saved!")

    st.divider()
    if st.session_state.df is not None:
        st.header("🔍 Filters")
        dff = st.session_state.df
        # Date range
        if "InvoiceDate" in dff.columns:
            min_d = dff["InvoiceDate"].min().date()
            max_d = dff["InvoiceDate"].max().date()
            date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        else:
            date_range = None

        filters = {"date_range": date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None}

        for ui_col, key in [("Branch", "branch"), ("Salesperson", "salesperson"),
                            ("ClientName", "clientname"), ("Specialty", "specialty")]:
            if ui_col in dff.columns:
                opts = sorted(dff[ui_col].dropna().astype(str).unique().tolist())
                sel = st.multiselect(ui_col, options=opts[:100] if len(opts) > 100 else opts)
                filters[key] = sel if sel else None

        filters["high_value_only"] = st.checkbox("Only tests ≥ ₹999", value=False)
        filters["test_search"] = st.text_input("Search Test Name", placeholder="Enter test name...")

# Guard: need data
if st.session_state.df is None:
    st.info("👆 Upload a file in the sidebar to begin.")
    st.stop()

# Apply filters for all below
df_all = st.session_state.df
df_view = apply_filters(df_all, locals().get("filters", {}))

# -----------------------------------------------------------------------------
# Original 4-column metrics (using filtered data)
# -----------------------------------------------------------------------------
col1, col2, col3, col4 = st.columns(4)
total_revenue = float(df_view["NetRevenue"].sum()) if "NetRevenue" in df_view.columns else 0.0
total_billed = float(df_view["BilledAmount"].sum()) if "BilledAmount" in df_view.columns else 0.0
total_tests = len(df_view)
total_due = max(0.0, total_billed - total_revenue) if total_billed and total_revenue is not None else 0.0

with col1:
    st.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
with col2:
    st.metric("✅ Amount Collected (Net)", f"₹{total_revenue:,.0f}")
with col3:
    st.metric("⏳ Amount Due (Billed - Net)", f"₹{total_due:,.0f}")
with col4:
    st.metric("🧪 Total Tests", f"{total_tests:,}")

# -----------------------------------------------------------------------------
# New Target Tracking 3-metric row (using filtered data)
# -----------------------------------------------------------------------------
t = compute_target_metrics(df_view, float(st.session_state.monthly_target))
tc1, tc2, tc3 = st.columns(3)
with tc1:
    st.metric("🎯 Target by Today", f"₹{t['expected_revenue']:,.0f}",
              delta=f"₹{t['variance_amount']:,.0f} ({t['variance_percent']:+.1f}%)")
with tc2:
    st.metric("📈 Monthly Projection", f"₹{t['monthly_projection']:,.0f}",
              delta=f"₹{t['monthly_projection'] - float(st.session_state.monthly_target):,.0f}")
with tc3:
    st.metric("📊 Target Completion", f"{t['completion_percent']:.1f}%",
              delta=f"Day {t['days_elapsed']} of {t['days_in_month']}")

st.divider()

# -----------------------------------------------------------------------------
# Original charts (kept structure; use filtered data)
# Daily Revenue Trend
# -----------------------------------------------------------------------------
if "InvoiceDate" in df_view.columns and "NetRevenue" in df_view.columns:
    st.subheader("📈 Daily Revenue Trend")
    daily = df_view.groupby(df_view["InvoiceDate"].dt.date)["NetRevenue"].sum().reset_index()
    daily.columns = ["Date", "Revenue"]
    fig_daily = px.line(daily, x="Date", y="Revenue", markers=True, title="Daily Revenue Trend")
    fig_daily.update_layout(height=400)
    st.plotly_chart(fig_daily, use_container_width=True)

# Two-column section: Marketing Person revenue + Collection Rate
c1, c2 = st.columns(2)
with c1:
    if "Salesperson" in df_view.columns and "NetRevenue" in df_view.columns:
        st.subheader("👥 Revenue by Salesperson")
        rank = df_view.groupby("Salesperson")["NetRevenue"].sum().sort_values(ascending=True)
        fig_sp = px.bar(x=rank.values, y=rank.index, orientation="h", title="Revenue by Salesperson")
        fig_sp.update_layout(height=400)
        st.plotly_chart(fig_sp, use_container_width=True)

with c2:
    st.subheader("💳 Collection Rate")
    total_gross = float(df_view["BilledAmount"].sum()) if "BilledAmount" in df_view.columns else 0.0
    total_net = float(df_view["NetRevenue"].sum()) if "NetRevenue" in df_view.columns else 0.0
    rate = (total_net / total_gross * 100.0) if total_gross > 0 else 0.0
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate,
        title={"text": "Collection Rate (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0, 50], "color": "lightgray"},
                {"range": [50, 80], "color": "yellow"},
                {"range": [80, 100], "color": "green"}
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90}
        }
    ))
    fig_g.update_layout(height=400)
    st.plotly_chart(fig_g, use_container_width=True)

# -----------------------------------------------------------------------------
# Original data table with download
# -----------------------------------------------------------------------------
st.subheader("📋 Detailed Data")
display_cols = []
for c in ["InvoiceDate", "NetRevenue", "BilledAmount", "Branch", "Salesperson", "ClientName", "Specialty", "TestName"]:
    if c in df_view.columns:
        display_cols.append(c)

if display_cols:
    default_cols = display_cols.copy()
    sel_cols = st.multiselect("Select columns to display:", display_cols, default=default_cols)
    if sel_cols:
        tbl = df_view[sel_cols].sort_values(display_cols, ascending=False)
        st.dataframe(tbl, use_container_width=True)
        csv = tbl.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download CSV", data=csv, file_name=f"lab_data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
else:
    st.info("No displayable columns found. Upload a dataset with required fields.")
