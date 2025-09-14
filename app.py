# Truemedix Analytics Dashboard — Fixed Try-Except Structure
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io, json, os, logging, calendar, re
from pathlib import Path

# Page config
st.set_page_config(
    page_title="Truemedix Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
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
    "monthly_target": 10000000.0,
    "targets": {1:10000000,2:9500000,3:11000000,4:10500000,5:11500000,6:12000000,
                7:13000000,8:12500000,9:11000000,10:14000000,11:13500000,12:15000000},
    "last_file_path": None
}

# Config helpers
@st.cache_data
def load_config():
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

def save_config(cfg):
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save config: {e}")

# File processing functions
def detect_encoding(file_bytes):
    try:
        import chardet
        result = chardet.detect(file_bytes)
        return result.get("encoding") or "utf-8"
    except Exception:
        return "utf-8"

def normalize_column_names(df):
    df_norm = df.copy()
    original_cols = df_norm.columns.tolist()
    col_map = {}
    
    for _, candidates in COLUMN_MAPPING.items():
        primary = candidates
        for orig in original_cols:
            o_clean = re.sub(r"\s+", " ", str(orig).strip().lower())
            for cand in candidates:
                c_clean = re.sub(r"\s+", " ", cand.strip().lower())
                if o_clean == c_clean or c_clean in o_clean:
                    col_map[orig] = primary
                    break
            if orig in col_map:
                break
    
    if col_map:
        df_norm.rename(columns=col_map, inplace=True)
        logger.info(f"Column mapping applied: {col_map}")
    return df_norm

def clean_excel_data(df):
    notes = []
    original_rows = len(df)
    
    # Remove empty rows
    df = df.dropna(how="all")
    
    # Remove summary rows
    summary_tokens = ["total","subtotal","grand total","sum","summary","aggregate","overall","combined","consolidated","final","end"]
    mask_keep = pd.Series(True, index=df.index)
    for col in df.select_dtypes(include=["object"]).columns:
        col_mask = ~df[col].astype(str).str.lower().str.contains("|".join(summary_tokens), regex=True, na=False)
        mask_keep &= col_mask
    df = df[mask_keep]
    
    if len(df) < original_rows:
        notes.append("Removed summary/total rows")
    
    # Clean numeric columns
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        try:
            s = df[col].astype(str).str.replace("₹","",regex=False).str.replace("Rs.","",regex=False)
            s = s.str.replace(",","",regex=False).str.replace("$","",regex=False).str.strip()
            s = s.replace(["nan","null","none","","N/A","n/a"], np.nan)
            s = pd.to_numeric(s, errors="coerce").fillna(0).abs()
            df[col] = s
        except Exception as e:
            notes.append(f"Numeric clean error {col}: {e}")
    
    # Parse dates
    for col in [c for c in DATE_COLUMNS if c in df.columns]:
        try:
            d = pd.to_datetime(df[col].astype(str), errors="coerce", dayfirst=True, infer_datetime_format=True)
            if col == "InvoiceDate":
                d = d.fillna(pd.Timestamp.now())
            elif col == "ClientAddedDate":
                d = d.fillna(pd.to_datetime(df.get("InvoiceDate"), errors="coerce"))
            df[col] = d
        except Exception as e:
            notes.append(f"Date parse error {col}: {e}")
    
    # Filter critical data
    if "InvoiceDate" in df.columns:
        df = df[df["InvoiceDate"].notna()]
    rev_cols = [c for c in ["NetRevenue","BilledAmount","TestMRP"] if c in df.columns]
    if rev_cols:
        df = df[df[rev_cols].gt(0).any(axis=1)]
    
    if len(df) == 0:
        notes.append("ERROR: No valid rows after cleaning")
    else:
        notes.append(f"Cleaning done: {len(df)} rows")
    
    return df, notes

# Fixed cached loader with proper try-except
@st.cache_data(hash_funcs={bytes: lambda b: hash(b)})
def load_and_process_file(file_bytes, filename):
    warnings = []
    errors = []
    df = None
    ext = Path(filename).suffix.lower()
    
    try:
        # Read file based on extension
        if ext == ".csv":
            enc = detect_encoding(file_bytes)
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), encoding=enc)
            except Exception:
                for fallback in ["utf-8","latin1","cp1252"]:
                    try:
                        df = pd.read_csv(io.BytesIO(file_bytes), encoding=fallback)
                        warnings.append(f"Used {fallback} encoding for CSV")
                        break
                    except Exception:
                        continue
                if df is None:
                    errors.append("Could not parse CSV")
                    return None, tuple(warnings), tuple(errors)
        
        elif ext in [".xlsx",".xls"]:
            try:
                xl = pd.ExcelFile(io.BytesIO(file_bytes))
                df = None
                for sheet in xl.sheet_names:
                    try:
                        temp = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
                        if len(temp) > 0 and len(temp.columns) > 5:
                            df = temp
                            if sheet != xl.sheet_names:
                                warnings.append(f"Using sheet '{sheet}'")
                            break
                    except Exception as e:
                        warnings.append(f"Skipping sheet '{sheet}': {e}")
                if df is None:
                    df = pd.read_excel(io.BytesIO(file_bytes), sheet_name=0)
            except Exception as e:
                errors.append(f"Error reading Excel: {e}")
                return None, tuple(warnings), tuple(errors)
        
        else:
            errors.append(f"Unsupported file type: {ext}")
            return None, tuple(warnings), tuple(errors)
        
        if df is None or len(df) == 0:
            errors.append("No rows found in file")
            return None, tuple(warnings), tuple(errors)
        
        # Process the data
        df = normalize_column_names(df)
        df, cleaning_notes = clean_excel_data(df)
        warnings.extend(cleaning_notes)
        
        # Validate columns
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            errors.append(f"Missing required columns: {', '.join(missing)}")
        
        return df, tuple(warnings), tuple(errors)
        
    except Exception as e:
        logger.exception("File processing error")
        errors.append(f"Unexpected error: {e}")
        return None, tuple(warnings), tuple(errors)

# Transform and filter functions
@st.cache_data
def transform_data(df):
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

def apply_filters(df, filters):
    out = df.copy()
    try:
        if filters.get("date_range"):
            start_date, end_date = filters["date_range"]
            out = out[(out["InvoiceDate"] >= pd.Timestamp(start_date)) &
                      (out["InvoiceDate"] <= pd.Timestamp(end_date))]
        
        for col_ui, key in [("Branch","branch"),("Salesperson","salesperson"),("ClientName","clientname"),("Specialty","specialty")]:
            if key in filters and filters[key] and col_ui in out.columns:
                out = out[out[col_ui].isin(filters[key])]
        
        if filters.get("high_value_only") and "Tests_999_Flag" in out.columns:
            out = out[out["Tests_999_Flag"]]
        
        if filters.get("test_search") and "TestName" in out.columns:
            term = str(filters["test_search"]).lower().strip()
            if term:
                out = out[out["TestName"].astype(str).str.lower().str.contains(term, na=False)]
    except Exception as e:
        logger.warning(f"Filter error: {e}")
    
    return out

# Target tracking
def compute_target_metrics(df, monthly_target):
    if df is None or df.empty:
        return {
            "daily_target": 0, "days_in_month": 30, "days_elapsed": 0, "days_remaining": 30,
            "expected_revenue": 0, "actual_revenue": 0, "variance_amount": 0, "variance_percent": 0,
            "monthly_projection": 0, "completion_percent": 0, "daily_average": 0,
            "remaining_target": monthly_target, "required_daily_run_rate": 0
        }
    
    today = datetime.now()
    y, m, d = today.year, today.month, today.day
    days_in_month = calendar.monthrange(y, m)[1]
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
    
    return {
        "daily_target": daily_target, "days_in_month": days_in_month, "days_elapsed": days_elapsed, "days_remaining": days_remaining,
        "expected_revenue": expected, "actual_revenue": actual, "variance_amount": variance_amount, "variance_percent": variance_percent,
        "monthly_projection": monthly_projection, "completion_percent": completion_percent, "daily_average": daily_avg,
        "remaining_target": remaining_target, "required_daily_run_rate": required_daily_run_rate
    }

# Initialize session
config = load_config()
if "df" not in st.session_state:
    st.session_state.df = None
if "warnings" not in st.session_state:
    st.session_state.warnings = []
if "monthly_target" not in st.session_state:
    mon = datetime.now().month
    st.session_state.monthly_target = float(config.get("targets", {}).get(mon, config.get("monthly_target", 10000000.0)))

# Premium sidebar navigation
with st.sidebar:
    st.markdown("### 📊 Truemedix Navigation")
    nav = st.radio(
        "",
        ["🔄 Upload & Status", "📈 Overview", "👥 Performance", "💰 Collections", "🎯 Targets", "📋 Data"],
        index=0
    )

# Header
st.title("Truemedix Analytics Dashboard")
st.caption("Minimal, fast, and reliable revenue insights with robust Excel handling and intelligent target tracking.")

# Navigation content
if nav == "🔄 Upload & Status":
    st.subheader("Data Upload")
    upl = st.file_uploader("Upload Excel/CSV", type=["csv","xlsx","xls"])
    if upl is not None:
        with st.spinner("Processing file..."):
            file_bytes = upl.read()
            df_raw, notes, errs = load_and_process_file(file_bytes, upl.name)
            
            if errs:
                st.error("File processing errors:")
                for e in errs:
                    st.error(f"• {e}")
            
            if notes:
                with st.expander("Processing details", expanded=len(errs) == 0):
                    for n in notes:
                        if str(n).startswith("ERROR"):
                            st.error(n)
                        else:
                            st.info(n)
                st.session_state.warnings = list(notes)
            
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
        df_info = st.session_state.df
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Records", f"{len(df_info):,}")
        with colB:
            if "InvoiceDate" in df_info.columns:
                min_date = df_info["InvoiceDate"].min().date()
                max_date = df_info["InvoiceDate"].max().date()
                st.metric("Date Range", f"{min_date} → {max_date}")
        with colC:
            if "NetRevenue" in df_info.columns:
                total_rev = df_info["NetRevenue"].sum()
                st.metric("Total Revenue", f"₹{total_rev:,.0f}")

# Filter UI helper
def filters_ui(df):
    with st.expander("🔍 Filters", expanded=True):
        cols = st.columns(4)
        date_range = None
        
        if "InvoiceDate" in df.columns:
            min_d = df["InvoiceDate"].min().date()
            max_d = df["InvoiceDate"].max().date()
            with cols:
                date_range = st.date_input("Date Range", value=(min_d, max_d), min_value=min_d, max_value=max_d)
        
        f = {"date_range": date_range if isinstance(date_range, tuple) and len(date_range) == 2 else None}
        
        with cols[1]:
            if "Branch" in df.columns:
                f["branch"] = st.multiselect("Branch", sorted(df["Branch"].dropna().astype(str).unique().tolist()))
        
        with cols[22]:
            if "Salesperson" in df.columns:
                f["salesperson"] = st.multiselect("Salesperson", sorted(df["Salesperson"].dropna().astype(str).unique().tolist()))
        
        with cols[23]:
            if "Specialty" in df.columns:
                f["specialty"] = st.multiselect("Specialty", sorted(df["Specialty"].dropna().astype(str).unique().tolist()))
        
        cols2 = st.columns(2)
        with cols2:
            f["high_value_only"] = st.checkbox("Only tests ≥ ₹999", value=False)
        with cols2[1]:
            f["test_search"] = st.text_input("Search Test Name", placeholder="Enter test name...")
    
    return f

# Overview section
elif nav == "📈 Overview" and st.session_state.df is not None:
    df_view = apply_filters(st.session_state.df, filters_ui(st.session_state.df))
    
    # Metrics
    m1, m2, m3, m4 = st.columns(4, gap="medium")
    total_revenue = float(df_view["NetRevenue"].sum()) if "NetRevenue" in df_view.columns else 0.0
    total_billed = float(df_view["BilledAmount"].sum()) if "BilledAmount" in df_view.columns else 0.0
    total_tests = len(df_view)
    total_due = max(0.0, total_billed - total_revenue) if total_billed and total_revenue is not None else 0.0
    
    with m1: 
        st.metric("Revenue", f"₹{total_revenue:,.0f}")
    with m2: 
        st.metric("Collected", f"₹{total_revenue:,.0f}")
    with m3: 
        st.metric("Due", f"₹{total_due:,.0f}")
    with m4: 
        st.metric("Tests", f"{total_tests:,}")
    
    # Target metrics
    t = compute_target_metrics(df_view, float(st.session_state.monthly_target))
    tm1, tm2, tm3 = st.columns(3, gap="medium")
    
    with tm1: 
        target_val = f"₹{t['expected_revenue']:,.0f}"
        variance_amt = t['variance_amount']
        variance_pct = t['variance_percent']
        target_delta = f"₹{variance_amt:,.0f} ({variance_pct:+.1f}%)"
        st.metric("Target by Today", target_val, delta=target_delta)
    
    with tm2: 
        proj_val = f"₹{t['monthly_projection']:,.0f}"
        monthly_target_val = float(st.session_state.monthly_target)
        proj_delta_amt = t['monthly_projection'] - monthly_target_val
        proj_delta = f"₹{proj_delta_amt:,.0f}"
        st.metric("Monthly Projection", proj_val, delta=proj_delta)
    
    with tm3: 
        comp_val = f"{t['completion_percent']:.1f}%"
        comp_delta = f"Day {t['days_elapsed']} of {t['days_in_month']}"
        st.metric("Target Completion", comp_val, delta=comp_delta)
    
    st.divider()
    
    # Daily revenue chart
    if "InvoiceDate" in df_view.columns and "NetRevenue" in df_view.columns:
        st.subheader("📊 Daily Revenue")
        daily = df_view.groupby(df_view["InvoiceDate"].dt.date)["NetRevenue"].sum().reset_index()
        daily.columns = ["Date","Revenue"]
        fig_daily = px.line(daily, x="Date", y="Revenue", markers=True)
        fig_daily.update_layout(height=380, showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig_daily, use_container_width=True)

# Performance section
elif nav == "👥 Performance" and st.session_state.df is not None:
    df_view = apply_filters(st.session_state.df, filters_ui(st.session_state.df))
    c1, c2 = st.columns(2, gap="medium")
    
    with c1:
        if "Salesperson" in df_view.columns and "NetRevenue" in df_view.columns:
            st.subheader("By Salesperson")
            rank = df_view.groupby("Salesperson")["NetRevenue"].sum().sort_values(ascending=True)
            fig_sp = px.bar(x=rank.values, y=rank.index, orientation="h")
            fig_sp.update_layout(height=380, showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_sp, use_container_width=True)
    
    with c2:
        if "Branch" in df_view.columns and "NetRevenue" in df_view.columns:
            st.subheader("By Branch")
            rankb = df_view.groupby("Branch")["NetRevenue"].sum().sort_values(ascending=True)
            fig_b = px.bar(x=rankb.values, y=rankb.index, orientation="h")
            fig_b.update_layout(height=380, showlegend=False, margin=dict(l=10,r=10,t=30,b=10))
            st.plotly_chart(fig_b, use_container_width=True)

# Collections section
elif nav == "💰 Collections" and st.session_state.df is not None:
    df_view = apply_filters(st.session_state.df, filters_ui(st.session_state.df))
    st.subheader("Collection Rate Analysis")
    
    total_gross = float(df_view["BilledAmount"].sum()) if "BilledAmount" in df_view.columns else 0.0
    total_net = float(df_view["NetRevenue"].sum()) if "NetRevenue" in df_view.columns else 0.0
    rate = (total_net / total_gross * 100.0) if total_gross > 0 else 0.0
    
    fig_g = go.Figure(go.Indicator(
        mode="gauge+number",
        value=rate,
        title={"text": "Collection Rate (%)"},
        gauge={
            "axis": {"range": [0,100]},
            "bar": {"color": "darkblue"},
            "steps": [
                {"range": [0,50], "color": "lightgray"},
                {"range": [50,80], "color": "yellow"},
                {"range": [80,100], "color": "green"}
            ],
            "threshold": {"line": {"color": "red", "width": 4}, "thickness": 0.75, "value": 90}
        }
    ))
    fig_g.update_layout(height=380, margin=dict(l=10,r=10,t=30,b=10))
    st.plotly_chart(fig_g, use_container_width=True)

# Targets section
elif nav == "🎯 Targets":
    st.subheader("Monthly Target Configuration")
    current_month = datetime.now().month
    
    val = st.number_input(
        f"Target (₹) - {calendar.month_name[current_month]}",
        value=float(st.session_state.monthly_target or 10000000.0),
        min_value=0.0, 
        step=100000.0, 
        format="%.0f"
    )
    st.session_state.monthly_target = float(val)
    
    # Quick preset buttons
    pc1, pc2, pc3, pc4 = st.columns(4, gap="small")
    
    with pc1:
        if st.button("₹50L"): 
            st.session_state.monthly_target = 5_000_000.0
            st.rerun()
    with pc2:
        if st.button("₹1Cr"): 
            st.session_state.monthly_target = 10_000_000.0
            st.rerun()
    with pc3:
        if st.button("₹1.5Cr"): 
            st.session_state.monthly_target = 15_000_000.0
            st.rerun()
    with pc4:
        if st.button("₹2Cr"): 
            st.session_state.monthly_target = 20_000_000.0
            st.rerun()
    
    if st.button("💾 Save Target"):
        config["monthly_target"] = float(st.session_state.monthly_target)
        config["targets"][current_month] = float(st.session_state.monthly_target)
        save_config(config)
        st.success("✅ Target saved successfully!")

# Data section
elif nav == "📋 Data" and st.session_state.df is not None:
    df_view = apply_filters(st.session_state.df, filters_ui(st.session_state.df))
    st.subheader("Detailed Data View")
    
    display_cols = [c for c in ["InvoiceDate","NetRevenue","BilledAmount","Branch","Salesperson","ClientName","Specialty","TestName"] if c in df_view.columns]
    
    if display_cols:
        sel_cols = st.multiselect("Select Columns to Display", display_cols, default=display_cols)
        if sel_cols:
            tbl = df_view[sel_cols].sort_values(display_cols, ascending=False)
            st.dataframe(tbl, use_container_width=True)
            
            csv = tbl.to_csv(index=False).encode("utf-8")
            dt_str = datetime.now().strftime('%Y%m%d')
            st.download_button("📥 Download CSV", data=csv, file_name=f"truemedix_data_{dt_str}.csv", mime="text/csv")
    else:
        st.info("No displayable columns found. Upload a dataset with required fields.")

# Show upload prompt if no data
else:
    st.info("👆 Please upload a file using 'Upload & Status' to begin analyzing your data.")
