# Truemedix Analytics Dashboard — Premium Nav + Error-Free
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io, json, os, logging, calendar, re
from pathlib import Path

# -------- Page config --------
st.set_page_config(
    page_title="Truemedix Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------- Logging --------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------- Constants / Config --------
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

# -------- Config helpers --------
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

# -------- Robust file processing --------
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
        primary = candidates[0]
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

def clean_excel_data(df):
    notes = []
    original_rows = len(df)

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

    # Numeric cleaning
    for col in [c for c in NUMERIC_COLUMNS if c in df.columns]:
        try:
            s = df[col].astype(str).str.replace("₹","",regex=False).str.replace("Rs.","",regex=False)
            s = s.str.replace(",","",regex=False).str.replace("$","",regex=False).str.strip()
            s = s.replace(["nan","null","none","","N/A","n/a"], np.nan)
            s = pd.to_numeric(s, errors="coerce").fillna(0)
            s = s.abs()
            df[col] = s
        except Exception as e:
            notes.append(f"Numeric clean error {col}: {e}")

    # Date parsing
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

    # Critical filters
    if "InvoiceDate" in df.columns:
        df = df[df["InvoiceDate"].notna()]
    rev_cols = [c for c in ["NetRevenue","BilledAmount","TestMRP"] if c in df.columns]
    if rev_cols:
        df = df[df[rev_cols].gt(0).any(axis=1)]

    # Finalize
    if len(df) == 0:
        notes.append("ERROR: No valid rows after cleaning")
    else:
        notes.append(f"Cleaning done: {len(df)} rows")

    return df, notes

# Hash-safe, list-safe cached loader
@st.cache_data(hash_funcs={bytes: lambda b: hash(b)})
def load_and_process_file(file_bytes, filename):
    warnings = []
    errors = []
    df = None
    ext = Path(filename).suffix.lower()

    try:
        # Read file
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
                    errors.append("Could not parse CSV with common encodings")
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
                            if sheet != xl.sheet_names[0]:
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
            return None
