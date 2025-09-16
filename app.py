“””
Diagnostics Laboratory Business Metrics Dashboard
A production-ready interactive dashboard for analyzing laboratory business performance.
“””

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from datetime import datetime, timedelta
import json
import os
from typing import Dict, List, Optional, Tuple
import logging

# Configure logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(**name**)

# ==============================================================================

# CONFIGURATION - Edit these for field names and metric definitions

# ==============================================================================

REQUIRED_COLUMNS = [
‘InvoiceDate’, ‘Branch’, ‘Salesperson’, ‘ClientName’, ‘TestName’,
‘Specialty’, ‘TestMRP’, ‘BilledAmount’, ‘NetRevenue’, ‘InvoiceID’, ‘ClientAddedDate’
]

DATE_COLUMNS = [‘InvoiceDate’, ‘ClientAddedDate’]
NUMERIC_COLUMNS = [‘TestMRP’, ‘BilledAmount’, ‘NetRevenue’]

# Metric definitions (used in tooltips and documentation)

METRIC_DEFINITIONS = {
‘MTD Revenue’: ‘Total Net Revenue for the selected month to date’,
‘Revenue Variance’: ‘Difference between actual revenue and target for the month’,
‘Clients Added MTD’: ‘Number of new clients onboarded in the current month’,
‘High Value Tests’: ‘Count of tests with Net Revenue ≥ ₹999’,
‘Avg Ticket Size’: ‘Average Net Revenue per invoice’
}

# Default revenue targets by month (can be overridden in UI)

DEFAULT_TARGETS = {
1: 1000000, 2: 950000, 3: 1100000, 4: 1050000, 5: 1150000, 6: 1200000,
7: 1300000, 8: 1250000, 9: 1100000, 10: 1400000, 11: 1350000, 12: 1500000
}

CONFIG_FILE = ‘dashboard_config.json’

# ==============================================================================

# UTILITY FUNCTIONS

# ==============================================================================

@st.cache_data
def load_config() -> Dict:
“”“Load configuration from file.”””
if os.path.exists(CONFIG_FILE):
try:
with open(CONFIG_FILE, ‘r’) as f:
return json.load(f)
except Exception as e:
logger.warning(f”Could not load config: {e}”)
return {‘last_file_path’: None, ‘targets’: DEFAULT_TARGETS.copy()}

def save_config(config: Dict):
“”“Save configuration to file.”””
try:
with open(CONFIG_FILE, ‘w’) as f:
json.dump(config, f, indent=2)
except Exception as e:
logger.warning(f”Could not save config: {e}”)

@st.cache_data
def validate_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
“”“Validate the uploaded data against requirements.”””
errors = []

```
# Check required columns
missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
if missing_cols:
    errors.append(f"Missing required columns: {', '.join(missing_cols)}")

if not errors:  # Only proceed if we have all required columns
    # Check date parsing success rate
    for col in DATE_COLUMNS:
        if col in df.columns:
            try:
                parsed_dates = pd.to_datetime(df[col], errors='coerce')
                success_rate = (1 - parsed_dates.isna().sum() / len(df)) * 100
                if success_rate < 99:
                    errors.append(f"{col} date parsing success rate: {success_rate:.1f}% (expected ≥99%)")
            except Exception as e:
                errors.append(f"Error parsing {col}: {str(e)}")
    
    # Check numeric columns
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            try:
                numeric_vals = pd.to_numeric(df[col], errors='coerce')
                if col == 'NetRevenue' and (numeric_vals < 0).any():
                    errors.append(f"{col} contains negative values")
            except Exception as e:
                errors.append(f"Error validating {col}: {str(e)}")
    
    # Check InvoiceID uniqueness
    if 'InvoiceID' in df.columns:
        duplicate_rate = df['InvoiceID'].duplicated().sum() / len(df) * 100
        if duplicate_rate > 1:  # Allow some duplicates for multi-line invoices
            errors.append(f"InvoiceID duplicate rate: {duplicate_rate:.1f}% (expected ≤1%)")

return len(errors) == 0, errors
```

@st.cache_data
def load_and_validate_file(file_content: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
“”“Load and validate uploaded file.”””
try:
if filename.endswith(’.csv’):
df = pd.read_csv(io.BytesIO(file_content))
elif filename.endswith((’.xlsx’, ‘.xls’)):
df = pd.read_excel(io.BytesIO(file_content))
else:
return None, [“Unsupported file format. Please upload CSV or Excel files.”]

```
    logger.info(f"Loaded {len(df)} rows from {filename}")
    
    # Validate data
    is_valid, errors = validate_data(df)
    
    if is_valid:
        return df, []
    else:
        return df, errors
        
except Exception as e:
    logger.error(f"Error loading file {filename}: {e}")
    return None, [f"Error loading file: {str(e)}"]
```

@st.cache_data
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
“”“Apply all data transformations.”””
df = df.copy()

```
# Parse dates
for col in DATE_COLUMNS:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors='coerce')

# Convert numeric columns
for col in NUMERIC_COLUMNS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Create derived fields
df['Month'] = df['InvoiceDate'].dt.to_period('M').dt.start_time
df['ClientAddedMonth'] = df['ClientAddedDate'].dt.to_period('M').dt.start_time
df['Tests_999_Flag'] = df['NetRevenue'] >= 999
df['SpecialtyTest'] = (~df['Specialty'].isna()) & (df['TestMRP'] >= 999)

# Add day of month for sparklines
df['InvoiceDay'] = df['InvoiceDate'].dt.day

logger.info(f"Transformed data: {len(df)} rows, {len(df.columns)} columns")
return df
```

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
“”“Apply all selected filters to the dataframe.”””
filtered_df = df.copy()

```
# Date range filter
if filters.get('date_range'):
    start_date, end_date = filters['date_range']
    filtered_df = filtered_df[
        (filtered_df['InvoiceDate'] >= pd.Timestamp(start_date)) &
        (filtered_df['InvoiceDate'] <= pd.Timestamp(end_date))
    ]

# Month filter
if filters.get('months'):
    filtered_df = filtered_df[filtered_df['Month'].isin(filters['months'])]

# Categorical filters
for col in ['Branch', 'Salesperson', 'ClientName', 'Specialty']:
    if filters.get(col.lower()):
        filtered_df = filtered_df[filtered_df[col].isin(filters[col.lower()])]

# High value tests filter
if filters.get('high_value_only'):
    filtered_df = filtered_df[filtered_df['Tests_999_Flag']]

# Text search filter
if filters.get('test_search'):
    search_term = filters['test_search'].lower()
    filtered_df = filtered_df[
        filtered_df['TestName'].str.lower().str.contains(search_term, na=False)
    ]

return filtered_df
```

def create_kpi_cards(df: pd.DataFrame, selected_month: Optional[pd.Timestamp], targets: Dict) -> Dict:
“”“Calculate KPI metrics.”””
kpis = {}

```
# MTD Revenue
if selected_month:
    mtd_df = df[df['Month'] == selected_month]
    current_day = datetime.now().day
    mtd_df = mtd_df[mtd_df['InvoiceDay'] <= current_day]
    kpis['mtd_revenue'] = mtd_df['NetRevenue'].sum()
    
    # Monthly target and variance
    month_num = selected_month.month
    target = targets.get(month_num, 0)
    actual = df[df['Month'] == selected_month]['NetRevenue'].sum()
    kpis['target'] = target
    kpis['variance'] = actual - target
    kpis['variance_pct'] = (kpis['variance'] / target * 100) if target > 0 else 0
else:
    kpis['mtd_revenue'] = df['NetRevenue'].sum()
    kpis['target'] = 0
    kpis['variance'] = 0
    kpis['variance_pct'] = 0

# Clients added MTD
if selected_month:
    clients_added = df[df['ClientAddedMonth'] == selected_month]['ClientName'].nunique()
    kpis['clients_added_mtd'] = clients_added
else:
    kpis['clients_added_mtd'] = df['ClientName'].nunique()

# High value tests
kpis['high_value_tests'] = df['Tests_999_Flag'].sum()

return kpis
```

def create_sparkline(df: pd.DataFrame, selected_month: Optional[pd.Timestamp]) -> go.Figure:
“”“Create daily revenue sparkline for selected month.”””
if selected_month is None:
return go.Figure()

```
month_df = df[df['Month'] == selected_month]
daily_revenue = month_df.groupby('InvoiceDay')['NetRevenue'].sum().reset_index()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=daily_revenue['InvoiceDay'],
    y=daily_revenue['NetRevenue'],
    mode='lines+markers',
    line=dict(width=2, color='#1f77b4'),
    marker=dict(size=4),
    name='Daily Revenue'
))

fig.update_layout(
    height=100,
    margin=dict(l=0, r=0, t=0, b=0),
    showlegend=False,
    xaxis=dict(showgrid=False, showticklabels=False),
    yaxis=dict(showgrid=False, showticklabels=False),
    plot_bgcolor='rgba(0,0,0,0)'
)

return fig
```

def create_monthly_performance_chart(df: pd.DataFrame, targets: Dict) -> go.Figure:
“”“Create monthly revenue vs target chart.”””
monthly_revenue = df.groupby(‘Month’)[‘NetRevenue’].sum().reset_index()
monthly_revenue = monthly_revenue.sort_values(‘Month’).tail(12)  # Last 12 months

```
monthly_revenue['Target'] = monthly_revenue['Month'].dt.month.map(targets)
monthly_revenue['Variance'] = monthly_revenue['NetRevenue'] - monthly_revenue['Target']

fig = go.Figure()

# Revenue bars
fig.add_trace(go.Bar(
    x=monthly_revenue['Month'],
    y=monthly_revenue['NetRevenue'],
    name='Actual Revenue',
    marker_color='lightblue'
))

# Target line
fig.add_trace(go.Scatter(
    x=monthly_revenue['Month'],
    y=monthly_revenue['Target'],
    mode='lines+markers',
    name='Target',
    line=dict(color='red', width=2),
    marker=dict(size=8)
))

# Add variance annotations
for idx, row in monthly_revenue.iterrows():
    variance_color = 'green' if row['Variance'] >= 0 else 'red'
    fig.add_annotation(
        x=row['Month'],
        y=max(row['NetRevenue'], row['Target']) + 50000,
        text=f"₹{row['Variance']:,.0f}",
        showarrow=False,
        font=dict(color=variance_color, size=10)
    )

fig.update_layout(
    title='Monthly Revenue vs Target (Last 12 Months)',
    xaxis_title='Month',
    yaxis_title='Revenue (₹)',
    height=400,
    hovermode='x unified'
)

return fig
```

def create_clients_added_chart(df: pd.DataFrame) -> go.Figure:
“”“Create clients added per month chart.”””
clients_by_month = df.groupby(‘ClientAddedMonth’)[‘ClientName’].nunique().reset_index()
clients_by_month = clients_by_month.sort_values(‘ClientAddedMonth’).tail(12)

```
fig = px.bar(
    clients_by_month,
    x='ClientAddedMonth',
    y='ClientName',
    title='New Clients Added Per Month',
    labels={'ClientName': 'Number of New Clients', 'ClientAddedMonth': 'Month'}
)

fig.update_layout(height=400)
return fig
```

def create_specialty_tests_chart(df: pd.DataFrame) -> go.Figure:
“”“Create high-value specialty tests chart.”””
high_value_df = df[df[‘Tests_999_Flag’]]
specialty_counts = high_value_df.groupby(‘Specialty’).size().reset_index()
specialty_counts.columns = [‘Specialty’, ‘Count’]
specialty_counts = specialty_counts.sort_values(‘Count’, ascending=True).tail(20)

```
fig = px.bar(
    specialty_counts,
    x='Count',
    y='Specialty',
    orientation='h',
    title='High-Value Tests (≥₹999) by Specialty',
    labels={'Count': 'Number of Tests'}
)

fig.update_layout(height=600)
return fig
```

def create_dimension_analysis_chart(df: pd.DataFrame, dimension: str) -> go.Figure:
“”“Create revenue analysis by selected dimension.”””
dim_analysis = df.groupby(dimension).agg({
‘InvoiceID’: ‘nunique’,
‘TestName’: ‘count’,
‘NetRevenue’: [‘sum’, ‘mean’]
}).round(2)

```
dim_analysis.columns = ['Invoices', 'Tests', 'Total_Revenue', 'Avg_Ticket']
dim_analysis = dim_analysis.sort_values('Total_Revenue', ascending=False).head(20)

fig = px.bar(
    dim_analysis.reset_index(),
    x=dimension,
    y='Total_Revenue',
    title=f'Revenue by {dimension} (Top 20)',
    labels={'Total_Revenue': 'Total Revenue (₹)'}
)

fig.update_layout(height=500, xaxis_tickangle=45)
return fig
```

def create_dimension_table(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
“”“Create detailed table for selected dimension.”””
table_data = df.groupby(dimension).agg({
‘InvoiceID’: ‘nunique’,
‘TestName’: ‘count’,
‘NetRevenue’: [‘sum’, ‘mean’]
}).round(2)

```
table_data.columns = ['Invoices', 'Tests', 'NetRevenue', 'AvgTicket']
table_data = table_data.sort_values('NetRevenue', ascending=False).head(50)

# Format currency columns
table_data['NetRevenue'] = table_data['NetRevenue'].apply(lambda x: f"₹{x:,.2f}")
table_data['AvgTicket'] = table_data['AvgTicket'].apply(lambda x: f"₹{x:,.2f}")

return table_data.reset_index()
```

def export_to_csv(df: pd.DataFrame, filename: str) -> str:
“”“Export dataframe to CSV and return download link.”””
csv = df.to_csv(index=False)
b64 = base64.b64encode(csv.encode()).decode()
href = f’<a href="data:file/csv;base64,{b64}" download="{filename}">Download {filename}</a>’
return href

# ==============================================================================

# STREAMLIT APP

# ==============================================================================

def main():
st.set_page_config(
page_title=“Lab Analytics Dashboard”,
page_icon=“🧪”,
layout=“wide”,
initial_sidebar_state=“expanded”
)

```
st.title("🧪 Diagnostics Laboratory Analytics Dashboard")

# Load configuration
config = load_config()

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'targets' not in st.session_state:
    st.session_state.targets = config.get('targets', DEFAULT_TARGETS.copy())

# Sidebar for file upload and filters
with st.sidebar:
    st.header("📁 Data Upload")
    
    uploaded_file = st.file_uploader(
        "Upload Excel/CSV file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your lab data file with all required columns"
    )
    
    if uploaded_file is not None:
        # Load and validate file
        file_content = uploaded_file.read()
        df, errors = load_and_validate_file(file_content, uploaded_file.name)
        
        if df is not None and not errors:
            st.session_state.df = transform_data(df)
            config['last_file_path'] = uploaded_file.name
            save_config(config)
            st.success(f"✅ Loaded {len(st.session_state.df)} records")
        elif errors:
            st.error("❌ Data validation failed:")
            for error in errors:
                st.error(f"• {error}")
            if df is not None:
                st.session_state.df = transform_data(df)  # Load anyway for debugging
    
    # Show current dataset info
    if st.session_state.df is not None:
        st.info(f"📊 Current dataset: {len(st.session_state.df):,} rows")
        st.info(f"📅 Date range: {st.session_state.df['InvoiceDate'].min().date()} to {st.session_state.df['InvoiceDate'].max().date()}")
    
    st.divider()
    
    # Filters section
    if st.session_state.df is not None:
        st.header("🔍 Filters")
        
        df = st.session_state.df
        
        # Date range filter
        min_date = df['InvoiceDate'].min().date()
        max_date = df['InvoiceDate'].max().date()
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
        
        # Month selection
        available_months = sorted(df['Month'].dt.strftime('%Y-%m').unique())
        selected_month_str = st.selectbox(
            "Focus Month",
            options=[None] + available_months,
            format_func=lambda x: "All Months" if x is None else x
        )
        
        selected_month = pd.Timestamp(selected_month_str) if selected_month_str else None
        
        # Multi-select filters
        branch_options = sorted(df['Branch'].dropna().unique())
        selected_branches = st.multiselect("Branch", options=branch_options)
        
        salesperson_options = sorted(df['Salesperson'].dropna().unique()) 
        selected_salespersons = st.multiselect("Salesperson", options=salesperson_options)
        
        client_options = sorted(df['ClientName'].dropna().unique())
        selected_clients = st.multiselect("Client Name", options=client_options[:100])  # Limit for performance
        
        specialty_options = sorted(df['Specialty'].dropna().unique())
        selected_specialties = st.multiselect("Specialty", options=specialty_options)
        
        # Toggle and search filters
        high_value_only = st.checkbox("Only tests ≥ ₹999", value=False)
        test_search = st.text_input("Search Test Name", placeholder="Enter test name...")
        
        # Compile filters
        filters = {
            'date_range': date_range if len(date_range) == 2 else None,
            'months': [selected_month] if selected_month else None,
            'branch': selected_branches if selected_branches else None,
            'salesperson': selected_salespersons if selected_salespersons else None,
            'clientname': selected_clients if selected_clients else None,
            'specialty': selected_specialties if selected_specialties else None,
            'high_value_only': high_value_only,
            'test_search': test_search if test_search else None
        }
        
        st.divider()
        
        # Revenue targets
        st.header("🎯 Revenue Targets")
        st.write("Set monthly revenue targets:")
        
        targets = st.session_state.targets.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            for month in range(1, 7):
                month_name = pd.Timestamp(f"2024-{month:02d}-01").strftime('%B')
                targets[month] = st.number_input(
                    f"{month_name}",
                    value=targets[month],
                    step=10000,
                    format="%d"
                )
        
        with col2:
            for month in range(7, 13):
                month_name = pd.Timestamp(f"2024-{month:02d}-01").strftime('%B')
                targets[month] = st.number_input(
                    f"{month_name}",
                    value=targets[month],
                    step=10000,
                    format="%d"
                )
        
        st.session_state.targets = targets
        
        if st.button("💾 Save Targets"):
            config['targets'] = targets
            save_config(config)
            st.success("Targets saved!")

# Main content area
if st.session_state.df is None:
    st.info("👆 Please upload a dataset to begin analysis")
    
    # Show sample data structure
    st.subheader("📋 Expected Data Structure")
    sample_data = {
        'InvoiceDate': ['2024-01-15', '2024-01-16'],
        'Branch': ['Main Branch', 'North Branch'],
        'Salesperson': ['John Doe', 'Jane Smith'], 
        'ClientName': ['Hospital A', 'Clinic B'],
        'TestName': ['Blood Test', 'X-Ray'],
        'Specialty': ['Hematology', 'Radiology'],
        'TestMRP': [1500, 2000],
        'BilledAmount': [1200, 1800],
        'NetRevenue': [1000, 1500],
        'InvoiceID': ['INV001', 'INV002'],
        'ClientAddedDate': ['2023-12-01', '2024-01-10']
    }
    st.dataframe(pd.DataFrame(sample_data))
    return

# Apply filters to data
df = st.session_state.df
filtered_df = apply_filters(df, filters)

st.write(f"📊 Showing {len(filtered_df):,} records (filtered from {len(df):,} total)")

# Calculate KPIs
kpis = create_kpi_cards(filtered_df, selected_month, st.session_state.targets)

# KPI Cards Row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="💰 MTD Revenue",
        value=f"₹{kpis['mtd_revenue']:,.0f}",
        help=METRIC_DEFINITIONS['MTD Revenue']
    )

with col2:
    variance_color = "normal" if kpis['variance'] >= 0 else "inverse"
    st.metric(
        label="📊 Revenue Variance",
        value=f"₹{kpis['variance']:,.0f}",
        delta=f"{kpis['variance_pct']:+.1f}%",
        help=METRIC_DEFINITIONS['Revenue Variance']
    )

with col3:
    st.metric(
        label="👥 Clients Added MTD",
        value=f"{kpis['clients_added_mtd']:,}",
        help=METRIC_DEFINITIONS['Clients Added MTD']
    )

with col4:
    st.metric(
        label="⭐ High Value Tests",
        value=f"{kpis['high_value_tests']:,}",
        help=METRIC_DEFINITIONS['High Value Tests']
    )

# Sparkline for MTD
if selected_month:
    st.subheader("📈 Daily Revenue Trend (Selected Month)")
    sparkline_fig = create_sparkline(filtered_df, selected_month)
    st.plotly_chart(sparkline_fig, use_container_width=True)

# Main Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Monthly Revenue vs Target")
    monthly_fig = create_monthly_performance_chart(filtered_df, st.session_state.targets)
    st.plotly_chart(monthly_fig, use_container_width=True)

with col2:
    st.subheader("👥 New Clients Per Month") 
    clients_fig = create_clients_added_chart(filtered_df)
    st.plotly_chart(clients_fig, use_container_width=True)

# Specialty analysis
st.subheader("⭐ High-Value Tests by Specialty")
specialty_fig = create_specialty_tests_chart(filtered_df)
st.plotly_chart(specialty_fig, use_container_width=True)

# Top high-value tests table
if st.checkbox("📋 Show Top High-Value Tests Details"):
    high_value_tests = filtered_df[filtered_df['Tests_999_Flag']]
    if not high_value_tests.empty:
        test_summary = high_value_tests.groupby(['TestName', 'Specialty']).agg({
            'NetRevenue': ['count', 'sum', 'mean']
        }).round(2)
        test_summary.columns = ['Count', 'TotalRevenue', 'AvgRevenue']
        test_summary = test_summary.sort_values('TotalRevenue', ascending=False).head(20)
        st.dataframe(test_summary)

# Dimensional analysis
st.subheader("📈 Revenue Analysis by Dimension")

dimension_choice = st.radio(
    "Choose dimension:",
    options=['Salesperson', 'Branch', 'ClientName'],
    horizontal=True
)

col1, col2 = st.columns([2, 1])

with col1:
    dim_fig = create_dimension_analysis_chart(filtered_df, dimension_choice)
    st.plotly_chart(dim_fig, use_container_width=True)

with col2:
    st.write(f"**Top 20 {dimension_choice} Details**")
    dim_table = create_dimension_table(filtered_df, dimension_choice)
    st.dataframe(dim_table, height=500)

# Export section
st.subheader("📥 Export Data")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 Export Filtered Data"):
        csv_link = export_to_csv(filtered_df, "filtered_lab_data.csv")
        st.markdown(csv_link, unsafe_allow_html=True)

with col2:
    if st.button("📈 Export KPI Summary"):
        kpi_df = pd.DataFrame([{
            'Metric': k.replace('_', ' ').title(),
            'Value': v
        } for k, v in kpis.items()])
        csv_link = export_to_csv(kpi_df, "kpi_summary.csv")
        st.markdown(csv_link, unsafe_allow_html=True)

with col3:
    st.write("**Chart Export**: Right-click charts → Save as PNG")
```

if **name** == “**main**”:
main()