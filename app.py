"""
Truemedix Diagnostics Lab Dashboard - Production Ready
Enhanced backend with professional UI design
"""

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
import calendar
import re
import warnings
from typing import Dict, List, Any, Optional, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ==============================================================================
# ENHANCED CONFIGURATION - TRUEMEDIX SPECIFIC
# ==============================================================================

# Comprehensive column mapping for all possible variations in your files
COLUMN_VARIATIONS = {
    'date': [
        'Date', 'Invoice Date', 'Bill Date', 'Transaction Date', 'Entry Date',
        'Test Date', 'Report Date', 'Collection Date', 'Booking Date', 'InvoiceDate',
        'date', 'invoice_date', 'bill_date', 'transaction_date',
        'DATE', 'INVOICE_DATE', 'BILL_DATE', 'TRANSACTION_DATE'
    ],
    'client': [
        'Client_Name', 'Client Name', 'Customer', 'Hospital', 'Organization',
        'Client', 'Customer Name', 'Hospital Name', 'Clinic', 'Clinic Name', 'ClientName',
        'client_name', 'customer_name', 'hospital_name', 'organization',
        'CLIENT_NAME', 'CUSTOMER_NAME', 'HOSPITAL_NAME', 'ORGANIZATION',
        'Patient', 'Patient Name', 'Referrer', 'Referring Doctor'
    ],
    'amount': [
        'Amount', 'Net Amount', 'Final Amount', 'Revenue', 'Net Revenue', 'NetRevenue',
        'Total Amount', 'Bill Amount', 'Invoice Amount', 'Gross Amount', 'BilledAmount',
        'amount', 'net_amount', 'final_amount', 'revenue', 'total_amount',
        'AMOUNT', 'NET_AMOUNT', 'FINAL_AMOUNT', 'REVENUE', 'TOTAL_AMOUNT',
        'Value', 'Price', 'Cost', 'Fee', 'Charge', 'Total'
    ],
    'specialty': [
        'Specialty', 'Test Name', 'Service', 'Investigation', 'Test Type', 'TestName',
        'Test', 'Service Name', 'Investigation Name', 'Department',
        'specialty', 'test_name', 'service', 'investigation', 'test_type',
        'SPECIALTY', 'TEST_NAME', 'SERVICE', 'INVESTIGATION', 'TEST_TYPE'
    ],
    'zone': [
        'Zone', 'Territory', 'Region', 'Area', 'Location', 'Branch',
        'City', 'State', 'District', 'Sector', 'Division',
        'zone', 'territory', 'region', 'area', 'location', 'branch',
        'ZONE', 'TERRITORY', 'REGION', 'AREA', 'LOCATION', 'BRANCH'
    ],
    'salesperson': [
        'Salesperson', 'Sales Person', 'Executive', 'Agent', 'Marketing Person',
        'Sales Executive', 'Sales Rep', 'Representative', 'Marketing Executive',
        'salesperson', 'sales_person', 'executive', 'agent',
        'SALESPERSON', 'SALES_PERSON', 'EXECUTIVE', 'AGENT'
    ],
    'payment_status': [
        'Payment_Status', 'Payment Status', 'Collection Status', 'Status',
        'Payment', 'Collection', 'Paid', 'Outstanding', 'Due Status',
        'payment_status', 'collection_status', 'payment', 'collection',
        'PAYMENT_STATUS', 'COLLECTION_STATUS', 'PAYMENT', 'COLLECTION'
    ]
}

CONFIG_FILE = 'truemedix_config.json'
DEFAULT_TARGET = 10000000  # ₹1 crore
SPECIALTY_TEST_THRESHOLD = 999  # Minimum amount for specialty tests

# Default revenue targets by month
DEFAULT_TARGETS = {
    1: 1000000, 2: 950000, 3: 1100000, 4: 1050000, 5: 1150000, 6: 1200000,
    7: 1300000, 8: 1250000, 9: 1100000, 10: 1400000, 11: 1350000, 12: 1500000
}

# Metric definitions
METRIC_DEFINITIONS = {
    'MTD Revenue': 'Total Net Revenue for the selected month to date',
    'Revenue Variance': 'Difference between actual revenue and target for the month',
    'Clients Added MTD': 'Number of new clients onboarded in the current month',
    'High Value Tests': 'Count of tests with Revenue ≥ ₹999',
    'Avg Ticket Size': 'Average Revenue per transaction'
}

# ==============================================================================
# ENHANCED INTELLIGENT ANALYSIS ENGINE
# ==============================================================================

class TruemedixAnalyzer:
    """Enhanced intelligent analyzer for your specific billing data"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.original_columns = list(df.columns)
        self.analysis_cache = {}
        self.column_mapping = {}
        
    def detect_column_structure(self) -> Dict[str, str]:
        """Enhanced detection and mapping of your file's column structure"""
        column_mapping = {}
        df_columns = list(self.df.columns)
        
        # Enhanced matching with fuzzy logic
        for standard_key, variations in COLUMN_VARIATIONS.items():
            matched_column = None
            
            # Exact match first
            for col in df_columns:
                if col in variations:
                    matched_column = col
                    break
            
            # Fuzzy matching if no exact match
            if not matched_column:
                for col in df_columns:
                    col_clean = str(col).strip().lower()
                    for variation in variations:
                        variation_clean = variation.strip().lower()
                        # Check if variation is contained in column name or vice versa
                        if (variation_clean in col_clean or col_clean in variation_clean) and len(col_clean) > 2:
                            matched_column = col
                            break
                    if matched_column:
                        break
            
            # Keyword-based matching for amount columns
            if standard_key == 'amount' and not matched_column:
                for col in df_columns:
                    col_lower = str(col).lower()
                    if any(keyword in col_lower for keyword in ['amt', 'val', 'price', 'cost', 'fee', 'charge', 'total', 'sum']):
                        matched_column = col
                        break
            
            if matched_column:
                column_mapping[matched_column] = standard_key
        
        self.column_mapping = column_mapping
        return column_mapping

    def clean_and_prepare_data(self) -> pd.DataFrame:
        """Enhanced data cleaning specifically for your file format"""
        df = self.df.copy()
        
        # Detect column mapping
        column_mapping = self.detect_column_structure()
        
        if not column_mapping:
            # Fallback: try to find amount-like columns by data type
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64'] or pd.api.types.is_numeric_dtype(df[col]):
                    # Check if it looks like monetary values
                    sample_values = df[col].dropna().head(10)
                    if len(sample_values) > 0 and sample_values.max() > 100:  # Likely monetary
                        column_mapping[col] = 'amount'
                        break
            
            # Try to find client-like columns
            for col in df.columns:
                if df[col].dtype == 'object' and df[col].nunique() > 5:
                    column_mapping[col] = 'client'
                    break
        
        # Create reverse mapping for renaming
        rename_mapping = {orig: mapped for orig, mapped in column_mapping.items()}
        mapped_df = df.rename(columns=rename_mapping)
        
        # Clean date columns
        date_columns = [col for col in mapped_df.columns if 'date' in col.lower()]
        for col in date_columns:
            mapped_df[col] = pd.to_datetime(mapped_df[col], errors='coerce')
        
        # Enhanced amount column cleaning
        amount_columns = [col for col in mapped_df.columns if any(x in col.lower() for x in ['amount', 'revenue', 'bill', 'total', 'value', 'price'])]
        for col in amount_columns:
            if mapped_df[col].dtype == 'object':
                # Remove currency symbols and convert to numeric
                mapped_df[col] = mapped_df[col].astype(str).str.replace('₹', '', regex=False)
                mapped_df[col] = mapped_df[col].str.replace('Rs.', '', regex=False)
                mapped_df[col] = mapped_df[col].str.replace('Rs', '', regex=False)
                mapped_df[col] = mapped_df[col].str.replace(',', '', regex=False)
                mapped_df[col] = mapped_df[col].str.replace('INR', '', regex=False)
                mapped_df[col] = mapped_df[col].str.strip()
                mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce').fillna(0)
        
        # Handle specialty tests and create derived fields
        if 'amount' in mapped_df.columns:
            mapped_df['is_specialty_test'] = mapped_df['amount'] >= SPECIALTY_TEST_THRESHOLD
            mapped_df['Tests_999_Flag'] = mapped_df['amount'] >= SPECIALTY_TEST_THRESHOLD
        
        # Create month fields
        if 'date' in mapped_df.columns:
            mapped_df['Month'] = mapped_df['date'].dt.to_period('M').dt.start_time
            mapped_df['InvoiceDay'] = mapped_df['date'].dt.day
        
        return mapped_df

# ==============================================================================
# ENHANCED UTILITY FUNCTIONS
# ==============================================================================

@st.cache_data
def load_config() -> Dict:
    """Load configuration from file."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
    return {'last_file_path': None, 'targets': DEFAULT_TARGETS.copy()}

def save_config(config: Dict):
    """Save configuration to file."""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save config: {e}")

@st.cache_data
def load_and_validate_file(file_content: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """Enhanced loading and processing of your specific Excel/CSV file format"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
        elif filename.endswith(('.xlsx', '.xls')):
            try:
                df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
            except Exception:
                try:
                    df = pd.read_excel(io.BytesIO(file_content), engine='xlrd')
                except Exception:
                    df = pd.read_excel(io.BytesIO(file_content))
        else:
            return None, ["Unsupported file format. Please upload CSV or Excel files."]

        # Remove completely empty rows and columns
        df = df.dropna(how='all', axis=0)  # Remove empty rows
        df = df.dropna(how='all', axis=1)  # Remove empty columns
        
        # Clean column names
        df.columns = df.columns.astype(str).str.strip()
        
        if len(df) == 0:
            return None, ["Error: File appears to be empty after cleaning"]
        
        logger.info(f"Loaded {len(df)} rows from {filename}")
        return df, []
        
    except Exception as e:
        logger.error(f"Error loading file {filename}: {e}")
        return None, [f"Error loading file: {str(e)}"]

@st.cache_data
def transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all data transformations using Truemedix analyzer."""
    analyzer = TruemedixAnalyzer(df)
    transformed_df = analyzer.clean_and_prepare_data()
    
    # Additional transformations for dashboard compatibility
    # Map our column names to dashboard expected names
    column_map = {
        'date': 'InvoiceDate',
        'client': 'ClientName', 
        'amount': 'NetRevenue',
        'specialty': 'Specialty',
        'zone': 'Branch',
        'salesperson': 'Salesperson'
    }
    
    # Rename columns to match dashboard expectations
    for old_name, new_name in column_map.items():
        if old_name in transformed_df.columns and new_name not in transformed_df.columns:
            transformed_df = transformed_df.rename(columns={old_name: new_name})
    
    # Create missing fields with defaults if they don't exist
    if 'InvoiceDate' not in transformed_df.columns and 'date' in transformed_df.columns:
        transformed_df['InvoiceDate'] = transformed_df['date']
    
    if 'ClientName' not in transformed_df.columns and 'client' in transformed_df.columns:
        transformed_df['ClientName'] = transformed_df['client']
        
    if 'NetRevenue' not in transformed_df.columns and 'amount' in transformed_df.columns:
        transformed_df['NetRevenue'] = transformed_df['amount']
        
    if 'Branch' not in transformed_df.columns and 'zone' in transformed_df.columns:
        transformed_df['Branch'] = transformed_df['zone']
        
    if 'Specialty' not in transformed_df.columns and 'specialty' in transformed_df.columns:
        transformed_df['Specialty'] = transformed_df['specialty']
        
    if 'Salesperson' not in transformed_df.columns and 'salesperson' in transformed_df.columns:
        transformed_df['Salesperson'] = transformed_df['salesperson']
    
    # Fill missing required fields
    required_fields = ['TestName', 'ClientAddedDate', 'InvoiceID', 'TestMRP', 'BilledAmount']
    for field in required_fields:
        if field not in transformed_df.columns:
            if field == 'TestName':
                transformed_df[field] = transformed_df.get('Specialty', 'Unknown Test')
            elif field == 'ClientAddedDate':
                transformed_df[field] = transformed_df.get('InvoiceDate', pd.Timestamp.now())
            elif field == 'InvoiceID':
                transformed_df[field] = range(1, len(transformed_df) + 1)
            elif field in ['TestMRP', 'BilledAmount']:
                transformed_df[field] = transformed_df.get('NetRevenue', 0)
    
    # Create ClientAddedMonth
    if 'ClientAddedDate' in transformed_df.columns:
        transformed_df['ClientAddedMonth'] = pd.to_datetime(transformed_df['ClientAddedDate']).dt.to_period('M').dt.start_time
    
    logger.info(f"Transformed  {len(transformed_df)} rows, {len(transformed_df.columns)} columns")
    return transformed_df

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply all selected filters to the dataframe."""
    filtered_df = df.copy()

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
        filter_key = col.lower()
        if filters.get(filter_key):
            if col in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[col].isin(filters[filter_key])]

    # High value tests filter
    if filters.get('high_value_only'):
        filtered_df = filtered_df[filtered_df['Tests_999_Flag']]

    # Text search filter
    if filters.get('test_search'):
        search_term = filters['test_search'].lower()
        if 'TestName' in filtered_df.columns:
            filtered_df = filtered_df[
                filtered_df['TestName'].str.lower().str.contains(search_term, na=False)
            ]

    return filtered_df

def create_kpi_cards(df: pd.DataFrame, selected_month: Optional[pd.Timestamp], targets: Dict) -> Dict:
    """Calculate KPI metrics."""
    kpis = {}

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
    if selected_month and 'ClientAddedMonth' in df.columns:
        clients_added = df[df['ClientAddedMonth'] == selected_month]['ClientName'].nunique()
        kpis['clients_added_mtd'] = clients_added
    else:
        kpis['clients_added_mtd'] = df['ClientName'].nunique()

    # High value tests
    kpis['high_value_tests'] = df['Tests_999_Flag'].sum()

    return kpis

def create_sparkline(df: pd.DataFrame, selected_month: Optional[pd.Timestamp]) -> go.Figure:
    """Create daily revenue sparkline for selected month."""
    if selected_month is None:
        return go.Figure()

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

def create_monthly_performance_chart(df: pd.DataFrame, targets: Dict) -> go.Figure:
    """Create monthly revenue vs target chart."""
    monthly_revenue = df.groupby('Month')['NetRevenue'].sum().reset_index()
    monthly_revenue = monthly_revenue.sort_values('Month').tail(12)  # Last 12 months

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

def create_clients_added_chart(df: pd.DataFrame) -> go.Figure:
    """Create clients added per month chart."""
    if 'ClientAddedMonth' not in df.columns:
        # Fallback: use InvoiceDate
        df['ClientAddedMonth'] = df['Month']
    
    clients_by_month = df.groupby('ClientAddedMonth')['ClientName'].nunique().reset_index()
    clients_by_month = clients_by_month.sort_values('ClientAddedMonth').tail(12)

    fig = px.bar(
        clients_by_month,
        x='ClientAddedMonth',
        y='ClientName',
        title='New Clients Added Per Month',
        labels={'ClientName': 'Number of New Clients', 'ClientAddedMonth': 'Month'}
    )

    fig.update_layout(height=400)
    return fig

def create_specialty_tests_chart(df: pd.DataFrame) -> go.Figure:
    """Create high-value specialty tests chart."""
    high_value_df = df[df['Tests_999_Flag']]
    if 'Specialty' not in high_value_df.columns:
        high_value_df['Specialty'] = 'Unknown'
    
    specialty_counts = high_value_df.groupby('Specialty').size().reset_index()
    specialty_counts.columns = ['Specialty', 'Count']
    specialty_counts = specialty_counts.sort_values('Count', ascending=True).tail(20)

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

def create_dimension_analysis_chart(df: pd.DataFrame, dimension: str) -> go.Figure:
    """Create revenue analysis by selected dimension."""
    if dimension not in df.columns:
        # Fallback to available column
        available_dims = ['Branch', 'Salesperson', 'ClientName']
        dimension = next((dim for dim in available_dims if dim in df.columns), 'ClientName')
    
    dim_analysis = df.groupby(dimension).agg({
        'NetRevenue': ['sum', 'mean', 'count']
    }).round(2)

    dim_analysis.columns = ['Total_Revenue', 'Avg_Revenue', 'Count']
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

def create_dimension_table(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Create detailed table for selected dimension."""
    if dimension not in df.columns:
        available_dims = ['Branch', 'Salesperson', 'ClientName']
        dimension = next((dim for dim in available_dims if dim in df.columns), 'ClientName')
    
    table_data = df.groupby(dimension).agg({
        'NetRevenue': ['sum', 'mean', 'count']
    }).round(2)

    table_data.columns = ['NetRevenue', 'AvgTicket', 'Tests']
    table_data = table_data.sort_values('NetRevenue', ascending=False).head(50)

    # Format currency columns
    table_data['NetRevenue'] = table_data['NetRevenue'].apply(lambda x: f"₹{x:,.2f}")
    table_data['AvgTicket'] = table_data['AvgTicket'].apply(lambda x: f"₹{x:,.2f}")

    return table_data.reset_index()

def export_to_csv(df: pd.DataFrame, filename: str) -> str:
    """Export dataframe to CSV and return download link."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="file/csv;base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

# ==============================================================================
# STREAMLIT APP - PROFESSIONAL UI
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Truemedix Analytics Dashboard",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🧪 Truemedix Diagnostics Analytics Dashboard")

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
            help="Upload your Truemedix billing data file"
        )
        
        if uploaded_file is not None:
            # Load and validate file
            file_content = uploaded_file.read()
            df, errors = load_and_validate_file(file_content, uploaded_file.name)
            
            if df is not None:
                st.session_state.df = transform_data(df)
                config['last_file_path'] = uploaded_file.name
                save_config(config)
                st.success(f"✅ Loaded {len(st.session_state.df)} records")
                
                if errors:
                    st.info("ℹ️ Data validation notes:")
                    for error in errors:
                        st.info(f"• {error}")
            elif errors:
                st.error("❌ Error loading file:")
                for error in errors:
                    st.error(f"• {error}")
        
        # Show current dataset info
        if st.session_state.df is not None:
            df = st.session_state.df
            st.info(f"📊 Current dataset: {len(df):,} rows")
            if 'InvoiceDate' in df.columns:
                st.info(f"📅 Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")
        
        st.divider()
        
        # Filters section
        if st.session_state.df is not None:
            st.header("🔍 Filters")
            
            df = st.session_state.df
            
            # Date range filter
            if 'InvoiceDate' in df.columns:
                min_date = df['InvoiceDate'].min().date()
                max_date = df['InvoiceDate'].max().date()
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
            else:
                date_range = None
            
            # Month selection
            if 'Month' in df.columns:
                available_months = sorted(df['Month'].dt.strftime('%Y-%m').unique())
                selected_month_str = st.selectbox(
                    "Focus Month",
                    options=[None] + available_months,
                    format_func=lambda x: "All Months" if x is None else x
                )
                selected_month = pd.Timestamp(selected_month_str) if selected_month_str else None
            else:
                selected_month = None
            
            # Multi-select filters
            filters = {}
            
            if 'Branch' in df.columns:
                branch_options = sorted(df['Branch'].dropna().unique())
                selected_branches = st.multiselect("Branch", options=branch_options)
                filters['branch'] = selected_branches if selected_branches else None
            
            if 'Salesperson' in df.columns:
                salesperson_options = sorted(df['Salesperson'].dropna().unique())
                selected_salespersons = st.multiselect("Salesperson", options=salesperson_options)
                filters['salesperson'] = selected_salespersons if selected_salespersons else None
            
            if 'ClientName' in df.columns:
                client_options = sorted(df['ClientName'].dropna().unique())
                selected_clients = st.multiselect("Client Name", options=client_options[:100])
                filters['clientname'] = selected_clients if selected_clients else None
            
            if 'Specialty' in df.columns:
                specialty_options = sorted(df['Specialty'].dropna().unique())
                selected_specialties = st.multiselect("Specialty", options=specialty_options)
                filters['specialty'] = selected_specialties if selected_specialties else None
            
            # Toggle and search filters
            high_value_only = st.checkbox("Only tests ≥ ₹999", value=False)
            test_search = st.text_input("Search Test Name", placeholder="Enter test name...")
            
            # Compile all filters
            filters.update({
                'date_range': date_range if date_range and len(date_range) == 2 else None,
                'months': [selected_month] if selected_month else None,
                'high_value_only': high_value_only,
                'test_search': test_search if test_search else None
            })
            
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
        st.info("👆 Please upload your Truemedix dataset to begin analysis")
        
        # Show sample data structure
        st.subheader("📋 Expected Data Structure")
        st.write("Your file should contain columns that match any of these patterns:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Date columns:**")
            st.write("Date, Invoice Date, Bill Date, Transaction Date")
            st.write("**Amount columns:**")
            st.write("Amount, Net Revenue, Total Amount, Revenue")
            st.write("**Client columns:**")
            st.write("Client Name, Customer, Hospital, Organization")
        
        with col2:
            st.write("**Specialty columns:**")
            st.write("Specialty, Test Name, Service, Investigation")
            st.write("**Zone/Branch columns:**")
            st.write("Zone, Territory, Branch, Region, Location")
            st.write("**Salesperson columns:**")
            st.write("Salesperson, Sales Person, Executive, Agent")
        
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
            if 'TestName' in high_value_tests.columns and 'Specialty' in high_value_tests.columns:
                test_summary = high_value_tests.groupby(['TestName', 'Specialty']).agg({
                    'NetRevenue': ['count', 'sum', 'mean']
                }).round(2)
                test_summary.columns = ['Count', 'TotalRevenue', 'AvgRevenue']
                test_summary = test_summary.sort_values('TotalRevenue', ascending=False).head(20)
                st.dataframe(test_summary)
            else:
                st.info("TestName and Specialty columns not available for detailed breakdown")

    # Dimensional analysis
    st.subheader("📈 Revenue Analysis by Dimension")

    # Available dimensions based on data
    available_dimensions = [col for col in ['Salesperson', 'Branch', 'ClientName'] if col in filtered_df.columns]
    if available_dimensions:
        dimension_choice = st.radio(
            "Choose dimension:",
            options=available_dimensions,
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
    else:
        st.info("No dimensional data available for analysis")

    # Export section
    st.subheader("📥 Export Data")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("📊 Export Filtered Data"):
            csv_link = export_to_csv(filtered_df, "truemedix_filtered_data.csv")
            st.markdown(csv_link, unsafe_allow_html=True)

    with col2:
        if st.button("📈 Export KPI Summary"):
            kpi_df = pd.DataFrame([{
                'Metric': k.replace('_', ' ').title(),
                'Value': v
            } for k, v in kpis.items()])
            csv_link = export_to_csv(kpi_df, "truemedix_kpi_summary.csv")
            st.markdown(csv_link, unsafe_allow_html=True)

    with col3:
        st.write("**Chart Export**: Right-click charts → Save as PNG")


if __name__ == "__main__":
    main()
