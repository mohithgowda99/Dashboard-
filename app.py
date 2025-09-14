"""
Truemedix Diagnostics Lab Dashboard
Intuitive tab-based layout for better user experience
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
import json
import os
import calendar
import re
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION
# ==============================================================================

COLUMN_MAPPING = {
    'InvoiceDate': ['Invoice Date', 'Date', 'Transaction Date', 'Bill Date'],
    'Branch': ['Branch', 'Location', 'Centre', 'Lab'],
    'Salesperson': ['Salesperson', 'Sales Person', 'Executive', 'Agent', 'Marketing Person', 'Marketing Person(Organisation)'],
    'ClientName': ['ClientName', 'Client Name', 'Customer', 'Hospital', 'Patient'],
    'TestName': ['TestName', 'Test Name', 'Service', 'Investigation'],
    'NetRevenue': ['NetRevenue', 'Net Revenue', 'Net Amount', 'Final Amount', 'Collected'],
    'BilledAmount': ['BilledAmount', 'Billed Amount', 'Bill Amount', 'Gross Amount', 'Total'],
    'ClientAddedDate': ['ClientAddedDate', 'Client Added Date', 'Registration Date', 'Join Date']
}

CONFIG_FILE = 'truemedix_config.json'
DEFAULT_TARGET = 10000000

# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {'monthly_target': DEFAULT_TARGET}

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f)
    except:
        pass

def normalize_columns(df):
    column_map = {}
    for standard, variations in COLUMN_MAPPING.items():
        for col in df.columns:
            col_clean = col.lower().strip()
            all_options = [standard.lower()] + [v.lower() for v in variations]
            if col_clean in all_options:
                column_map[col] = standard
                break
    return df.rename(columns=column_map)

def clean_data(df):
    df = df.copy()
    
    # Remove summary rows
    summary_indicators = ['total', 'subtotal', 'grand total', 'sum', 'summary']
    for col in df.select_dtypes(include=['object']).columns:
        mask = ~df[col].astype(str).str.lower().str.contains('|'.join(summary_indicators), na=False)
        df = df[mask]

    # Clean numeric columns
    for col in ['NetRevenue', 'BilledAmount']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('₹', '').str.replace(',', '').str.replace('Rs.', '')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).abs()

    # Clean date columns
    for col in ['InvoiceDate', 'ClientAddedDate']:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Handle missing ClientAddedDate
    if 'ClientAddedDate' not in df.columns or df['ClientAddedDate'].isna().all():
        first_invoice = df.groupby('ClientName')['InvoiceDate'].min().reset_index()
        first_invoice.columns = ['ClientName', 'ClientAddedDate']
        df = df.merge(first_invoice, on='ClientName', how='left', suffixes=('', '_new'))
        if 'ClientAddedDate_new' in df.columns:
            df['ClientAddedDate'] = df['ClientAddedDate'].fillna(df['ClientAddedDate_new'])
            df = df.drop('ClientAddedDate_new', axis=1)

    # Remove invalid data
    df = df[df['NetRevenue'] > 0]
    df = df[df['InvoiceDate'].notna()]
    
    return df

@st.cache_data
def load_excel_file(file_content, filename):
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            try:
                df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
            except:
                df = pd.read_excel(io.BytesIO(file_content), engine='xlrd')
        
        df = normalize_columns(df)
        df = clean_data(df)
        return df, []
    except Exception as e:
        return None, [f"Error: {str(e)}"]

def calculate_target_metrics(df, monthly_target):
    if df.empty:
        return {
            'expected_revenue': 0, 'actual_revenue': 0, 'variance_amount': 0,
            'variance_percent': 0, 'monthly_projection': 0, 'completion_percent': 0,
            'days_elapsed': 0, 'days_in_month': 30
        }
    
    now = datetime.now()
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    days_elapsed = now.day
    daily_target = monthly_target / days_in_month
    expected_revenue = daily_target * days_elapsed

    current_month = now.replace(day=1)
    current_data = df[df['InvoiceDate'] >= current_month]
    actual_revenue = current_data['NetRevenue'].sum()

    variance_amount = actual_revenue - expected_revenue
    variance_percent = (variance_amount / expected_revenue * 100) if expected_revenue > 0 else 0
    daily_average = actual_revenue / days_elapsed if days_elapsed > 0 else 0
    monthly_projection = daily_average * days_in_month
    completion_percent = (actual_revenue / monthly_target * 100) if monthly_target > 0 else 0

    return {
        'expected_revenue': expected_revenue, 'actual_revenue': actual_revenue,
        'variance_amount': variance_amount, 'variance_percent': variance_percent,
        'monthly_projection': monthly_projection, 'completion_percent': completion_percent,
        'days_elapsed': days_elapsed, 'days_in_month': days_in_month
    }

def calculate_clients_added_mtd(df):
    if df.empty or 'ClientAddedDate' not in df.columns:
        return 0
    current_month = datetime.now().replace(day=1)
    current_month_clients = df[df['ClientAddedDate'] >= current_month]
    return current_month_clients['ClientName'].nunique()

# ==============================================================================
# MAIN APP
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Truemedix Analytics",
        page_icon="🧪",
        layout="wide"
    )
    
    # Header
    st.title("🧪 Truemedix Analytics Dashboard")
    st.markdown("**Simple, reliable revenue and client tracking for your diagnostics lab**")
    st.divider()
    
    # Initialize session state
    config = load_config()
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'monthly_target' not in st.session_state:
        st.session_state.monthly_target = config.get('monthly_target', DEFAULT_TARGET)

    # File upload at the top (always visible)
    st.subheader("📁 Upload Your Data")
    uploaded_file = st.file_uploader(
        "Choose Excel/CSV file",
        type=['xlsx', 'xls', 'csv'],
        help="Upload your lab billing data to get started"
    )
    
    if uploaded_file:
        file_content = uploaded_file.read()
        df, errors = load_excel_file(file_content, uploaded_file.name)
        
        if errors:
            for error in errors:
                st.error(error)
        
        if df is not None and len(df) > 0:
            st.session_state.df = df
            st.success(f"✅ Successfully loaded {len(df):,} records from {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")

    # Show content only if data is loaded
    if st.session_state.df is not None:
        df = st.session_state.df
        
        # Create intuitive tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Dashboard", 
            "🏆 Performance", 
            "💰 Collections", 
            "🎯 Targets", 
            "📋 Data"
        ])
        
        # Common filters (show above tabs)
        with st.expander("🔍 Filter Data", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                date_range = st.date_input(
                    "Date Range",
                    value=(df['InvoiceDate'].min().date(), df['InvoiceDate'].max().date()),
                    min_value=df['InvoiceDate'].min().date(),
                    max_value=df['InvoiceDate'].max().date()
                )
            
            with col2:
                branches = st.multiselect(
                    "Branch",
                    options=sorted(df['Branch'].unique()) if 'Branch' in df.columns else []
                )
            
            with col3:
                salespersons = st.multiselect(
                    "Salesperson", 
                    options=sorted(df['Salesperson'].unique()) if 'Salesperson' in df.columns else []
                )
            
            with col4:
                high_value_only = st.checkbox("High-value tests only (≥₹999)")

        # Apply filters
        filtered_df = df.copy()
        if len(date_range) == 2:
            filtered_df = filtered_df[
                (filtered_df['InvoiceDate'].dt.date >= date_range[0]) &
                (filtered_df['InvoiceDate'].dt.date <= date_range[1])
            ]
        if branches:
            filtered_df = filtered_df[filtered_df['Branch'].isin(branches)]
        if salespersons:
            filtered_df = filtered_df[filtered_df['Salesperson'].isin(salespersons)]
        if high_value_only:
            filtered_df = filtered_df[filtered_df['NetRevenue'] >= 999]

        # TAB 1: DASHBOARD
        with tab1:
            # 5-metric row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            total_revenue = filtered_df['NetRevenue'].sum()
            amount_collected = filtered_df['NetRevenue'].sum()
            amount_due = filtered_df['BilledAmount'].sum() - amount_collected
            total_tests = len(filtered_df)
            clients_added_mtd = calculate_clients_added_mtd(filtered_df)
            
            with col1:
                st.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
            with col2:
                st.metric("✅ Collected", f"₹{amount_collected:,.0f}")
            with col3:
                st.metric("⏳ Due", f"₹{amount_due:,.0f}")
            with col4:
                st.metric("🧪 Tests", f"{total_tests:,}")
            with col5:
                st.metric("👥 Clients Added (MTD)", f"{clients_added_mtd:,}")
            
            st.divider()
            
            # Target tracking
            target_metrics = calculate_target_metrics(filtered_df, st.session_state.monthly_target)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "🎯 Target by Today",
                    f"₹{target_metrics['expected_revenue']:,.0f}",
                    delta=f"₹{target_metrics['variance_amount']:+,.0f} ({target_metrics['variance_percent']:+.1f}%)"
                )
            with col2:
                projection_vs_target = target_metrics['monthly_projection'] - st.session_state.monthly_target
                st.metric(
                    "📈 Monthly Projection",
                    f"₹{target_metrics['monthly_projection']:,.0f}",
                    delta=f"₹{projection_vs_target:+,.0f}"
                )
            with col3:
                st.metric(
                    "🏁 Completion %",
                    f"{target_metrics['completion_percent']:.1f}%",
                    delta=f"Day {target_metrics['days_elapsed']}/{target_metrics['days_in_month']}"
                )
            
            st.divider()
            
            # Charts
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📈 Daily Revenue Trend")
                daily_revenue = filtered_df.groupby(filtered_df['InvoiceDate'].dt.date)['NetRevenue'].sum().reset_index()
                fig = px.line(daily_revenue, x='InvoiceDate', y='NetRevenue', markers=True)
                fig.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("👨‍💼 Top Performers")
                if 'Salesperson' in filtered_df.columns:
                    performance = filtered_df.groupby('Salesperson')['NetRevenue'].sum().reset_index()
                    performance = performance.sort_values('NetRevenue', ascending=True).tail(5)
                    fig = px.bar(performance, x='NetRevenue', y='Salesperson', orientation='h')
                    fig.update_layout(height=300, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

        # TAB 2: PERFORMANCE
        with tab2:
            st.subheader("🏆 Performance Analysis")
            
            analysis_type = st.radio("Show:", ["Top Performers", "Bottom Performers"], horizontal=True)
            num_performers = st.slider("Number to show:", 1, 10, 5)
            
            if 'Salesperson' in filtered_df.columns:
                performance = filtered_df.groupby('Salesperson').agg({
                    'NetRevenue': ['sum', 'count', 'mean'],
                    'ClientName': 'nunique'
                }).round(2)
                
                performance.columns = ['Total_Revenue', 'Total_Tests', 'Avg_Test_Value', 'Unique_Clients']
                
                if analysis_type == "Top Performers":
                    performance = performance.sort_values('Total_Revenue', ascending=False).head(num_performers)
                else:
                    performance = performance.sort_values('Total_Revenue', ascending=True).head(num_performers)
                
                performance = performance.reset_index()
                
                fig = px.bar(performance, x='Total_Revenue', y='Salesperson', orientation='h',
                           title=f"{analysis_type} by Revenue")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(performance, use_container_width=True)

        # TAB 3: COLLECTIONS
        with tab3:
            st.subheader("💰 Collections Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Billed", f"₹{filtered_df['BilledAmount'].sum():,.0f}")
                st.metric("Total Collected", f"₹{filtered_df['NetRevenue'].sum():,.0f}")
            with col2:
                collection_rate = (filtered_df['NetRevenue'].sum() / filtered_df['BilledAmount'].sum() * 100) if filtered_df['BilledAmount'].sum() > 0 else 0
                outstanding = filtered_df['BilledAmount'].sum() - filtered_df['NetRevenue'].sum()
                st.metric("Collection Rate", f"{collection_rate:.1f}%")
                st.metric("Outstanding", f"₹{outstanding:,.0f}")
            
            # Collection rate gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=collection_rate,
                title={'text': "Collection Rate %"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

        # TAB 4: TARGETS
        with tab4:
            st.subheader("🎯 Target Management")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                target = st.number_input(
                    f"Monthly Target (₹) - {calendar.month_name[datetime.now().month]}",
                    value=float(st.session_state.monthly_target),
                    step=100000.0,
                    format="%.0f"
                )
                st.session_state.monthly_target = target
            
            with col2:
                if st.button("💾 Save Target", type="primary"):
                    config['monthly_target'] = target
                    save_config(config)
                    st.success("Saved!")
            
            # Quick presets
            st.write("**Quick Presets:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if st.button("₹50L", use_container_width=True):
                    st.session_state.monthly_target = 5000000
                    st.rerun()
            with col2:
                if st.button("₹1Cr", use_container_width=True):
                    st.session_state.monthly_target = 10000000
                    st.rerun()
            with col3:
                if st.button("₹1.5Cr", use_container_width=True):
                    st.session_state.monthly_target = 15000000
                    st.rerun()
            with col4:
                if st.button("₹2Cr", use_container_width=True):
                    st.session_state.monthly_target = 20000000
                    st.rerun()
            
            st.divider()
            
            # Target progress
            target_metrics = calculate_target_metrics(filtered_df, st.session_state.monthly_target)
            progress = min(100, target_metrics['completion_percent'])
            st.progress(progress / 100, text=f"Monthly Progress: {progress:.1f}%")
            
            # Target summary metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Achieved So Far", f"₹{target_metrics['actual_revenue']:,.0f}")
            with col2:
                st.metric("Remaining Target", f"₹{st.session_state.monthly_target - target_metrics['actual_revenue']:,.0f}")
            with col3:
                st.metric("Days Remaining", f"{target_metrics['days_in_month'] - target_metrics['days_elapsed']}")

        # TAB 5: DATA
        with tab5:
            st.subheader("📋 Raw Data")
            st.write(f"Showing {len(filtered_df):,} records")
            
            # Column selection
            available_columns = list(filtered_df.columns)
            selected_columns = st.multiselect(
                "Select columns:",
                available_columns,
                default=available_columns[:6]
            )
            
            if selected_columns:
                display_df = filtered_df[selected_columns]
                st.dataframe(display_df, use_container_width=True)
                
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv,
                    file_name="truemedix_data.csv",
                    mime="text/csv",
                    type="primary"
                )
    
    else:
        st.info("👆 **Please upload your Excel/CSV file above to start analyzing your lab data**")
        
        # Show expected format
        st.subheader("📋 Expected Data Format")
        sample_data = {
            'InvoiceDate': ['2024-01-15', '2024-01-16'],
            'Branch': ['Main Branch', 'North Branch'],
            'Salesperson': ['John Doe', 'Jane Smith'],
            'ClientName': ['Hospital A', 'Clinic B'],
            'NetRevenue': [1000, 1500],
            'BilledAmount': [1200, 1800]
        }
        st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

if __name__ == "__main__":
    main()
