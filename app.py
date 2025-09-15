"""
Truemedix Diagnostics Lab Dashboard - Enhanced & Error-Free
Intelligent billing analysis for your exact data structure
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
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION - BASED ON YOUR EXACT FILE STRUCTURE
# ==============================================================================

# Flexible column mapping for variations in your file
COLUMN_VARIATIONS = {
'date': ['Date', 'Invoice Date', 'Bill Date', 'Transaction Date', 'Entry Date'],
'client': ['Client_Name', 'Client Name', 'Customer', 'Hospital', 'Organization'],
'amount': ['Amount', 'Net Amount', 'Final Amount', 'Revenue', 'Net Revenue'],
'specialty': ['Specialty', 'Test Name', 'Service', 'Investigation', 'Test Type'],
'zone': ['Zone', 'Territory', 'Region', 'Area', 'Location'],
'payment_status': ['Payment_Status', 'Payment Status', 'Collection Status', 'Status'],
'organization': ['Organization', 'Org', 'Hospital Type', 'Client Type'],
'bill_amount': ['Bill_Amount', 'Bill Amount', 'Gross Amount', 'Total Amount'],
'salesperson': ['Salesperson', 'Sales Person', 'Executive', 'Agent', 'Marketing Person'],
'branch': ['Branch', 'Location', 'Centre', 'Lab']
}

CONFIG_FILE = 'truemedix_config.json'
DEFAULT_TARGET = 10000000  # ₹1 crore
SPECIALTY_TEST_THRESHOLD = 999  # Minimum amount for specialty tests

# ==============================================================================
# INTELLIGENT ANALYSIS ENGINE
# ==============================================================================

class TruemedixAnalyzer:
    """Intelligent analyzer for your specific billing data"""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.original_columns = list(df.columns)
        self.analysis_cache = {}
        self.column_mapping = {}
        
    def detect_column_structure(self) -> Dict[str, str]:
        """Detect and map your file's column structure"""
        column_mapping = {}
        df_columns_lower = [col.lower().strip() for col in self.df.columns]
        
        for standard_key, variations in COLUMN_VARIATIONS.items():
            for variation in variations:
                variation_lower = variation.lower().strip()
                if variation_lower in df_columns_lower:
                    original_col = self.df.columns[df_columns_lower.index(variation_lower)]
                    column_mapping[original_col] = standard_key
                    break
        
        self.column_mapping = column_mapping
        return column_mapping

    def clean_and_prepare_data(self) -> pd.DataFrame:
        """Clean data specifically for your file format"""
        df = self.df.copy()
        
        # Detect column mapping
        column_mapping = self.detect_column_structure()
        
        # Create reverse mapping for renaming
        rename_mapping = {orig: mapped for orig, mapped in column_mapping.items()}
        mapped_df = df.rename(columns=rename_mapping)
        
        # Clean date columns
        date_columns = [col for col in mapped_df.columns if 'date' in col.lower()]
        for col in date_columns:
            mapped_df[col] = pd.to_datetime(mapped_df[col], errors='coerce')
        
        # Clean amount columns
        amount_columns = [col for col in mapped_df.columns if any(x in col.lower() for x in ['amount', 'revenue', 'bill'])]
        for col in amount_columns:
            if mapped_df[col].dtype == 'object':
                # Remove currency symbols and convert to numeric
                mapped_df[col] = mapped_df[col].astype(str).str.replace('₹', '', regex=False)
                mapped_df[col] = mapped_df[col].str.replace(',', '', regex=False)
                mapped_df[col] = mapped_df[col].str.replace('Rs.', '', regex=False)
                mapped_df[col] = pd.to_numeric(mapped_df[col], errors='coerce').fillna(0)
        
        # Handle specialty tests
        if 'amount' in mapped_df.columns:
            mapped_df['is_specialty_test'] = mapped_df['amount'] >= SPECIALTY_TEST_THRESHOLD
        
        return mapped_df

    def calculate_comprehensive_metrics(self) -> Dict[str, Any]:
        """Calculate all business metrics from your data"""
        df = self.clean_and_prepare_data()
        
        metrics = {
            'total_records': len(df),
            'date_range': {
                'start': df['date'].min() if 'date' in df.columns else None,
                'end': df['date'].max() if 'date' in df.columns else None
            },
            'revenue_metrics': {},
            'client_analysis': {},
            'zone_performance': {},
            'specialty_analysis': {},
            'payment_analysis': {},
            'trend_analysis': {}
        }
        
        # Revenue Metrics
        if 'amount' in df.columns:
            metrics['revenue_metrics'] = {
                'total_revenue': float(df['amount'].sum()),
                'average_transaction': float(df['amount'].mean()),
                'median_transaction': float(df['amount'].median()),
                'max_transaction': float(df['amount'].max()),
                'min_transaction': float(df['amount'].min()),
                'specialty_revenue': float(df[df['is_specialty_test']]['amount'].sum() if 'is_specialty_test' in df.columns else 0),
                'specialty_count': int(df['is_specialty_test'].sum() if 'is_specialty_test' in df.columns else 0)
            }
        
        # Client Analysis
        if 'client' in df.columns and 'amount' in df.columns:
            client_stats = df.groupby('client')['amount'].agg(['sum', 'count', 'mean']).round(2)
            
            metrics['client_analysis'] = {
                'total_clients': int(df['client'].nunique()),
                'top_clients_data': client_stats.nlargest(10, 'sum').to_dict('index')
            }
        
        # Zone/Territory Performance
        if 'zone' in df.columns and 'amount' in df.columns:
            zone_stats = df.groupby('zone')['amount'].agg(['sum', 'count', 'mean']).round(2)
            
            metrics['zone_performance'] = {
                'total_zones': int(df['zone'].nunique()),
                'zone_data': zone_stats.to_dict('index')
            }
        
        # Payment Analysis
        if 'payment_status' in df.columns:
            payment_stats = df['payment_status'].value_counts()
            metrics['payment_analysis'] = {
                'payment_distribution': payment_stats.to_dict()
            }
        
        return metrics

    def answer_natural_query(self, question: str) -> str:
        """Answer natural language questions about the data"""
        df = self.clean_and_prepare_data()
        question_lower = question.lower()
        
        try:
            # Top clients queries
            if any(phrase in question_lower for phrase in ['top client', 'best client', 'highest client']):
                if 'client' in df.columns and 'amount' in df.columns:
                    top_clients = df.groupby('client')['amount'].sum().nlargest(10)
                    result = "Top 10 Clients by Revenue:\n"
                    for client, revenue in top_clients.items():
                        result += f"{client}: ₹{revenue:,.2f}\n"
                    return result
            
            # Zone performance queries
            elif any(phrase in question_lower for phrase in ['zone perform', 'best zone', 'territory']):
                if 'zone' in df.columns and 'amount' in df.columns:
                    zone_performance = df.groupby('zone')['amount'].sum().sort_values(ascending=False)
                    result = "Zone Performance Rankings:\n"
                    for zone, revenue in zone_performance.items():
                        result += f"{zone}: ₹{revenue:,.2f}\n"
                    return result
            
            # Specialty test queries
            elif any(phrase in question_lower for phrase in ['specialty', 'high value', '999']):
                if 'amount' in df.columns:
                    specialty_tests = df[df['amount'] >= SPECIALTY_TEST_THRESHOLD]
                    specialty_revenue = specialty_tests['amount'].sum()
                    specialty_count = len(specialty_tests)
                    return f"Specialty Tests (≥₹{SPECIALTY_TEST_THRESHOLD}):\nCount: {specialty_count:,}\nRevenue: ₹{specialty_revenue:,.2f}"
            
            # Payment/Collection queries
            elif any(phrase in question_lower for phrase in ['payment', 'collection', 'outstanding']):
                if 'payment_status' in df.columns:
                    payment_dist = df['payment_status'].value_counts()
                    result = "Payment Status Distribution:\n"
                    for status, count in payment_dist.items():
                        result += f"{status}: {count:,}\n"
                    return result
            
            # Monthly/Daily trend queries
            elif any(phrase in question_lower for phrase in ['trend', 'monthly', 'daily', 'growth']):
                if 'date' in df.columns and 'amount' in df.columns:
                    monthly_trend = df.groupby(df['date'].dt.to_period('M'))['amount'].sum()
                    result = "Monthly Revenue Trend:\n"
                    for period, revenue in monthly_trend.items():
                        result += f"{period}: ₹{revenue:,.2f}\n"
                    return result
            
            else:
                return """I can help you analyze:
• Top clients - "Show me top 10 clients"
• Zone performance - "Which zone is performing best?"
• Specialty tests - "What's our specialty test performance?"
• Payment status - "Show me payment collection rate"
• Revenue trends - "Show me monthly revenue trends"

Please ask specific questions about these areas."""

        except Exception as e:
            return f"Error processing query: {str(e)}"


# ==============================================================================
# ENHANCED UTILITY FUNCTIONS
# ==============================================================================

def load_config() -> Dict[str, Any]:
    """Load dashboard configuration"""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {'monthly_target': DEFAULT_TARGET}

def save_config(config: Dict[str, Any]) -> None:
    """Save dashboard configuration"""
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    except Exception:
        pass

def safe_rerun():
    """Safely handle Streamlit rerun for different versions"""
    try:
        st.rerun()
    except AttributeError:
        try:
            st.experimental_rerun()
        except AttributeError:
            st.write("Please refresh the page to see changes")

@st.cache_data
def load_your_specific_file(file_content: bytes, filename: str) -> Tuple[Optional[pd.DataFrame], Optional[TruemedixAnalyzer], List[str]]:
    """Load and process your specific Excel/CSV file format"""
    try:
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(file_content))
        else:
            # Try different engines for Excel
            try:
                df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
            except Exception:
                df = pd.read_excel(io.BytesIO(file_content), engine='xlrd')

        # Remove completely empty rows
        df = df.dropna(how='all')
        
        # Initialize analyzer for your data
        analyzer = TruemedixAnalyzer(df)
        processed_df = analyzer.clean_and_prepare_data()
        
        return processed_df, analyzer, []

    except Exception as e:
        return None, None, [f"Error loading file: {str(e)}"]

def calculate_target_metrics(df: pd.DataFrame, monthly_target: float) -> Dict[str, float]:
    """Calculate target tracking metrics with your data structure"""
    if df.empty:
        return {
            'expected_revenue': 0, 'actual_revenue': 0, 'variance_amount': 0,
            'variance_percent': 0, 'monthly_projection': 0, 'completion_percent': 0,
            'days_elapsed': 0, 'days_in_month': 30
        }

    # Current month calculations
    now = datetime.now()
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    days_elapsed = now.day

    daily_target = monthly_target / days_in_month
    expected_revenue = daily_target * days_elapsed

    # Current month data based on your file structure
    current_month = now.replace(day=1)
    if 'date' in df.columns:
        current_data = df[df['date'] >= current_month]
        actual_revenue = current_data['amount'].sum() if 'amount' in current_data.columns else 0
    else:
        actual_revenue = 0

    # Calculate metrics
    variance_amount = actual_revenue - expected_revenue
    variance_percent = (variance_amount / expected_revenue * 100) if expected_revenue > 0 else 0
    daily_average = actual_revenue / days_elapsed if days_elapsed > 0 else 0
    monthly_projection = daily_average * days_in_month
    completion_percent = (actual_revenue / monthly_target * 100) if monthly_target > 0 else 0

    return {
        'expected_revenue': expected_revenue,
        'actual_revenue': actual_revenue,
        'variance_amount': variance_amount,
        'variance_percent': variance_percent,
        'monthly_projection': monthly_projection,
        'completion_percent': completion_percent,
        'days_elapsed': days_elapsed,
        'days_in_month': days_in_month
    }

# ==============================================================================
# ENHANCED STREAMLIT DASHBOARD
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Truemedix Analytics - Enhanced",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load config
    config = load_config()

    # Initialize session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'monthly_target' not in st.session_state:
        st.session_state.monthly_target = config.get('monthly_target', DEFAULT_TARGET)

    # Sidebar Navigation
    with st.sidebar:
        st.title("🧪 Truemedix Analytics")
        st.caption("Enhanced Intelligence System")
        
        page = st.selectbox(
            "Navigation",
            ["Upload", "Intelligence", "Overview", "Performance", "Collections", "Targets", "Query Engine"],
            index=1 if st.session_state.df is not None else 0
        )
        
        st.divider()
        
        # File upload section
        if page == "Upload" or st.session_state.df is None:
            st.header("📁 Upload Your Data")
            
            uploaded_file = st.file_uploader(
                "Upload your billing file",
                type=['xlsx', 'xls', 'csv'],
                help="Upload your exact billing file format"
            )
            
            if uploaded_file:
                file_content = uploaded_file.read()
                df, analyzer, errors = load_your_specific_file(file_content, uploaded_file.name)
                
                if errors:
                    for error in errors:
                        st.error(error)
                
                if df is not None and analyzer is not None:
                    st.session_state.df = df
                    st.session_state.analyzer = analyzer
                    st.success(f"✅ Loaded {len(df):,} records")
                    
                    # Show detected structure
                    st.info("🔍 Detected File Structure:")
                    column_mapping = analyzer.detect_column_structure()
                    for orig, mapped in column_mapping.items():
                        st.text(f"'{orig}' → {mapped}")
        
        # Target settings
        if st.session_state.df is not None:
            st.header("🎯 Monthly Target")
            
            # Quick presets
            col1, col2 = st.columns(2)
            with col1:
                if st.button("₹50L"):
                    st.session_state.monthly_target = 5000000
                    safe_rerun()
                if st.button("₹1.5Cr"):
                    st.session_state.monthly_target = 15000000
                    safe_rerun()
            with col2:
                if st.button("₹1Cr"):
                    st.session_state.monthly_target = 10000000
                    safe_rerun()
                if st.button("₹2Cr"):
                    st.session_state.monthly_target = 20000000
                    safe_rerun()
            
            # Manual input
            target = st.number_input(
                "Custom Target (₹)",
                value=float(st.session_state.monthly_target),
                step=100000.0,
                format="%.0f"
            )
            st.session_state.monthly_target = target
            
            if st.button("💾 Save Target"):
                config['monthly_target'] = target
                save_config(config)
                st.success("Target saved!")

    # Main content based on page selection
    if st.session_state.df is None:
        st.title("🧪 Truemedix Analytics - Enhanced Intelligence")
        st.info("👈 Please upload your billing file to begin intelligent analysis")
        
        st.subheader("🚀 Enhanced Features")
        st.markdown("""
        **Intelligent Analysis for Your Exact File Format:**
        
        ✅ **Smart Column Detection** - Automatically detects your file structure  
        ✅ **Comprehensive Metrics** - Revenue, clients, zones, specialties, payments  
        ✅ **Natural Language Queries** - Ask questions in plain English  
        ✅ **Territory Analysis** - Zone/region performance tracking  
        ✅ **Payment Intelligence** - Collection rates and status tracking  
        ✅ **Specialty Test Focus** - High-value test identification (≥₹999)  
        ✅ **Business Intelligence** - Trends, insights, and recommendations  
        """)
        return

    df = st.session_state.df
    analyzer = st.session_state.analyzer

    # Intelligence Page
    if page == "Intelligence":
        st.title("🧠 Business Intelligence Dashboard")
        
        # Calculate comprehensive metrics
        with st.spinner("Analyzing your data..."):
            metrics = analyzer.calculate_comprehensive_metrics()
        
        # Display key insights
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Records", 
                f"{metrics['total_records']:,}",
                help="Total transactions in your file"
            )
        
        with col2:
            if metrics['revenue_metrics']:
                st.metric(
                    "Total Revenue", 
                    f"₹{metrics['revenue_metrics']['total_revenue']:,.0f}",
                    help="Sum of all revenue"
                )
        
        with col3:
            if metrics['client_analysis']:
                st.metric(
                    "Unique Clients", 
                    f"{metrics['client_analysis']['total_clients']:,}",
                    help="Number of unique clients"
                )
        
        with col4:
            if metrics['revenue_metrics'] and metrics['revenue_metrics']['specialty_count']:
                st.metric(
                    "Specialty Tests", 
                    f"{metrics['revenue_metrics']['specialty_count']:,}",
                    help=f"High-value tests ≥₹{SPECIALTY_TEST_THRESHOLD}"
                )
        
        # Detailed Analysis Sections
        if metrics['revenue_metrics']:
            st.subheader("💰 Revenue Analysis")
            rev_col1, rev_col2, rev_col3 = st.columns(3)
            
            with rev_col1:
                st.metric("Average Transaction", f"₹{metrics['revenue_metrics']['average_transaction']:,.2f}")
            with rev_col2:
                st.metric("Median Transaction", f"₹{metrics['revenue_metrics']['median_transaction']:,.2f}")
            with rev_col3:
                st.metric("Max Transaction", f"₹{metrics['revenue_metrics']['max_transaction']:,.2f}")
        
        # Client Intelligence
        if 'client' in df.columns and 'amount' in df.columns:
            st.subheader("👥 Client Intelligence")
            
            # Top clients analysis
            top_clients = df.groupby('client')['amount'].agg(['sum', 'count', 'mean']).round(2)
            top_clients.columns = ['Total Revenue', 'Transactions', 'Avg Transaction']
            top_clients = top_clients.sort_values('Total Revenue', ascending=False).head(10)
            
            # Format currency
            top_clients['Total Revenue'] = top_clients['Total Revenue'].apply(lambda x: f"₹{x:,.2f}")
            top_clients['Avg Transaction'] = top_clients['Avg Transaction'].apply(lambda x: f"₹{x:,.2f}")
            
            st.dataframe(top_clients, use_container_width=True)
        
        # Zone Performance
        if 'zone' in df.columns and 'amount' in df.columns:
            st.subheader("🗺️ Territory Performance")
            
            zone_performance = df.groupby('zone')['amount'].agg(['sum', 'count', 'mean']).round(2)
            zone_performance.columns = ['Total Revenue', 'Transactions', 'Avg Transaction']
            zone_performance = zone_performance.sort_values('Total Revenue', ascending=False)
            
            # Visualization
            fig = px.bar(
                zone_performance.reset_index(),
                x='zone',
                y='Total Revenue',
                title="Revenue by Territory/Zone"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Format and display table
            zone_performance['Total Revenue'] = zone_performance['Total Revenue'].apply(lambda x: f"₹{x:,.2f}")
            zone_performance['Avg Transaction'] = zone_performance['Avg Transaction'].apply(lambda x: f"₹{x:,.2f}")
            st.dataframe(zone_performance, use_container_width=True)

    # Natural Language Query Engine
    elif page == "Query Engine":
        st.title("🤖 Natural Language Query Engine")
        st.caption("Ask questions about your data in plain English")
        
        # Sample questions
        st.subheader("💡 Try These Questions:")
        sample_questions = [
            "Show me top 10 clients this month",
            "Which zone is performing best?", 
            "What's our specialty test performance?",
            "Show me payment collection rate",
            "What are the monthly revenue trends?",
            "Who are my biggest clients?",
            "How many high-value tests do we have?"
        ]
        
        selected_question = st.selectbox("Quick Questions:", [""] + sample_questions)
        
        # Query input
        user_question = st.text_input(
            "Ask your question:",
            value=selected_question,
            placeholder="e.g., 'Show me top performing zones' or 'What's our collection rate?'"
        )
        
        if st.button("🔍 Analyze") and user_question:
            with st.spinner("Analyzing your data..."):
                try:
                    answer = analyzer.answer_natural_query(user_question)
                    st.success("Analysis Complete!")
                    st.text_area("Answer:", answer, height=200)
                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
        
        # Export analysis results
        if st.button("📊 Generate Full Analysis Report"):
            with st.spinner("Generating comprehensive report..."):
                metrics = analyzer.calculate_comprehensive_metrics()
                
                # Create comprehensive report
                report_data = []
                
                # Add all available metrics to report
                if metrics['revenue_metrics']:
                    for key, value in metrics['revenue_metrics'].items():
                        report_data.append({'Metric': f"Revenue - {key}", 'Value': value})
                
                if report_ :
                    report_df = pd.DataFrame(report_data)
                    csv = report_df.to_csv(index=False)
                    
                    st.download_button(
                        label="📥 Download Analysis Report",
                        data=csv,
                        file_name=f"truemedix_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                        mime="text/csv"
                    )

    # Enhanced Overview Page
    elif page == "Overview":
        st.title("📊 Enhanced Dashboard Overview")
        
        # Calculate metrics using your data structure
        if 'amount' in df.columns:
            total_revenue = df['amount'].sum()
            total_tests = len(df)
            avg_transaction = df['amount'].mean()
            
            # Specialty tests
            specialty_tests = df[df['amount'] >= SPECIALTY_TEST_THRESHOLD] if 'amount' in df.columns else pd.DataFrame()
            specialty_count = len(specialty_tests)
            specialty_revenue = specialty_tests['amount'].sum() if not specialty_tests.empty else 0
            
            # Client metrics
            unique_clients = df['client'].nunique() if 'client' in df.columns else 0
            
            # 5-metric row
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("💰 Total Revenue", f"₹{total_revenue:,.0f}")
            with col2:
                st.metric("🧪 Total Tests", f"{total_tests:,}")
            with col3:
                st.metric("👥 Unique Clients", f"{unique_clients:,}")
            with col4:
                st.metric("⭐ Specialty Tests", f"{specialty_count:,}")
            with col5:
                st.metric("📊 Avg Transaction", f"₹{avg_transaction:,.0f}")
            
            # Target tracking
            st.divider()
            target_metrics = calculate_target_metrics(df, st.session_state.monthly_target)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                variance_delta = f"₹{target_metrics['variance_amount']:+,.0f} ({target_metrics['variance_percent']:+.1f}%)"
                st.metric(
                    "🎯 Target by Today",
                    f"₹{target_metrics['expected_revenue']:,.0f}",
                    delta=variance_delta
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
                    "🏁 Target Completion %",
                    f"{target_metrics['completion_percent']:.1f}%",
                    delta=f"Day {target_metrics['days_elapsed']}/{target_metrics['days_in_month']}"
                )
            
            # Charts based on your data structure
            st.divider()
            
            # Revenue trend
            if 'date' in df.columns:
                st.subheader("📈 Revenue Trend")
                daily_revenue = df.groupby(df['date'].dt.date)['amount'].sum().reset_index()
                daily_revenue.columns = ['Date', 'Revenue']
                
                fig = px.line(daily_revenue, x='Date', y='Revenue', title="Daily Revenue")
                st.plotly_chart(fig, use_container_width=True)
            
            # Top performers
            if 'client' in df.columns:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("🏆 Top Clients")
                    top_clients = df.groupby('client')['amount'].sum().nlargest(5)
                    st.bar_chart(top_clients)
                
                with col2:
                    if 'zone' in df.columns:
                        st.subheader("🗺️ Zone Performance")
                        zone_performance = df.groupby('zone')['amount'].sum().nlargest(5)
                        st.bar_chart(zone_performance)
        else:
            st.warning("Amount column not found in your data. Please check file format.")

    # Continue with other pages
    elif page == "Performance":
        st.title("🏆 Performance Analysis")
        st.info("Performance analysis features coming soon...")

    elif page == "Collections":
        st.title("💳 Collections Analysis")
        st.info("Collections analysis features coming soon...")

    elif page == "Targets":
        st.title("🎯 Target Management")
        target_metrics = calculate_target_metrics(df, st.session_state.monthly_target)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Monthly Target", f"₹{st.session_state.monthly_target:,.0f}")
        with col2:
            st.metric("Achieved So Far", f"₹{target_metrics['actual_revenue']:,.0f}")
        with col3:
            st.metric("Projected Final", f"₹{target_metrics['monthly_projection']:,.0f}")
        
        # Progress bar
        progress = min(100, target_metrics['completion_percent'])
        st.progress(progress / 100, text=f"Target Progress: {progress:.1f}%")


if __name__ == "__main__":
    main()
