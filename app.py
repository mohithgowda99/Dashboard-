"""
Advanced Truemedix Diagnostics Analytics Dashboard
Production-ready system with intelligent data handling, anomaly detection, and scalability
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
import glob
import calendar
import re
import warnings
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import hashlib
from dataclasses import dataclass, asdict, field
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION & DATA MODELS - FIXED
# ==============================================================================

@dataclass
class DataConfig:
    """Configuration for data processing"""
    historical_folder: str = "historical"
    daily_folder: str = "daily"
    master_db: str = "truemedix_master.db"
    config_file: str = "truemedix_config.json"
    specialty_threshold: int = 999
    anomaly_threshold: float = 0.25  # 25% deviation threshold
    supported_formats: List[str] = field(default_factory=lambda: ['.csv', '.xlsx', '.xls'])

@dataclass
class ProcessingStats:
    """Statistics from data processing"""
    files_processed: int = 0
    records_loaded: int = 0
    records_deduplicated: int = 0
    missing_columns: List[str] = field(default_factory=list)
    fallbacks_applied: List[str] = field(default_factory=list)
    anomalies_detected: List[Dict] = field(default_factory=list)

# Enhanced column mapping with priorities
COLUMN_MAPPING = {
    'date_columns': {
        'primary': ['InvoiceDate', 'Date', 'Invoice Date', 'Bill Date', 'Transaction Date'],
        'secondary': ['Test Date', 'Report Date', 'Entry Date', 'Created Date'],
        'fallback': datetime.now()
    },
    'amount_columns': {
        'primary': ['NetRevenue', 'Net Revenue', 'Amount', 'Total Amount', 'Revenue'],
        'secondary': ['BilledAmount', 'Billed Amount', 'Final Amount', 'Invoice Amount'],
        'fallback': 0
    },
    'client_columns': {
        'primary': ['ClientName', 'Client Name', 'Customer', 'Hospital Name'],
        'secondary': ['Organization', 'Client', 'Hospital', 'Clinic Name'],
        'fallback': 'Unknown Client'
    },
    'branch_columns': {
        'primary': ['Branch', 'Location', 'Zone', 'Territory'],
        'secondary': ['Region', 'Area', 'City', 'Centre'],
        'fallback': 'Unknown Branch'
    },
    'salesperson_columns': {
        'primary': ['Salesperson', 'Sales Person', 'Executive', 'Sales Executive'],
        'secondary': ['Agent', 'Representative', 'Marketing Person', 'Sales Rep'],
        'fallback': 'Unknown Salesperson'
    },
    'specialty_columns': {
        'primary': ['Specialty', 'Test Name', 'Service', 'Investigation'],
        'secondary': ['Test Type', 'Department', 'Category', 'Test'],
        'fallback': 'General'
    },
    'id_columns': {
        'primary': ['BillID', 'Bill ID', 'InvoiceID', 'Invoice ID'],
        'secondary': ['ID', 'TransactionID', 'Transaction ID', 'Reference'],
        'fallback': None  # Will generate unique ID
    }
}

# ==============================================================================
# INTELLIGENT DATA PROCESSOR - SIMPLIFIED & FIXED
# ==============================================================================

class AdvancedDataProcessor:
    """Advanced data processor with intelligent column mapping and fallback logic"""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.stats = ProcessingStats()
        self.ensure_directories()
        
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        try:
            Path(self.config.historical_folder).mkdir(exist_ok=True)
            Path(self.config.daily_folder).mkdir(exist_ok=True)
        except Exception as e:
            logger.warning(f"Could not create directories: {e}")
    
    def find_column(self, df: pd.DataFrame, column_type: str) -> Tuple[str, str]:
        """
        Intelligently find the best matching column for a given type
        Returns: (column_name, source_level)
        """
        if column_type not in COLUMN_MAPPING:
            return None, 'missing'
            
        mapping = COLUMN_MAPPING[column_type]
        df_columns = df.columns.tolist()
        
        # Try primary columns first
        for col_pattern in mapping.get('primary', []):
            for df_col in df_columns:
                if self._columns_match(col_pattern, df_col):
                    return df_col, 'primary'
        
        # Try secondary columns
        for col_pattern in mapping.get('secondary', []):
            for df_col in df_columns:
                if self._columns_match(col_pattern, df_col):
                    return df_col, 'secondary'
        
        return None, 'missing'
    
    def _columns_match(self, pattern: str, column: str) -> bool:
        """Check if column matches pattern (case-insensitive, flexible)"""
        try:
            pattern_clean = str(pattern).strip().lower().replace('_', ' ')
            column_clean = str(column).strip().lower().replace('_', ' ')
            
            # Exact match
            if pattern_clean == column_clean:
                return True
            
            # Fuzzy match (one contains the other)
            if pattern_clean in column_clean or column_clean in pattern_clean:
                return True
            
            # Keyword match for amount columns
            if 'amount' in pattern_clean or 'revenue' in pattern_clean:
                amount_keywords = ['amt', 'revenue', 'total', 'amount', 'bill', 'value', 'price']
                return any(keyword in column_clean for keyword in amount_keywords)
            
            return False
        except Exception:
            return False
    
    def process_single_file(self, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Process a single file with intelligent column mapping"""
        try:
            # Load file based on extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path, engine='openpyxl')
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Clean dataframe
            df = df.dropna(how='all', axis=0)  # Remove empty rows
            df = df.dropna(how='all', axis=1)  # Remove empty columns
            df.columns = df.columns.astype(str).str.strip()  # Clean column names
            
            if len(df) == 0:
                return None, {"error": "File is empty after cleaning"}
            
            # Apply intelligent column mapping
            processed_df, mapping_info = self._apply_intelligent_mapping(df, file_path)
            
            return processed_df, mapping_info
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return None, {"error": str(e)}
    
    def _apply_intelligent_mapping(self, df: pd.DataFrame, file_path: str) -> Tuple[pd.DataFrame, Dict]:
        """Apply intelligent column mapping with fallbacks"""
        processed_df = df.copy()
        mapping_info = {
            'file_path': file_path,
            'original_columns': list(df.columns),
            'mappings_applied': {},
            'fallbacks_used': [],
            'warnings': []
        }
        
        # Process each column type
        for column_type in COLUMN_MAPPING.keys():
            target_column = column_type.replace('_columns', '').title()
            
            found_column, source_level = self.find_column(df, column_type)
            
            if found_column:
                # Use found column
                if found_column != target_column:
                    processed_df = processed_df.rename(columns={found_column: target_column})
                    mapping_info['mappings_applied'][target_column] = {
                        'source': found_column,
                        'level': source_level
                    }
                
                # Apply data cleaning based on column type
                processed_df = self._clean_column_data(processed_df, target_column, column_type)
                
            else:
                # Apply fallback
                fallback_value = COLUMN_MAPPING[column_type]['fallback']
                if fallback_value is not None:
                    if column_type == 'id_columns':
                        # Generate unique IDs
                        processed_df[target_column] = [f"AUTO_{i}_{hash(str(row))[:8]}" 
                                                     for i, row in enumerate(processed_df.itertuples())]
                    else:
                        processed_df[target_column] = fallback_value
                    
                    mapping_info['fallbacks_used'].append(target_column)
                    self.stats.fallbacks_applied.append(f"{target_column} in {file_path}")
        
        # Add derived columns
        processed_df = self._add_derived_columns(processed_df)
        
        return processed_df, mapping_info
    
    def _clean_column_data(self, df: pd.DataFrame, column: str, column_type: str) -> pd.DataFrame:
        """Clean data based on column type"""
        try:
            if column_type == 'date_columns':
                df[column] = pd.to_datetime(df[column], errors='coerce')
                # Fill NaT with current date
                df[column] = df[column].fillna(pd.Timestamp.now())
                
            elif column_type == 'amount_columns':
                # Clean currency and convert to numeric
                if df[column].dtype == 'object':
                    df[column] = df[column].astype(str)
                    df[column] = df[column].str.replace('₹', '', regex=False)
                    df[column] = df[column].str.replace('Rs.', '', regex=False)
                    df[column] = df[column].str.replace(',', '', regex=False)
                    df[column] = df[column].str.replace('INR', '', regex=False)
                    df[column] = df[column].str.strip()
                
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)
                
            elif column_type in ['client_columns', 'branch_columns', 'salesperson_columns', 'specialty_columns']:
                # Clean text data
                df[column] = df[column].astype(str).str.strip()
                fallback = COLUMN_MAPPING[column_type]['fallback']
                df[column] = df[column].replace(['nan', 'NaN', '', 'null'], fallback)
        except Exception as e:
            logger.warning(f"Error cleaning column {column}: {e}")
        
        return df
    
    def _add_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived columns for analysis"""
        try:
            # High-value test flag
            if 'Amount' in df.columns:
                df['HighValueTest'] = df['Amount'] >= self.config.specialty_threshold
            
            # Time-based columns
            if 'Date' in df.columns:
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['YearMonth'] = df['Date'].dt.to_period('M')
                df['DayOfWeek'] = df['Date'].dt.day_name()
                df['IsWeekend'] = df['Date'].dt.weekday >= 5
            
            # Revenue categories
            if 'Amount' in df.columns:
                df['RevenueCategory'] = pd.cut(df['Amount'], 
                                             bins=[0, 500, 1000, 5000, float('inf')],
                                             labels=['Low', 'Medium', 'High', 'Premium'])
        except Exception as e:
            logger.warning(f"Error adding derived columns: {e}")
        
        return df
    
    def load_all_data(self) -> Tuple[pd.DataFrame, ProcessingStats]:
        """Load and merge all data from historical and daily folders"""
        all_dataframes = []
        unique_ids = set()
        
        # Process historical data
        historical_files = self._get_files_in_folder(self.config.historical_folder)
        daily_files = self._get_files_in_folder(self.config.daily_folder)
        
        all_files = historical_files + daily_files
        self.stats.files_processed = len(all_files)
        
        if not all_files:
            st.warning("⚠️ No data files found in historical/ or daily/ folders")
            return pd.DataFrame(), self.stats
        
        for file_path in all_files:
            df, mapping_info = self.process_single_file(file_path)
            
            if df is not None:
                # Track processing
                self.stats.records_loaded += len(df)
                
                # Deduplicate based on ID column
                if 'Id' in df.columns:
                    before_dedup = len(df)
                    df = df[~df['Id'].isin(unique_ids)]
                    unique_ids.update(df['Id'].tolist())
                    after_dedup = len(df)
                    self.stats.records_deduplicated += (before_dedup - after_dedup)
                
                if len(df) > 0:
                    df['SourceFile'] = os.path.basename(file_path)
                    all_dataframes.append(df)
        
        # Combine all dataframes
        if all_dataframes:
            master_df = pd.concat(all_dataframes, ignore_index=True)
            master_df = self._final_data_processing(master_df)
        else:
            master_df = pd.DataFrame()
        
        return master_df, self.stats
    
    def _get_files_in_folder(self, folder_path: str) -> List[str]:
        """Get all supported files in a folder"""
        files = []
        if os.path.exists(folder_path):
            for ext in self.config.supported_formats:
                pattern = os.path.join(folder_path, f"*{ext}")
                files.extend(glob.glob(pattern))
        return sorted(files)
    
    def _final_data_processing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final processing and validation"""
        try:
            # Sort by date
            if 'Date' in df.columns:
                df = df.sort_values('Date')
            
            # Reset index
            df = df.reset_index(drop=True)
            
            # Add processing timestamp
            df['ProcessedAt'] = pd.Timestamp.now()
        except Exception as e:
            logger.warning(f"Error in final processing: {e}")
        
        return df

# ==============================================================================
# SIMPLIFIED VISUALIZATION ENGINE
# ==============================================================================

class SimpleVisualizations:
    """Simplified visualization components"""
    
    @staticmethod
    def create_revenue_chart(df: pd.DataFrame) -> go.Figure:
        """Create basic revenue trend chart"""
        if 'Date' not in df.columns or 'Amount' not in df.columns:
            fig = go.Figure()
            fig.add_annotation(text="No revenue data available", x=0.5, y=0.5)
            return fig
        
        try:
            daily_revenue = df.groupby('Date')['Amount'].sum().reset_index()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_revenue['Date'],
                y=daily_revenue['Amount'],
                mode='lines+markers',
                name='Daily Revenue',
                line=dict(width=2)
            ))
            
            fig.update_layout(
                title='Daily Revenue Trend',
                xaxis_title='Date',
                yaxis_title='Revenue (₹)',
                height=400
            )
            
            return fig
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5)
            return fig
    
    @staticmethod
    def create_kpi_summary(df: pd.DataFrame) -> Dict[str, Any]:
        """Create basic KPI summary"""
        try:
            kpis = {
                'total_revenue': df['Amount'].sum() if 'Amount' in df.columns else 0,
                'total_transactions': len(df),
                'avg_transaction': df['Amount'].mean() if 'Amount' in df.columns else 0,
                'unique_clients': df['Client'].nunique() if 'Client' in df.columns else 0,
                'date_range': f"{df['Date'].min().date()} to {df['Date'].max().date()}" if 'Date' in df.columns else 'N/A'
            }
            return kpis
        except Exception as e:
            return {
                'total_revenue': 0,
                'total_transactions': 0,
                'avg_transaction': 0,
                'unique_clients': 0,
                'date_range': 'N/A',
                'error': str(e)
            }

# ==============================================================================
# SIMPLIFIED STREAMLIT APPLICATION
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Truemedix Analytics",
        page_icon="🧪",
        layout="wide"
    )
    
    st.title("🧪 Truemedix Enhanced Analytics Dashboard")
    st.markdown("*Intelligent data processing with fault tolerance*")
    
    # Initialize configuration
    config = DataConfig()
    
    # Initialize session state
    if 'master_data' not in st.session_state:
        st.session_state.master_data = None
    if 'processing_stats' not in st.session_state:
        st.session_state.processing_stats = None
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = None
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Data Management")
        
        # Data refresh section
        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Processing data files..."):
                try:
                    processor = AdvancedDataProcessor(config)
                    master_data, stats = processor.load_all_data()
                    
                    st.session_state.master_data = master_data
                    st.session_state.processing_stats = stats
                    st.session_state.last_refresh = datetime.now()
                    
                    if len(master_data) > 0:
                        st.success(f"✅ Loaded {len(master_data):,} records from {stats.files_processed} files")
                    else:
                        st.warning("⚠️ No data loaded. Check if files exist in historical/ or daily/ folders")
                        
                except Exception as e:
                    st.error(f"❌ Error loading  {str(e)}")
        
        # Show processing stats
        if st.session_state.processing_stats:
            stats = st.session_state.processing_stats
            st.subheader("📊 Processing Stats")
            
            st.metric("Files Processed", stats.files_processed)
            st.metric("Records Loaded", f"{stats.records_loaded:,}")
            
            if stats.fallbacks_applied:
                st.warning(f"⚠️ {len(stats.fallbacks_applied)} fallbacks applied")
    
    # Main content
    if st.session_state.master_data is None or len(st.session_state.master_data) == 0:
        st.info("👆 Click 'Load Data' to process your files")
        
        # Show expected folder structure
        st.subheader("📁 Expected Structure")
        st.code("""
        project/
        ├── historical/     # Historical data files
        │   └── *.xlsx, *.csv
        ├── daily/         # Daily data files  
        │   └── *.xlsx, *.csv
        └── app.py         # This dashboard
        """)
        
        st.subheader("📋 Sample Data Format")
        sample_data = pd.DataFrame({
            'Date': ['2024-01-15', '2024-01-16'],
            'Client Name': ['Hospital A', 'Clinic B'],
            'Amount': [1500, 2000],
            'Branch': ['Main', 'North']
        })
        st.dataframe(sample_data)
        
        return
    
    # Get data and show dashboard
    df = st.session_state.master_data
    
    # Create KPIs
    kpis = SimpleVisualizations.create_kpi_summary(df)
    
    # Display KPIs
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("💰 Total Revenue", f"₹{kpis['total_revenue']:,.0f}")
    
    with col2:
        st.metric("🧪 Transactions", f"{kpis['total_transactions']:,}")
    
    with col3:
        st.metric("👥 Unique Clients", f"{kpis['unique_clients']:,}")
    
    with col4:
        st.metric("📅 Date Range", kpis['date_range'])
    
    # Show revenue chart
    st.subheader("📈 Revenue Trend")
    revenue_chart = SimpleVisualizations.create_revenue_chart(df)
    st.plotly_chart(revenue_chart, use_container_width=True)
    
    # Data overview
    st.subheader("📋 Data Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Dataset Info:**")
        st.write(f"• Total records: {len(df):,}")
        st.write(f"• Columns: {len(df.columns)}")
        st.write(f"• Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    with col2:
        st.write("**Available Columns:**")
        for col in df.columns[:10]:  # Show first 10 columns
            st.write(f"• {col}")
        if len(df.columns) > 10:
            st.write(f"... and {len(df.columns) - 10} more")
    
    # Sample data
    if st.checkbox("🔍 Show Sample Data"):
        st.dataframe(df.head(10), use_container_width=True)
    
    # Export data
    if st.button("📥 Export Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=f"truemedix_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
