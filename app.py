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
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# ==============================================================================
# STATISTICAL FUNCTIONS (REPLACING SCIPY)
# ==============================================================================

def calculate_zscore(data):
    """Calculate Z-scores without scipy"""
    if len(data) == 0:
        return []
    
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return [0] * len(data)
    
    return [(x - mean) / std for x in data]

def linear_regression(x, y):
    """Simple linear regression without scipy"""
    if len(x) != len(y) or len(x) < 2:
        return 0, 0, 0  # slope, intercept, r_value
    
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi * xi for xi in x)
    sum_y2 = sum(yi * yi for yi in y)
    
    # Calculate slope
    denominator = n * sum_x2 - sum_x * sum_x
    if denominator == 0:
        return 0, np.mean(y), 0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate correlation coefficient
    num = n * sum_xy - sum_x * sum_y
    den = math.sqrt((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y))
    
    if den == 0:
        r_value = 0
    else:
        r_value = num / den
    
    return slope, intercept, r_value

# ==============================================================================
# CONFIGURATION & DATA MODELS
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
# INTELLIGENT DATA PROCESSOR
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
# ANOMALY DETECTION ENGINE (WITHOUT SCIPY)
# ==============================================================================

class AnomalyDetector:
    """Advanced anomaly detection without scipy dependency"""
    
    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
    
    def detect_revenue_anomalies(self, df: pd.DataFrame) -> List[Dict]:
        """Detect revenue anomalies using statistical methods"""
        anomalies = []
        
        if 'Amount' not in df.columns or 'Date' not in df.columns:
            return anomalies
        
        try:
            # Monthly revenue analysis
            monthly_revenue = df.groupby(df['Date'].dt.to_period('M'))['Amount'].sum()
            
            if len(monthly_revenue) < 3:
                return anomalies  # Need at least 3 months for meaningful analysis
            
            # Calculate rolling statistics
            rolling_mean = monthly_revenue.rolling(window=3, min_periods=2).mean()
            rolling_std = monthly_revenue.rolling(window=3, min_periods=2).std()
            
            # Detect outliers using Z-score (manual calculation)
            z_scores = {}
            for period, revenue in monthly_revenue.items():
                mean_val = rolling_mean.get(period, revenue)
                std_val = rolling_std.get(period, 1)
                
                if std_val > 0:
                    z_score = abs(revenue - mean_val) / std_val
                    z_scores[period] = z_score
                    
                    if z_score > 2:  # 2 standard deviations
                        deviation = abs(revenue - mean_val) / (mean_val + 1e-8)
                        
                        if deviation > self.threshold:
                            anomalies.append({
                                'type': 'revenue_anomaly',
                                'period': str(period),
                                'actual_revenue': float(revenue),
                                'expected_revenue': float(mean_val),
                                'deviation_percent': float(deviation * 100),
                                'z_score': float(z_score),
                                'severity': 'high' if deviation > 0.5 else 'medium'
                            })
        
        except Exception as e:
            logger.warning(f"Error in anomaly detection: {e}")
        
        return anomalies

# ==============================================================================
# ADVANCED ANALYTICS ENGINE
# ==============================================================================

class AdvancedAnalytics:
    """Advanced analytics with trend analysis"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
    
    def calculate_trend_analysis(self, dimension: str, months: int = 6) -> Dict:
        """Calculate trend analysis for specified dimension"""
        if dimension not in self.df.columns:
            return {}
        
        try:
            # Filter to last N months
            end_date = self.df['Date'].max()
            start_date = end_date - pd.DateOffset(months=months)
            filtered_df = self.df[self.df['Date'] >= start_date]
            
            # Monthly trends
            monthly_trends = filtered_df.groupby([
                filtered_df['Date'].dt.to_period('M'), 
                dimension
            ])['Amount'].sum().unstack(fill_value=0)
            
            # Calculate growth rates
            growth_rates = monthly_trends.pct_change().mean() * 100
            
            # Calculate trend direction
            trends = {}
            for entity in monthly_trends.columns:
                values = monthly_trends[entity].values
                if len(values) >= 2:
                    # Simple linear trend using our custom function
                    x = list(range(len(values)))
                    slope, intercept, r_value = linear_regression(x, values)
                    
                    trends[entity] = {
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                        'trend_strength': abs(r_value),
                        'growth_rate': float(growth_rates.get(entity, 0)),
                        'total_revenue': float(values.sum()),
                        'avg_monthly': float(values.mean()),
                        'volatility': float(np.std(values) / (np.mean(values) + 1e-8))
                    }
            
            return trends
        
        except Exception as e:
            logger.warning(f"Error in trend analysis: {e}")
            return {}
    
    def calculate_rolling_metrics(self, window: int = 30) -> pd.DataFrame:
        """Calculate rolling averages and metrics"""
        if 'Date' not in self.df.columns:
            return pd.DataFrame()
        
        try:
            daily_metrics = self.df.groupby('Date').agg({
                'Amount': ['sum', 'count', 'mean'],
                'Client': 'nunique',
                'HighValueTest': 'sum'
            }).round(2)
            
            daily_metrics.columns = ['Revenue', 'Transactions', 'AvgTicket', 'UniqueClients', 'HighValueTests']
            
            # Calculate rolling metrics
            for col in daily_metrics.columns:
                daily_metrics[f'{col}_MA{window}'] = daily_metrics[col].rolling(window=window).mean()
                
                # Custom trend calculation
                def calc_trend(series):
                    if len(series) >= 2:
                        x = list(range(len(series)))
                        slope, _, _ = linear_regression(x, series.tolist())
                        return slope
                    return 0
                
                daily_metrics[f'{col}_Trend'] = daily_metrics[col].rolling(window=window).apply(calc_trend)
            
            return daily_metrics.reset_index()
        
        except Exception as e:
            logger.warning(f"Error calculating rolling metrics: {e}")
            return pd.DataFrame()

# ==============================================================================
# VISUALIZATION ENGINE
# ==============================================================================

class AdvancedVisualizations:
    """Advanced visualization components"""
    
    @staticmethod
    def create_trend_comparison_chart(df: pd.DataFrame, dimension: str, months: int = 6) -> go.Figure:
        """Create multi-line trend comparison chart"""
        try:
            end_date = df['Date'].max()
            start_date = end_date - pd.DateOffset(months=months)
            filtered_df = df[df['Date'] >= start_date]
            
            # Get top entities by revenue
            top_entities = filtered_df.groupby(dimension)['Amount'].sum().nlargest(10).index
            
            fig = go.Figure()
            
            for entity in top_entities:
                entity_data = filtered_df[filtered_df[dimension] == entity]
                monthly_revenue = entity_data.groupby(
                    entity_data['Date'].dt.to_period('M')
                )['Amount'].sum()
                
                fig.add_trace(go.Scatter(
                    x=monthly_revenue.index.astype(str),
                    y=monthly_revenue.values,
                    mode='lines+markers',
                    name=str(entity)[:20],  # Limit name length
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title=f'Revenue Trends by {dimension} (Last {months} Months)',
                xaxis_title='Month',
                yaxis_title='Revenue (₹)',
                height=500,
                hovermode='x unified'
            )
            
            return fig
        
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating chart: {str(e)}", x=0.5, y=0.5)
            return fig
    
    @staticmethod
    def create_anomaly_alert_chart(anomalies: List[Dict]) -> go.Figure:
        """Create anomaly visualization chart"""
        try:
            if not anomalies:
                fig = go.Figure()
                fig.add_annotation(
                    text="No anomalies detected",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                return fig
            
            # Separate by type
            revenue_anomalies = [a for a in anomalies if a['type'] == 'revenue_anomaly']
            
            fig = go.Figure()
            
            if revenue_anomalies:
                periods = [a['period'] for a in revenue_anomalies]
                actual = [a['actual_revenue'] for a in revenue_anomalies]
                expected = [a['expected_revenue'] for a in revenue_anomalies]
                
                fig.add_trace(go.Bar(
                    x=periods,
                    y=actual,
                    name='Actual Revenue',
                    marker_color='red'
                ))
                
                fig.add_trace(go.Bar(
                    x=periods,
                    y=expected,
                    name='Expected Revenue',
                    marker_color='lightblue'
                ))
            
            fig.update_layout(
                title='Revenue Anomalies Detection',
                xaxis_title='Period',
                yaxis_title='Revenue (₹)',
                height=400,
                barmode='group'
            )
            
            return fig
        
        except Exception as e:
            fig = go.Figure()
            fig.add_annotation(text=f"Error creating anomaly chart: {str(e)}", x=0.5, y=0.5)
            return fig
    
    @staticmethod
    def create_kpi_dashboard(df: pd.DataFrame, selected_period: Optional[str] = None) -> Dict[str, Any]:
        """Create comprehensive KPI dashboard"""
        try:
            kpis = {}
            
            # Filter data if period selected
            if selected_period and 'Date' in df.columns:
                period_df = df[df['Date'].dt.to_period('M').astype(str) == selected_period]
            else:
                period_df = df
            
            # Calculate KPIs
            kpis['total_revenue'] = float(period_df['Amount'].sum()) if 'Amount' in period_df.columns else 0
            kpis['total_transactions'] = len(period_df)
            kpis['avg_transaction'] = float(period_df['Amount'].mean()) if 'Amount' in period_df.columns else 0
            kpis['unique_clients'] = period_df['Client'].nunique() if 'Client' in period_df.columns else 0
            kpis['high_value_tests'] = period_df['HighValueTest'].sum() if 'HighValueTest' in period_df.columns else 0
            
            if 'Branch' in period_df.columns and len(period_df) > 0:
                top_branch_data = period_df.groupby('Branch')['Amount'].sum()
                kpis['top_branch'] = top_branch_data.idxmax() if not top_branch_data.empty else 'N/A'
            else:
                kpis['top_branch'] = 'N/A'
            
            # Calculate growth rates
            kpis['revenue_growth'] = 0
            if 'Date' in df.columns and len(df) > 0:
                current_month = df['Date'].max().to_period('M')
                last_month = current_month - 1
                
                current_revenue = df[df['Date'].dt.to_period('M') == current_month]['Amount'].sum()
                last_revenue = df[df['Date'].dt.to_period('M') == last_month]['Amount'].sum()
                
                if last_revenue > 0:
                    kpis['revenue_growth'] = float(((current_revenue - last_revenue) / last_revenue) * 100)
            
            return kpis
        
        except Exception as e:
            logger.warning(f"Error creating KPI dashboard: {e}")
            return {
                'total_revenue': 0,
                'total_transactions': 0,
                'avg_transaction': 0,
                'unique_clients': 0,
                'high_value_tests': 0,
                'top_branch': 'N/A',
                'revenue_growth': 0
            }

# ==============================================================================
# MAIN STREAMLIT APPLICATION
# ==============================================================================

def main():
    st.set_page_config(
        page_title="Advanced Truemedix Analytics",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .anomaly-alert {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .insight-box {
        background-color: #e8f5e8;
        border-left: 4px solid #28a745;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🧪 Advanced Truemedix Analytics Dashboard")
    st.markdown("*Intelligent data processing with anomaly detection and trend analysis*")
    
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
        if st.button("🔄 Refresh All Data", type="primary"):
            with st.spinner("Processing all data files..."):
                try:
                    processor = AdvancedDataProcessor(config)
                    master_data, stats = processor.load_all_data()
                    
                    st.session_state.master_data = master_data
                    st.session_state.processing_stats = stats
                    st.session_state.last_refresh = datetime.now()
                    
                    if len(master_data) > 0:
                        st.success(f"✅ Processed {stats.files_processed} files, loaded {stats.records_loaded:,} records")
                    else:
                        st.warning("⚠️ No data loaded. Check if files exist in folders")
                        
                except Exception as e:
                    st.error(f"❌ Error loading  {str(e)}")
                    logger.error(f"Data loading error: {e}")
        
        # Show processing stats
        if st.session_state.processing_stats:
            stats = st.session_state.processing_stats
            st.subheader("📊 Processing Statistics")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Files Processed", stats.files_processed)
                st.metric("Records Loaded", f"{stats.records_loaded:,}")
            with col2:
                st.metric("Duplicates Removed", f"{stats.records_deduplicated:,}")
                st.metric("Fallbacks Applied", len(stats.fallbacks_applied))
            
            if stats.fallbacks_applied:
                with st.expander("⚠️ Fallbacks Applied"):
                    for fallback in stats.fallbacks_applied:
                        st.warning(fallback)
        
        st.divider()
        
        # Configuration
        st.header("⚙️ Configuration")
        
        # Anomaly detection threshold
        anomaly_threshold = st.slider(
            "Anomaly Detection Threshold (%)",
            min_value=10,
            max_value=50,
            value=25,
            help="Percentage deviation to flag as anomaly"
        )
        config.anomaly_threshold = anomaly_threshold / 100
        
        # Specialty test threshold
        specialty_threshold = st.number_input(
            "High-Value Test Threshold (₹)",
            min_value=500,
            max_value=2000,
            value=999,
            step=100
        )
        config.specialty_threshold = specialty_threshold
        
        # Time period for analysis
        analysis_months = st.selectbox(
            "Analysis Period (Months)",
            options=[3, 6, 12],
            index=1
        )
    
    # Main content
    if st.session_state.master_data is None or len(st.session_state.master_data) == 0:
        st.info("👆 Click 'Refresh All Data' to load and process your files")
        
        # Show folder structure help
        st.subheader("📁 Expected Folder Structure")
        st.code("""
        project/
        ├── historical/          # Past 6 months data files
        │   ├── jan_billing.xlsx
        │   ├── feb_billing.csv
        │   └── ...
        ├── daily/              # Daily upload files
        │   ├── today_data.xlsx
        │   └── ...
        └── app.py              # This script
        """)
        
        st.subheader("🔧 Supported Features")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Intelligent Data Processing:**
            - ✅ Auto-detects column formats
            - ✅ Handles missing columns gracefully
            - ✅ Deduplicates records automatically
            - ✅ Supports CSV, Excel formats
            """)
        with col2:
            st.markdown("""
            **Advanced Analytics:**
            - ✅ Anomaly detection
            - ✅ Trend analysis
            - ✅ Rolling averages
            - ✅ KPI monitoring
            """)
        return
    
    # Get data
    df = st.session_state.master_data
    
    # Initialize analytics engines
    anomaly_detector = AnomalyDetector(config.anomaly_threshold)
    analytics = AdvancedAnalytics(df)
    
    # Detect anomalies
    revenue_anomalies = anomaly_detector.detect_revenue_anomalies(df)
    all_anomalies = revenue_anomalies
    
    # Show anomaly alerts
    if all_anomalies:
        st.markdown('<div class="anomaly-alert">', unsafe_allow_html=True)
        st.warning(f"🚨 {len(all_anomalies)} anomalies detected!")
        
        for anomaly in all_anomalies[:3]:  # Show top 3
            if anomaly['type'] == 'revenue_anomaly':
                st.write(f"📊 Revenue anomaly in {anomaly['period']}: "
                        f"{anomaly['deviation_percent']:.1f}% deviation from expected")
        
        if len(all_anomalies) > 3:
            st.write(f"... and {len(all_anomalies) - 3} more anomalies")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # KPI Dashboard
    st.subheader("📊 Key Performance Indicators")
    
    # Period selector
    available_periods = []
    if 'Date' in df.columns:
        available_periods = sorted(df['Date'].dt.to_period('M').astype(str).unique(), reverse=True)
    
    selected_period = None
    if available_periods:
        selected_period = st.selectbox("Select Period", options=["Current Month"] + available_periods)
        if selected_period == "Current Month":
            selected_period = None
    
    # Calculate KPIs
    kpis = AdvancedVisualizations.create_kpi_dashboard(df, selected_period)
    
    # Display KPIs
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "💰 Total Revenue",
            f"₹{kpis['total_revenue']:,.0f}",
            delta=f"{kpis['revenue_growth']:+.1f}%" if kpis['revenue_growth'] != 0 else None
        )
    
    with col2:
        st.metric("🧪 Transactions", f"{kpis['total_transactions']:,}")
    
    with col3:
        st.metric("👥 Unique Clients", f"{kpis['unique_clients']:,}")
    
    with col4:
        st.metric("⭐ High-Value Tests", f"{kpis['high_value_tests']:,}")
    
    with col5:
        st.metric("🏆 Top Branch", kpis['top_branch'])
    
    # Main analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Trend Analysis", 
        "🔍 Anomaly Detection", 
        "📊 Rolling Metrics", 
        "📋 Data Overview"
    ])
    
    with tab1:
        st.subheader("📈 Trend Analysis")
        
        # Dimension selector
        available_dimensions = [col for col in ['Branch', 'Client', 'Salesperson'] if col in df.columns]
        
        if available_dimensions:
            selected_dimension = st.selectbox("Select Dimension for Trend Analysis", available_dimensions)
            
            # Create trend chart
            trend_chart = AdvancedVisualizations.create_trend_comparison_chart(
                df, selected_dimension, analysis_months
            )
            st.plotly_chart(trend_chart, use_container_width=True)
            
            # Trend statistics
            trends = analytics.calculate_trend_analysis(selected_dimension, analysis_months)
            
            if trends:
                st.subheader(f"📊 {selected_dimension} Performance Summary")
                
                # Create summary table
                trend_summary = []
                for entity, data in trends.items():
                    trend_summary.append({
                        selected_dimension: entity,
                        'Trend': data['trend_direction'].title(),
                        'Growth Rate (%)': f"{data['growth_rate']:.1f}%",
                        'Total Revenue': f"₹{data['total_revenue']:,.0f}",
                        'Monthly Average': f"₹{data['avg_monthly']:,.0f}",
                        'Volatility': f"{data['volatility']:.2f}"
                    })
                
                if trend_summary:
                    trend_df = pd.DataFrame(trend_summary).head(10)
                    st.dataframe(trend_df, use_container_width=True)
        else:
            st.info("No dimensional data available for trend analysis")
    
    with tab2:
        st.subheader("🔍 Anomaly Detection")
        
        # Anomaly visualization
        anomaly_chart = AdvancedVisualizations.create_anomaly_alert_chart(all_anomalies)
        st.plotly_chart(anomaly_chart, use_container_width=True)
        
        # Detailed anomaly table
        if all_anomalies:
            st.subheader("📋 Anomaly Details")
            
            anomaly_details = []
            for anomaly in all_anomalies:
                if anomaly['type'] == 'revenue_anomaly':
                    anomaly_details.append({
                        'Type': 'Revenue',
                        'Period': anomaly['period'],
                        'Actual': f"₹{anomaly['actual_revenue']:,.0f}",
                        'Expected': f"₹{anomaly['expected_revenue']:,.0f}",
                        'Deviation': f"{anomaly['deviation_percent']:.1f}%",
                        'Severity': anomaly['severity'].title()
                    })
            
            if anomaly_details:
                anomaly_df = pd.DataFrame(anomaly_details)
                st.dataframe(anomaly_df, use_container_width=True)
        else:
            st.success("✅ No anomalies detected in the current analysis period")
    
    with tab3:
        st.subheader("📊 Rolling Metrics")
        
        # Rolling window selector
        window_days = st.slider("Rolling Window (Days)", 7, 90, 30)
        
        # Calculate rolling metrics
        rolling_metrics = analytics.calculate_rolling_metrics(window_days)
        
        if not rolling_metrics.empty:
            # Revenue trend with rolling average
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=rolling_metrics['Date'],
                y=rolling_metrics['Revenue'],
                mode='lines',
                name='Daily Revenue',
                line=dict(color='lightblue', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=rolling_metrics['Date'],
                y=rolling_metrics[f'Revenue_MA{window_days}'],
                mode='lines',
                name=f'{window_days}-Day Moving Average',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title=f'Revenue Trend with {window_days}-Day Moving Average',
                xaxis_title='Date',
                yaxis_title='Revenue (₹)',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show rolling metrics table
            display_columns = ['Date', 'Revenue', f'Revenue_MA{window_days}', 
                             'Transactions', 'UniqueClients', 'HighValueTests']
            available_columns = [col for col in display_columns if col in rolling_metrics.columns]
            
            if available_columns:
                st.dataframe(
                    rolling_metrics[available_columns].tail(30),
                    use_container_width=True
                )
        else:
            st.info("No data available for rolling metrics calculation")
    
    with tab4:
        st.subheader("📋 Data Overview")
        
        # Data summary
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
            if 'Date' in df.columns:
                st.metric("Date Range", f"{df['Date'].min().date()} to {df['Date'].max().date()}")
        
        with col2:
            st.metric("Columns", len(df.columns))
            st.metric("Data Quality", f"{(1 - df.isnull().sum().sum() / df.size) * 100:.1f}%")
        
        with col3:
            if 'SourceFile' in df.columns:
                st.metric("File Sources", df['SourceFile'].nunique())
            if st.session_state.last_refresh:
                st.metric("Last Refresh", st.session_state.last_refresh.strftime("%H:%M:%S"))
        
        # Column information
        st.subheader("📊 Column Information")
        
        column_info = []
        for col in df.columns:
            column_info.append({
                'Column': col,
                'Type': str(df[col].dtype),
                'Non-Null Count': f"{df[col].count():,}",
                'Null Count': f"{df[col].isnull().sum():,}",
                'Unique Values': f"{df[col].nunique():,}"
            })
        
        column_df = pd.DataFrame(column_info)
        st.dataframe(column_df, use_container_width=True)
        
        # Sample data
        st.subheader("🔍 Sample Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Export functionality
        st.subheader("📥 Export Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Export Master Dataset"):
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"truemedix_master_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📈 Export KPIs"):
                kpi_df = pd.DataFrame([kpis])
                csv = kpi_df.to_csv(index=False)
                st.download_button(
                    label="Download KPIs CSV",
                    data=csv,
                    file_name=f"truemedix_kpis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv"
                )

if __name__ == "__main__":
    main()
