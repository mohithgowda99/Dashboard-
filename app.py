import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="Truemedix Billing Dashboard", layout="wide")

# Load data function
@st.cache_data
def load_data(uploaded_file):
    try:
        df = pd.read_excel(uploaded_file)
        
        # Clean the data - remove summary rows
        df = df[df['Lab Name'].notna() | df['Patient Name'].notna()]
        df = df[df['Bill ID'].notna()]
        
        # Convert Date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create helper columns
        df['Month'] = df['Date'].dt.strftime('%Y-%m')
        df['Day'] = df['Date'].dt.date
        df['High_Value_Test'] = df['Gross'] >= 999
        
        # Clean marketing person columns
        df['Zone'] = df['Marketing Person(Organisation)'].fillna('Unknown')
        df['Salesperson'] = df['Billed By'].fillna('Unknown')
        df['Branch'] = df['Branch Name'].fillna('Main Branch')
        
        return df
    except Exception as e:
        st.error(f"Error loading  {str(e)}")
        return None

# Main dashboard
st.title("🏥 Truemedix Billing Dashboard")
st.markdown("**Day-wise Billing Analysis Dashboard**")
st.markdown("---")

# File upload
uploaded_file = st.file_uploader(
    "Upload your Truemedix billing Excel file", 
    type=['xlsx', 'xls'],
    help="Upload the day-wise billing report Excel file"
)

if uploaded_file is not None:
    df = load_data(uploaded_file)
    
    if df is not None and not df.empty:
        st.success(f"✅ Data loaded successfully! {len(df):,} billing records found.")
        
        # Sidebar filters
        st.sidebar.header("🔍 Filters")
        
        # Date range filter
        min_date = df['Date'].min().date()
        max_date = df['Date'].max().date()
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        with col2:
            end_date = st.date_input("End Date", value=max_date, min_value=min_date, max_value=max_date)
        
        # Other filters
        zones = ['All'] + sorted(df['Zone'].unique().tolist())
        selected_zone = st.sidebar.selectbox("Zone", zones)
        
        branches = ['All'] + sorted(df['Branch'].unique().tolist())
        selected_branch = st.sidebar.selectbox("Branch", branches)
        
        salespersons = ['All'] + sorted(df['Salesperson'].unique().tolist())
        selected_salesperson = st.sidebar.selectbox("Salesperson", salespersons)
        
        high_value_only = st.sidebar.checkbox("High Value Tests Only (₹999+)")
        
        # Apply filters
        filtered_df = df[
            (df['Date'].dt.date >= start_date) & 
            (df['Date'].dt.date <= end_date)
        ]
        
        if selected_zone != 'All':
            filtered_df = filtered_df[filtered_df['Zone'] == selected_zone]
        if selected_branch != 'All':
            filtered_df = filtered_df[filtered_df['Branch'] == selected_branch]
        if selected_salesperson != 'All':
            filtered_df = filtered_df[filtered_df['Salesperson'] == selected_salesperson]
        if high_value_only:
            filtered_df = filtered_df[filtered_df['High_Value_Test']]
        
        # Main metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        total_gross = filtered_df['Gross'].sum()
        total_net = filtered_df['Net'].sum()
        total_paid = filtered_df['Paid'].sum()
        total_due = filtered_df['Due'].sum()
        high_value_count = filtered_df['High_Value_Test'].sum()
        
        with col1:
            st.metric("Total Gross Revenue", f"₹{total_gross:,.0f}")
        with col2:
            st.metric("Total Net Revenue", f"₹{total_net:,.0f}")
        with col3:
            st.metric("Total Paid", f"₹{total_paid:,.0f}")
        with col4:
            st.metric("Total Due", f"₹{total_due:,.0f}")
        with col5:
            st.metric("High Value Tests", f"{high_value_count:,}")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Daily Analysis", "💰 Revenue Breakdown", "👥 Team Performance", "📈 Trends"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                # Daily revenue trend
                daily_revenue = filtered_df.groupby('Day').agg({
                    'Gross': 'sum',
                    'Net': 'sum',
                    'Paid': 'sum',
                    'Bill ID': 'count'
                }).reset_index()
                
                fig = px.line(daily_revenue, x='Day', y='Net', 
                            title="Daily Net Revenue Trend",
                            labels={'Net': 'Net Revenue (₹)', 'Day': 'Date'})
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Daily test count
                fig = px.bar(daily_revenue, x='Day', y='Bill ID',
                           title="Daily Test Count",
                           labels={'Bill ID': 'Number of Tests', 'Day': 'Date'})
                st.plotly_chart(fig, use_container_width=True)
            
            # High value tests analysis
            st.subheader("High Value Tests Analysis (₹999+)")
            col1, col2 = st.columns(2)
            
            with col1:
                high_value_daily = filtered_df[filtered_df['High_Value_Test']].groupby('Day').size().reset_index()
                high_value_daily.columns = ['Day', 'Count']
                
                fig = px.bar(high_value_daily, x='Day', y='Count',
                           title="Daily High Value Tests Count",
                           labels={'Count': 'Number of Tests ≥ ₹999'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Revenue distribution
                revenue_ranges = pd.cut(filtered_df['Gross'], 
                                      bins=[0, 100, 500, 999, 2000, float('inf')],
                                      labels=['₹0-100', '₹101-500', '₹501-999', '₹1000-2000', '₹2000+'])
                range_counts = revenue_ranges.value_counts()
                
                fig = px.pie(values=range_counts.values, names=range_counts.index,
                           title="Revenue Distribution by Test Value")
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                # Zone-wise revenue
                zone_revenue = filtered_df.groupby('Zone')['Net'].sum().sort_values(ascending=False).head(10)
                
                fig = px.bar(x=zone_revenue.values, y=zone_revenue.index, orientation='h',
                           title="Top 10 Zones by Net Revenue",
                           labels={'x': 'Net Revenue (₹)', 'y': 'Zone'})
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Branch-wise revenue
                branch_revenue = filtered_df.groupby('Branch')['Net'].sum().sort_values(ascending=False).head(10)
                
                fig = px.bar(x=branch_revenue.values, y=branch_revenue.index, orientation='h',
                           title="Top 10 Branches by Net Revenue",
                           labels={'x': 'Net Revenue (₹)', 'y': 'Branch'})
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                # Salesperson performance
                sales_performance = filtered_df.groupby('Salesperson').agg({
                    'Net': 'sum',
                    'Bill ID': 'count',
                    'High_Value_Test': 'sum'
                }).sort_values('Net', ascending=False).head(10)
                
                fig = px.bar(x=sales_performance.index, y=sales_performance['Net'],
                           title="Top 10 Salesperson by Net Revenue",
                           labels={'y': 'Net Revenue (₹)', 'x': 'Salesperson'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # High value tests by salesperson
                fig = px.bar(x=sales_performance.index, y=sales_performance['High_Value_Test'],
                           title="High Value Tests by Salesperson",
                           labels={'y': 'High Value Tests Count', 'x': 'Salesperson'})
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Collection efficiency
            filtered_df['Collection_Rate'] = (filtered_df['Paid'] / filtered_df['Net'] * 100).fillna(0)
            
            col1, col2 = st.columns(2)
            
            with col1:
                daily_collection = filtered_df.groupby('Day')['Collection_Rate'].mean().reset_index()
                
                fig = px.line(daily_collection, x='Day', y='Collection_Rate',
                            title="Daily Collection Rate Trend",
                            labels={'Collection_Rate': 'Collection Rate (%)'})
                fig.update_traces(line=dict(width=3))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Outstanding analysis
                daily_outstanding = filtered_df.groupby('Day')['Due'].sum().reset_index()
                
                fig = px.bar(daily_outstanding, x='Day', y='Due',
                           title="Daily Outstanding Amount",
                           labels={'Due': 'Outstanding Amount (₹)'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.subheader("📋 Detailed Data Summary")
        
        summary_data = filtered_df.groupby(['Day', 'Zone']).agg({
            'Gross': 'sum',
            'Net': 'sum',
            'Paid': 'sum',
            'Due': 'sum',
            'Bill ID': 'count',
            'High_Value_Test': 'sum'
        }).reset_index()
        
        summary_data.columns = ['Date', 'Zone', 'Gross Revenue', 'Net Revenue', 
                               'Paid Amount', 'Due Amount', 'Total Tests', 'High Value Tests']
        
        # Format currency columns
        currency_cols = ['Gross Revenue', 'Net Revenue', 'Paid Amount', 'Due Amount']
        for col in currency_cols:
            summary_data[col] = summary_data[col].apply(lambda x: f"₹{x:,.0f}")
        
        st.dataframe(summary_data, use_container_width=True)
        
        # Export option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data as CSV",
            data=csv,
            file_name=f"truemedix_billing_data_{start_date}_to_{end_date}.csv",
            mime="text/csv"
        )

else:
    st.info("👆 Please upload your Truemedix billing Excel file to begin analysis")
    
    # Show sample data structure
    st.subheader("Expected Data Structure")
    st.write("Your Excel file should contain columns like:")
    expected_cols = ['Date', 'Lab Name', 'Patient Name', 'Gross', 'Net', 'Paid', 'Due', 
                    'Branch Name', 'Billed By', 'Marketing Person(Organisation)']
    for col in expected_cols:
        st.write(f"• {col}")
