import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import random

st.set_page_config(page_title="Lab Dashboard", layout="wide")

# Generate enhanced sample data
@st.cache_data
def generate_enhanced_sample_data():
    """Generate comprehensive lab test data"""
    branches = ['Main Branch', 'North Branch', 'South Branch', 'East Branch', 'West Branch']
    salespersons = ['John Smith', 'Sarah Johnson', 'Michael Brown', 'Emily Davis', 'David Wilson', 'Lisa Anderson']
    clients = ['Apollo Hospital', 'Max Healthcare', 'Fortis Hospital', 'City Clinic', 'Metro Medical', 
               'Prime Healthcare', 'Unity Hospital', 'Care Center', 'Health Plus', 'Medical Plaza']
    specialties = ['Hematology', 'Biochemistry', 'Microbiology', 'Pathology', 'Radiology', 'Cardiology', 'Oncology']
    payment_methods = ['Cash', 'Card', 'Cheque', 'UPI', 'Net Banking', 'Insurance']
    referral_sources = ['Direct', 'Doctor Referral', 'Hospital Referral', 'Online', 'Marketing Campaign', 'Word of Mouth']
    referral_types = ['Referral', 'Organisation Referral', 'Direct']
    
    tests = {
        'Hematology': ['CBC', 'ESR', 'Hemoglobin', 'Platelet Count'],
        'Biochemistry': ['Lipid Profile', 'Liver Function', 'Kidney Function', 'Glucose Test'],
        'Microbiology': ['Blood Culture', 'Urine Culture', 'Stool Culture'],
        'Pathology': ['Biopsy', 'Cytology', 'Histopathology'],
        'Radiology': ['X-Ray', 'CT Scan', 'MRI', 'Ultrasound'],
        'Cardiology': ['ECG', 'Echo', 'Stress Test'],
        'Oncology': ['Tumor Markers', 'CEA', 'PSA']
    }
    
    data = []
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 12, 31)
    
    for i in range(3000):
        invoice_date = start_date + timedelta(days=random.randint(0, (end_date - start_date).days))
        specialty = random.choice(specialties)
        test_name = random.choice(tests[specialty])
        client_name = random.choice(clients)
        
        # Calculate pricing
        mrp_base = random.randint(1000, 4000) if specialty in ['Radiology', 'Oncology'] else random.randint(200, 1700)
        test_mrp = mrp_base
        
        discount_percent = random.uniform(0, 25)
        discount_amount = int(test_mrp * discount_percent / 100)
        billed_amount = test_mrp - discount_amount
        
        net_revenue = int(billed_amount * random.uniform(0.85, 0.98))
        collection_rate = random.uniform(0.85, 1.0)
        collected_amount = int(billed_amount * collection_rate)
        outstanding_amount = billed_amount - collected_amount
        
        data.append({
            'InvoiceDate': invoice_date.strftime('%Y-%m-%d'),
            'Branch': random.choice(branches),
            'Salesperson': random.choice(salespersons),
            'ClientName': client_name,
            'TestName': test_name,
            'Specialty': specialty,
            'TestMRP': test_mrp,
            'DiscountAmount': discount_amount,
            'DiscountPercent': discount_percent,
            'BilledAmount': billed_amount,
            'NetRevenue': net_revenue,
            'CollectedAmount': collected_amount,
            'OutstandingAmount': outstanding_amount,
            'PaymentMethod': random.choice(payment_methods),
            'ReferralSource': random.choice(referral_sources),
            'ReferralType': random.choice(referral_types),
            'InvoiceID': f"LAB{invoice_date.year}{invoice_date.month:02d}{random.randint(1000, 9999)}"
        })
    
    return pd.DataFrame(data).sort_values('InvoiceDate')

# File upload section
st.sidebar.markdown("## 📁 Data Source")
upload_option = st.sidebar.radio("Choose data source:", ["Upload Excel/CSV File", "Use Sample Data"])

if upload_option == "Upload Excel/CSV File":
    uploaded_file = st.sidebar.file_uploader(
        "Upload your Excel or CSV file", 
        type=['xlsx', 'xls', 'csv'],
        help="Upload your lab test data in Excel or CSV format"
    )
    
    if uploaded_file is not None:
        try:
            # Check file type and read accordingly
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                # For Excel files, you can specify sheet name
                sheet_names = pd.ExcelFile(uploaded_file).sheet_names
                if len(sheet_names) > 1:
                    selected_sheet = st.sidebar.selectbox("Select sheet:", sheet_names)
                    data = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                else:
                    data = pd.read_excel(uploaded_file)
            
            st.sidebar.success(f"✅ File uploaded successfully! {len(data)} records loaded.")
            
            # Show data preview
            with st.sidebar.expander("📋 Data Preview"):
                st.write(f"Columns: {list(data.columns)}")
                st.write(f"Rows: {len(data)}")
                
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")
            st.sidebar.info("Using sample data instead...")
            data = generate_enhanced_sample_data()
    else:
        st.sidebar.info("👆 Please upload a file to proceed")
        data = generate_enhanced_sample_data()
else:
    # Use sample data
    if 'data' not in st.session_state:
        st.session_state.data = generate_enhanced_sample_data()
    data = st.session_state.data
    st.sidebar.info("📊 Using generated sample data")

# Column mapping section
if uploaded_file is not None and data is not None:
    st.sidebar.markdown("### 📋 Column Mapping")
    st.sidebar.write("Map your Excel columns to required fields:")
    
    required_columns = {
        'InvoiceDate': 'Invoice Date',
        'Branch': 'Branch/Location', 
        'Salesperson': 'Sales Person',
        'ClientName': 'Client Name',
        'TestName': 'Test Name',
        'Specialty': 'Test Category/Specialty',
        'TestMRP': 'Test MRP/Price',
        'BilledAmount': 'Billed Amount',
        'NetRevenue': 'Net Revenue',
        'PaymentMethod': 'Payment Method'
    }
    
    # Create mapping
    column_mapping = {}
    available_columns = ['None'] + list(data.columns)
    
    for required_col, description in required_columns.items():
        mapped_col = st.sidebar.selectbox(
            f"{description}:",
            available_columns,
            key=f"map_{required_col}"
        )
        if mapped_col != 'None':
            column_mapping[required_col] = mapped_col
    
    # Apply mapping to rename columns
    if column_mapping:
        # Rename columns to match expected names
        data = data.rename(columns={v: k for k, v in column_mapping.items()})
        
        # Fill missing required columns with defaults
        for req_col in required_columns.keys():
            if req_col not in data.columns:
                if req_col in ['TestMRP', 'BilledAmount', 'NetRevenue']:
                    data[req_col] = 0
                else:
                    data[req_col] = 'Unknown'


# Sidebar filters
st.sidebar.title("🔬 Lab Dashboard Filters")

# Date range filter
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input("Start Date", value=datetime(2024, 1, 1))
with col2:
    end_date = st.date_input("End Date", value=datetime(2024, 12, 31))

# Other filters
branch_filter = st.sidebar.selectbox("Branch", ["All"] + sorted(data['Branch'].unique().tolist()))
salesperson_filter = st.sidebar.selectbox("Salesperson", ["All"] + sorted(data['Salesperson'].unique().tolist()))
specialty_filter = st.sidebar.selectbox("Specialty", ["All"] + sorted(data['Specialty'].unique().tolist()))
high_value_only = st.sidebar.checkbox("High Value Tests Only (₹999+)")

# Apply filters
filtered_data = data.copy()
filtered_data['InvoiceDate'] = pd.to_datetime(filtered_data['InvoiceDate'])
filtered_data = filtered_data[
    (filtered_data['InvoiceDate'] >= pd.to_datetime(start_date)) & 
    (filtered_data['InvoiceDate'] <= pd.to_datetime(end_date))
]

if branch_filter != "All":
    filtered_data = filtered_data[filtered_data['Branch'] == branch_filter]
if salesperson_filter != "All":
    filtered_data = filtered_data[filtered_data['Salesperson'] == salesperson_filter]
if specialty_filter != "All":
    filtered_data = filtered_data[filtered_data['Specialty'] == specialty_filter]
if high_value_only:
    filtered_data = filtered_data[filtered_data['NetRevenue'] >= 999]

# Main dashboard
st.title("🏥 Laboratory Management Dashboard")
st.markdown("---")

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "💰 Financial", "📈 Analytics", "👥 Clients"])

with tab1:
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = filtered_data['NetRevenue'].sum()
    total_tests = len(filtered_data)
    high_value_tests = len(filtered_data[filtered_data['NetRevenue'] >= 999])
    unique_clients = filtered_data['ClientName'].nunique()
    
    with col1:
        st.metric("Total Revenue", f"₹{total_revenue:,.0f}")
    with col2:
        st.metric("Total Tests", f"{total_tests:,}")
    with col3:
        st.metric("High Value Tests", f"{high_value_tests:,}")
    with col4:
        st.metric("Unique Clients", f"{unique_clients:,}")
    
    # Monthly revenue trend
    monthly_revenue = filtered_data.groupby(filtered_data['InvoiceDate'].dt.to_period('M'))['NetRevenue'].sum()
    
    fig_trend = px.line(
        x=monthly_revenue.index.astype(str), 
        y=monthly_revenue.values,
        title="Monthly Revenue Trend",
        labels={'x': 'Month', 'y': 'Revenue (₹)'}
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    
    # Branch performance
    branch_revenue = filtered_data.groupby('Branch')['NetRevenue'].sum().sort_values(ascending=True)
    
    fig_branch = px.bar(
        x=branch_revenue.values,
        y=branch_revenue.index,
        orientation='h',
        title="Revenue by Branch",
        labels={'x': 'Revenue (₹)', 'y': 'Branch'}
    )
    st.plotly_chart(fig_branch, use_container_width=True)

with tab2:
    col1, col2, col3, col4 = st.columns(4)
    
    total_billed = filtered_data['BilledAmount'].sum()
    total_collected = filtered_data['CollectedAmount'].sum()
    total_outstanding = filtered_data['OutstandingAmount'].sum()
    avg_discount = filtered_data['DiscountPercent'].mean()
    collection_rate = (total_collected / total_billed * 100) if total_billed > 0 else 0
    
    with col1:
        st.metric("Total Billed", f"₹{total_billed:,.0f}")
    with col2:
        st.metric("Total Collected", f"₹{total_collected:,.0f}")
    with col3:
        st.metric("Outstanding", f"₹{total_outstanding:,.0f}")
    with col4:
        st.metric("Collection Rate", f"{collection_rate:.1f}%")
    
    # Payment method breakdown
    payment_data = filtered_data.groupby('PaymentMethod')['NetRevenue'].sum().sort_values(ascending=False)
    
    fig_payment = px.pie(
        values=payment_data.values,
        names=payment_data.index,
        title="Revenue by Payment Method"
    )
    st.plotly_chart(fig_payment, use_container_width=True)
    
    # Discount analysis
    discount_data = filtered_data.groupby('Specialty')['DiscountPercent'].mean().sort_values(ascending=False)
    
    fig_discount = px.bar(
        x=discount_data.index,
        y=discount_data.values,
        title="Average Discount by Specialty",
        labels={'x': 'Specialty', 'y': 'Discount %'}
    )
    st.plotly_chart(fig_discount, use_container_width=True)

with tab3:
    # Test performance
    test_performance = filtered_data.groupby('TestName').agg({
        'NetRevenue': 'sum',
        'InvoiceID': 'count'
    }).rename(columns={'InvoiceID': 'Count'}).sort_values('NetRevenue', ascending=False).head(10)
    
    fig_tests = px.bar(
        test_performance,
        x=test_performance.index,
        y='NetRevenue',
        title="Top 10 Tests by Revenue",
        labels={'x': 'Test Name', 'y': 'Revenue (₹)'}
    )
    fig_tests.update_xaxes(tickangle=45)
    st.plotly_chart(fig_tests, use_container_width=True)
    
    # Salesperson performance
    sales_performance = filtered_data.groupby('Salesperson').agg({
        'NetRevenue': 'sum',
        'ClientName': 'nunique',
        'InvoiceID': 'count'
    }).rename(columns={'ClientName': 'UniqueClients', 'InvoiceID': 'TotalTests'})
    
    st.subheader("Salesperson Performance")
    st.dataframe(sales_performance.style.format({'NetRevenue': '₹{:,.0f}'}))

with tab4:
    # Client analysis
    client_performance = filtered_data.groupby('ClientName').agg({
        'NetRevenue': 'sum',
        'InvoiceID': 'count',
        'InvoiceDate': ['min', 'max']
    })
    client_performance.columns = ['TotalRevenue', 'TotalTests', 'FirstVisit', 'LastVisit']
    client_performance = client_performance.sort_values('TotalRevenue', ascending=False).head(20)
    
    st.subheader("Top 20 Clients by Revenue")
    st.dataframe(
        client_performance.style.format({
            'TotalRevenue': '₹{:,.0f}',
            'FirstVisit': lambda x: x.strftime('%Y-%m-%d'),
            'LastVisit': lambda x: x.strftime('%Y-%m-%d')
        })
    )
    
    # Referral source analysis
    referral_data = filtered_data.groupby(['ReferralType', 'ReferralSource'])['NetRevenue'].sum().reset_index()
    
    fig_referral = px.treemap(
        referral_data,
        path=['ReferralType', 'ReferralSource'],
        values='NetRevenue',
        title="Revenue by Referral Source"
    )
    st.plotly_chart(fig_referral, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("**Dashboard Summary:**")
col1, col2, col3 = st.columns(3)
with col1:
    st.write(f"📅 Date Range: {start_date} to {end_date}")
with col2:
    st.write(f"📊 Records Shown: {len(filtered_data):,}")
with col3:
    st.write(f"💰 Total Revenue: ₹{total_revenue:,.0f}")
