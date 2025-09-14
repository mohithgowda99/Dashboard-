import React, { useState, useEffect, useMemo } from ‘react’;
import { BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, Area, AreaChart, ComposedChart } from ‘recharts’;
import { Upload, Filter, TrendingUp, Users, DollarSign, Activity, Download, Calendar, Search, AlertTriangle, Target, Percent, CreditCard, RefreshCw, Clock } from ‘lucide-react’;

// Enhanced sample data generation with new fields
const generateEnhancedSampleData = () => {
const branches = [‘Main Branch’, ‘North Branch’, ‘South Branch’, ‘East Branch’, ‘West Branch’];
const salespersons = [‘John Smith’, ‘Sarah Johnson’, ‘Michael Brown’, ‘Emily Davis’, ‘David Wilson’, ‘Lisa Anderson’];
const clients = [‘Apollo Hospital’, ‘Max Healthcare’, ‘Fortis Hospital’, ‘City Clinic’, ‘Metro Medical’, ‘Prime Healthcare’, ‘Unity Hospital’, ‘Care Center’, ‘Health Plus’, ‘Medical Plaza’];
const specialties = [‘Hematology’, ‘Biochemistry’, ‘Microbiology’, ‘Pathology’, ‘Radiology’, ‘Cardiology’, ‘Oncology’];
const paymentMethods = [‘Cash’, ‘Card’, ‘Cheque’, ‘UPI’, ‘Net Banking’, ‘Insurance’];
const referralSources = [‘Direct’, ‘Doctor Referral’, ‘Hospital Referral’, ‘Online’, ‘Marketing Campaign’, ‘Word of Mouth’];
const referralTypes = [‘Referral’, ‘Organisation Referral’, ‘Direct’];

const tests = {
‘Hematology’: [‘CBC’, ‘ESR’, ‘Hemoglobin’, ‘Platelet Count’],
‘Biochemistry’: [‘Lipid Profile’, ‘Liver Function’, ‘Kidney Function’, ‘Glucose Test’],
‘Microbiology’: [‘Blood Culture’, ‘Urine Culture’, ‘Stool Culture’],
‘Pathology’: [‘Biopsy’, ‘Cytology’, ‘Histopathology’],
‘Radiology’: [‘X-Ray’, ‘CT Scan’, ‘MRI’, ‘Ultrasound’],
‘Cardiology’: [‘ECG’, ‘Echo’, ‘Stress Test’],
‘Oncology’: [‘Tumor Markers’, ‘CEA’, ‘PSA’]
};

const data = [];
const clientHistory = new Map();
const startDate = new Date(‘2024-01-01’);
const endDate = new Date(‘2024-12-31’);

for (let i = 0; i < 3000; i++) {
const invoiceDate = new Date(startDate.getTime() + Math.random() * (endDate.getTime() - startDate.getTime()));
const specialty = specialties[Math.floor(Math.random() * specialties.length)];
const testName = tests[specialty][Math.floor(Math.random() * tests[specialty].length)];
const clientName = clients[Math.floor(Math.random() * clients.length)];

```
// Track client visit history
if (!clientHistory.has(clientName)) {
  clientHistory.set(clientName, []);
}
clientHistory.get(clientName).push(invoiceDate);

const mrpBase = specialty === 'Radiology' || specialty === 'Oncology' ? 
               Math.random() * 3000 + 1000 : Math.random() * 1500 + 200;
const testMRP = Math.round(mrpBase);

// Calculate discount (0-25%)
const discountPercent = Math.random() * 0.25;
const discountAmount = testMRP * discountPercent;
const billedAmount = Math.round(testMRP - discountAmount);

// Net revenue after costs
const netRevenue = Math.round(billedAmount * (0.85 + Math.random() * 0.13));

// Collection amount (85-100% of billed amount)
const collectionRate = 0.85 + Math.random() * 0.15;
const collectedAmount = Math.round(billedAmount * collectionRate);
const outstandingAmount = billedAmount - collectedAmount;

// Client added date (before invoice date)
const clientAddedDate = new Date(invoiceDate.getTime() - Math.random() * 365 * 24 * 60 * 60 * 1000);

// Determine if repeat client
const clientVisits = clientHistory.get(clientName).filter(date => date <= invoiceDate).length;
const isRepeatClient = clientVisits > 1;

data.push({
  InvoiceDate: invoiceDate.toISOString().split('T')[0],
  Branch: branches[Math.floor(Math.random() * branches.length)],
  Salesperson: salespersons[Math.floor(Math.random() * salespersons.length)],
  ClientName: clientName,
  TestName: testName,
  Specialty: specialty,
  TestMRP: testMRP,
  DiscountAmount: Math.round(discountAmount),
  DiscountPercent: Math.round(discountPercent * 100),
  BilledAmount: billedAmount,
  NetRevenue: netRevenue,
  CollectedAmount: collectedAmount,
  OutstandingAmount: outstandingAmount,
  PaymentMethod: paymentMethods[Math.floor(Math.random() * paymentMethods.length)],
  ReferralSource: referralSources[Math.floor(Math.random() * referralSources.length)],
  ReferralType: referralTypes[Math.floor(Math.random() * referralTypes.length)],
  IsRepeatClient: isRepeatClient,
  ClientVisitNumber: clientVisits,
  InvoiceID: `LAB${invoiceDate.getFullYear()}${String(invoiceDate.getMonth() + 1).padStart(2, '0')}${Math.floor(Math.random() * 9000) + 1000}`,
  ClientAddedDate: clientAddedDate.toISOString().split('T')[0]
});
```

}

return data.sort((a, b) => new Date(a.InvoiceDate) - new Date(b.InvoiceDate));
};

const EnhancedLabDashboard = () => {
const [data, setData] = useState([]);
const [activeTab, setActiveTab] = useState(‘overview’);
const [filters, setFilters] = useState({
dateRange: { start: ‘’, end: ‘’ },
branch: ‘’,
salesperson: ‘’,
specialty: ‘’,
highValueOnly: false,
testSearch: ‘’
});
const [selectedMonth, setSelectedMonth] = useState(‘2024-12’);
const [targets, setTargets] = useState({
1: 1000000, 2: 950000, 3: 1100000, 4: 1050000, 5: 1150000, 6: 1200000,
7: 1300000, 8: 1250000, 9: 1100000, 10: 1400000, 11: 1350000, 12: 1500000
});

// Configuration states
const [config, setConfig] = useState({
projectionDays: 7,
alertThresholds: {
revenueDropPercent: -20,
highDiscountPercent: 25,
lowCollectionRate: 75
}
});

useEffect(() => {
const sampleData = generateEnhancedSampleData();
setData(sampleData);

```
const dates = sampleData.map(d => d.InvoiceDate).sort();
setFilters(prev => ({
  ...prev,
  dateRange: { start: dates[0], end: dates[dates.length - 1] }
}));
```

}, []);

// Apply filters to data
const filteredData = useMemo(() => {
return data.filter(row => {
if (filters.dateRange.start && row.InvoiceDate < filters.dateRange.start) return false;
if (filters.dateRange.end && row.InvoiceDate > filters.dateRange.end) return false;
if (filters.branch && row.Branch !== filters.branch) return false;
if (filters.salesperson && row.Salesperson !== filters.salesperson) return false;
if (filters.specialty && row.Specialty !== filters.specialty) return false;
if (filters.highValueOnly && row.NetRevenue < 999) return false;
if (filters.testSearch && !row.TestName.toLowerCase().includes(filters.testSearch.toLowerCase())) return false;
return true;
});
}, [data, filters]);

// Enhanced KPIs with new metrics
const enhancedKpis = useMemo(() => {
const currentMonthData = filteredData.filter(row =>
row.InvoiceDate.startsWith(selectedMonth)
);

```
const mtdRevenue = currentMonthData.reduce((sum, row) => sum + row.NetRevenue, 0);
const totalBilled = currentMonthData.reduce((sum, row) => sum + row.BilledAmount, 0);
const totalCollected = currentMonthData.reduce((sum, row) => sum + row.CollectedAmount, 0);
const totalOutstanding = currentMonthData.reduce((sum, row) => sum + row.OutstandingAmount, 0);
const totalDiscount = currentMonthData.reduce((sum, row) => sum + row.DiscountAmount, 0);

const highValueTests = filteredData.filter(row => row.NetRevenue >= 999).length;
const uniqueClients = new Set(currentMonthData.map(row => row.ClientName)).size;
const repeatClients = currentMonthData.filter(row => row.IsRepeatClient).length;

// Monthly projection based on recent days
const today = new Date();
const recentDaysStart = new Date(today.getTime() - (config.projectionDays * 24 * 60 * 60 * 1000));
const recentData = filteredData.filter(row => 
  new Date(row.InvoiceDate) >= recentDaysStart && new Date(row.InvoiceDate) <= today
);
const recentRevenue = recentData.reduce((sum, row) => sum + row.NetRevenue, 0);
const dailyAverage = recentRevenue / config.projectionDays;
const monthlyProjection = dailyAverage * 30;

// Collection rate
const collectionRate = totalBilled > 0 ? (totalCollected / totalBilled * 100) : 0;

// Discount rate
const grossRevenue = totalBilled + totalDiscount;
const discountRate = grossRevenue > 0 ? (totalDiscount / grossRevenue * 100) : 0;

const monthNum = parseInt(selectedMonth.split('-')[1]);
const target = targets[monthNum] || 0;
const variance = mtdRevenue - target;
const variancePct = target > 0 ? (variance / target * 100) : 0;

return {
  mtdRevenue,
  target,
  variance,
  variancePct,
  highValueTests,
  clientsAddedMtd: uniqueClients,
  repeatClients,
  totalDiscount,
  discountRate,
  collectionRate,
  totalOutstanding,
  monthlyProjection,
  dailyAverage
};
```

}, [filteredData, selectedMonth, targets, config.projectionDays]);

// Monthly projection data
const projectionData = useMemo(() => {
const monthlyRevenue = {};
filteredData.forEach(row => {
const month = row.InvoiceDate.substring(0, 7);
monthlyRevenue[month] = (monthlyRevenue[month] || 0) + row.NetRevenue;
});

```
return Object.entries(monthlyRevenue)
  .sort(([a], [b]) => a.localeCompare(b))
  .slice(-12)
  .map(([month, revenue]) => {
    const monthNum = parseInt(month.split('-')[1]);
    const target = targets[monthNum] || 0;
    const isCurrentMonth = month === selectedMonth;
    const projected = isCurrentMonth ? enhancedKpis.monthlyProjection : revenue;
    
    return {
      month,
      revenue,
      target,
      projected,
      variance: revenue - target,
      isProjection: isCurrentMonth
    };
  });
```

}, [filteredData, targets, selectedMonth, enhancedKpis.monthlyProjection]);

// Referral analysis data
const referralData = useMemo(() => {
const referralRevenue = {};
filteredData.forEach(row => {
const key = `${row.ReferralType} - ${row.ReferralSource}`;
referralRevenue[key] = (referralRevenue[key] || 0) + row.NetRevenue;
});

```
return Object.entries(referralRevenue)
  .map(([source, revenue]) => ({ source, revenue }))
  .sort((a, b) => b.revenue - a.revenue);
```

}, [filteredData]);

// Payment method analysis
const paymentData = useMemo(() => {
const paymentRevenue = {};
filteredData.forEach(row => {
const method = row.PaymentMethod || ‘Unknown’;
paymentRevenue[method] = (paymentRevenue[method] || 0) + row.NetRevenue;
});

```
return Object.entries(paymentRevenue)
  .map(([method, revenue]) => ({ method, revenue }))
  .sort((a, b) => b.revenue - a.revenue);
```

}, [filteredData]);

// Client retention analysis
const clientRetentionData = useMemo(() => {
const clientStats = {};
filteredData.forEach(row => {
if (!clientStats[row.ClientName]) {
clientStats[row.ClientName] = {
visits: 0,
totalRevenue: 0,
firstVisit: row.InvoiceDate,
lastVisit: row.InvoiceDate
};
}
clientStats[row.ClientName].visits += 1;
clientStats[row.ClientName].totalRevenue += row.NetRevenue;
if (row.InvoiceDate > clientStats[row.ClientName].lastVisit) {
clientStats[row.ClientName].lastVisit = row.InvoiceDate;
}
});

```
const retentionStats = {
  totalClients: Object.keys(clientStats).length,
  repeatClients: Object.values(clientStats).filter(client => client.visits > 1).length,
  averageVisits: Object.values(clientStats).reduce((sum, client) => sum + client.visits, 0) / Object.keys(clientStats).length,
  averageRevenuePerClient: Object.values(clientStats).reduce((sum, client) => sum + client.totalRevenue, 0) / Object.keys(clientStats).length
};

const topClients = Object.entries(clientStats)
  .map(([name, stats]) => ({ name, ...stats, avgPerVisit: stats.totalRevenue / stats.visits }))
  .sort((a, b) => b.totalRevenue - a.totalRevenue)
  .slice(0, 10);

return { retentionStats, topClients };
```

}, [filteredData]);

// Test mix analysis
const testMixData = useMemo(() => {
const testRevenue = {};
const testCounts = {};

```
filteredData.forEach(row => {
  testRevenue[row.TestName] = (testRevenue[row.TestName] || 0) + row.NetRevenue;
  testCounts[row.TestName] = (testCounts[row.TestName] || 0) + 1;
});

return Object.entries(testRevenue)
  .map(([test, revenue]) => ({
    test,
    revenue,
    count: testCounts[test],
    avgRevenue: revenue / testCounts[test]
  }))
  .sort((a, b) => b.revenue - a.revenue)
  .slice(0, 15);
```

}, [filteredData]);

// Trend analysis data
const trendData = useMemo(() => {
const monthlyData = {};

```
filteredData.forEach(row => {
  const month = row.InvoiceDate.substring(0, 7);
  
  if (!monthlyData[month]) {
    monthlyData[month] = { revenue: 0, discount: 0, transactions: 0 };
  }
  
  monthlyData[month].revenue += row.NetRevenue;
  monthlyData[month].discount += row.DiscountAmount;
  monthlyData[month].transactions += 1;
});

const monthlyTrends = Object.entries(monthlyData)
  .sort(([a], [b]) => a.localeCompare(b))
  .slice(-12)
  .map(([month, data]) => ({ period: month, ...data }));

return monthlyTrends;
```

}, [filteredData]);

// Salesperson efficiency data
const salespersonData = useMemo(() => {
const salesStats = {};

```
filteredData.forEach(row => {
  if (!salesStats[row.Salesperson]) {
    salesStats[row.Salesperson] = {
      revenue: 0,
      transactions: 0,
      clients: new Set(),
      discount: 0
    };
  }
  salesStats[row.Salesperson].revenue += row.NetRevenue;
  salesStats[row.Salesperson].transactions += 1;
  salesStats[row.Salesperson].clients.add(row.ClientName);
  salesStats[row.Salesperson].discount += row.DiscountAmount;
});

return Object.entries(salesStats)
  .map(([name, stats]) => ({
    name,
    revenue: stats.revenue,
    transactions: stats.transactions,
    uniqueClients: stats.clients.size,
    avgTicket: stats.revenue / stats.transactions,
    discountGiven: stats.discount,
    revenuePerClient: stats.revenue / stats.clients.size
  }))
  .sort((a, b) => b.revenue - a.revenue);
```

}, [filteredData]);

// Alerts calculation
const alerts = useMemo(() => {
const alertsList = [];

```
// Revenue drop alert
if (enhancedKpis.variancePct < config.alertThresholds.revenueDropPercent) {
  alertsList.push({
    type: 'warning',
    message: `Revenue is ${Math.abs(enhancedKpis.variancePct).toFixed(1)}% below target`,
    metric: 'Revenue Variance'
  });
}

// High discount alert
if (enhancedKpis.discountRate > config.alertThresholds.highDiscountPercent) {
  alertsList.push({
    type: 'error',
    message: `Discount rate is ${enhancedKpis.discountRate.toFixed(1)}% (above ${config.alertThresholds.highDiscountPercent}% threshold)`,
    metric: 'Discount Rate'
  });
}

// Low collection rate alert
if (enhancedKpis.collectionRate < config.alertThresholds.lowCollectionRate) {
  alertsList.push({
    type: 'error',
    message: `Collection rate is ${enhancedKpis.collectionRate.toFixed(1)}% (below ${config.alertThresholds.lowCollectionRate}% threshold)`,
    metric: 'Collection Rate'
  });
}

return alertsList;
```

}, [enhancedKpis, config.alertThresholds]);

const formatCurrency = (value) => `₹${(value || 0).toLocaleString()}`;
const formatNumber = (value) => (value || 0).toLocaleString();
const formatPercent = (value) => `${(value || 0).toFixed(1)}%`;

const colors = [’#8884d8’, ‘#82ca9d’, ‘#ffc658’, ‘#ff7c7c’, ‘#8dd1e1’, ‘#d084d0’, ‘#87d068’, ‘#ffc0cb’];

const renderOverviewTab = () => (
<div className="space-y-8">
{/* Enhanced KPI Cards */}
<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
<div className="bg-white rounded-lg shadow-sm border p-6">
<div className="flex items-center gap-3">
<div className="p-2 bg-green-100 rounded-lg">
<DollarSign className="h-6 w-6 text-green-600" />
</div>
<div>
<p className="text-sm font-medium text-gray-600">MTD Revenue</p>
<p className="text-2xl font-bold text-gray-900">{formatCurrency(enhancedKpis.mtdRevenue)}</p>
<p className="text-sm text-gray-500">Projected: {formatCurrency(enhancedKpis.monthlyProjection)}</p>
</div>
</div>
</div>

```
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex items-center gap-3">
        <div className={`p-2 rounded-lg ${enhancedKpis.variance >= 0 ? 'bg-green-100' : 'bg-red-100'}`}>
          <TrendingUp className={`h-6 w-6 ${enhancedKpis.variance >= 0 ? 'text-green-600' : 'text-red-600'}`} />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-600">Revenue Variance</p>
          <p className={`text-2xl font-bold ${enhancedKpis.variance >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {formatCurrency(enhancedKpis.variance)}
          </p>
          <p className={`text-sm ${enhancedKpis.variance >= 0 ? 'text-green-600' : 'text-red-600'}`}>
            {enhancedKpis.variancePct > 0 ? '+' : ''}{formatPercent(enhancedKpis.variancePct)}
          </p>
        </div>
      </div>
    </div>

    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-orange-100 rounded-lg">
          <Percent className="h-6 w-6 text-orange-600" />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-600">Collection Rate</p>
          <p className="text-2xl font-bold text-gray-900">{formatPercent(enhancedKpis.collectionRate)}</p>
          <p className="text-sm text-gray-500">Outstanding: {formatCurrency(enhancedKpis.totalOutstanding)}</p>
        </div>
      </div>
    </div>

    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-red-100 rounded-lg">
          <Percent className="h-6 w-6 text-red-600" />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-600">Discount Rate</p>
          <p className="text-2xl font-bold text-gray-900">{formatPercent(enhancedKpis.discountRate)}</p>
          <p className="text-sm text-gray-500">Total: {formatCurrency(enhancedKpis.totalDiscount)}</p>
        </div>
      </div>
    </div>
  </div>

  {/* Alerts Section */}
  {alerts.length > 0 && (
    <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-3">
        <AlertTriangle className="h-5 w-5 text-yellow-600" />
        <h3 className="font-semibold text-yellow-800">Alerts & Notifications</h3>
      </div>
      <div className="space-y-2">
        {alerts.map((alert, index) => (
          <div key={index} className={`p-2 rounded ${alert.type === 'error' ? 'bg-red-100 text-red-800' : 'bg-yellow-100 text-yellow-800'}`}>
            <span className="font-medium">{alert.metric}:</span> {alert.message}
          </div>
        ))}
      </div>
    </div>
  )}

  {/* Monthly Projection Chart */}
  <div className="bg-white rounded-lg shadow-sm border p-6">
    <h3 className="text-lg font-semibold text-gray-900 mb-4">Monthly Performance & Projection</h3>
    <ResponsiveContainer width="100%" height={400}>
      <ComposedChart data={projectionData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="month" />
        <YAxis tickFormatter={(value) => `₹${(value/1000).toFixed(0)}K`} />
        <Tooltip formatter={(value) => formatCurrency(value)} />
        <Legend />
        <Bar dataKey="revenue" fill="#8884d8" name="Actual Revenue" />
        <Bar dataKey="projected" fill="#82ca9d" name="Projected Revenue" fillOpacity={0.7} />
        <Line dataKey="target" stroke="#ff7300" strokeWidth={3} name="Target" />
      </ComposedChart>
    </ResponsiveContainer>
  </div>
</div>
```

);

const renderFinancialTab = () => (
<div className="space-y-8">
{/* Financial KPIs */}
<div className="grid grid-cols-1 md:grid-cols-4 gap-6">
<div className="bg-white rounded-lg shadow-sm border p-6">
<div className="flex items-center gap-3">
<div className="p-2 bg-blue-100 rounded-lg">
<DollarSign className="h-6 w-6 text-blue-600" />
</div>
<div>
<p className="text-sm font-medium text-gray-600">Total Billed</p>
<p className="text-2xl font-bold text-gray-900">
{formatCurrency(filteredData.reduce((sum, row) => sum + row.BilledAmount, 0))}
</p>
</div>
</div>
</div>

```
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-green-100 rounded-lg">
          <DollarSign className="h-6 w-6 text-green-600" />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-600">Total Collected</p>
          <p className="text-2xl font-bold text-gray-900">
            {formatCurrency(filteredData.reduce((sum, row) => sum + row.CollectedAmount, 0))}
          </p>
        </div>
      </div>
    </div>

    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-red-100 rounded-lg">
          <DollarSign className="h-6 w-6 text-red-600" />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-600">Outstanding</p>
          <p className="text-2xl font-bold text-gray-900">
            {formatCurrency(filteredData.reduce((sum, row) => sum + row.OutstandingAmount, 0))}
          </p>
        </div>
      </div>
    </div>

    <div className="bg-white rounded-lg shadow-sm border p-6">
      <div className="flex items-center gap-3">
        <div className="p-2 bg-purple-100 rounded-lg">
          <Percent className="h-6 w-6 text-purple-600" />
        </div>
        <div>
          <p className="text-sm font-medium text-gray-600">Avg Discount</p>
          <p className="text-2xl font-bold text-gray-900">
            {formatPercent(filteredData.reduce((sum, row) => sum + row.DiscountPercent, 0) / filteredData.length)}
          </p>
        </div>
      </div>
    </div>
  </div>

  {/* Charts */}
  <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
    <div className="bg-white rounded-lg shadow-sm border p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Payment Method Breakdown</h3>
      <ResponsiveContainer width="100%" height={300}>
        <PieChart>
          <Pie
            data={paymentData}
            cx="50%"
            cy="50%"
            labelLine={false}
            label={({ method, percent }) => `${method} ${(percent * 100).toFixed(0)}%`}
            outerRadius={80}
            fill="#8884d8"
            dataKey="revenue"
          >
            {paymentData.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
            ))}
          </Pie>
          <Tooltip formatter={(value) => formatCurrency(value)} />
        </PieChart>
      </ResponsiveContainer>
    </div>
```
