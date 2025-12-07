# -*- coding: utf-8 -*-
"""
G3 Security Engagement CMGR (Compound Monthly Growth Rate) Analysis
Evaluates revenue growth acceleration before vs after G3 engagements
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import sqlalchemy
import boto3
import json
import warnings
warnings.filterwarnings('ignore')

def get_secret(secret_name, region_name, key):
    """Function to retrieve secret credentials from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(service_name='secretsmanager', region_name=region_name)
    secret_response = client.get_secret_value(SecretId=secret_name)
    secret_string = secret_response['SecretString']
    secret_json = json.loads(secret_string)
    return secret_json[key]

def sql_pull(dw, sql):
    """Function to pull data from PostgreSQL database."""
    url = get_secret(dw, 'us-east-1', 'url')
    port = get_secret(dw, 'us-east-1', 'port')
    db = get_secret(dw, 'us-east-1', 'db')
    un = get_secret(dw, 'us-east-1', 'username')
    pw = get_secret(dw, 'us-east-1', 'password')

    engine = sqlalchemy.create_engine(f'postgresql://{un}:{pw}@{url}:{port}/{db}')
    conn = engine.connect()
    dat = conn.execute(sqlalchemy.text(sql)).fetchall()
    df = pd.DataFrame(dat)
    if len(dat) > 0:
        df.columns = dat[0].keys()
    conn.close()
    return df

def calculate_cmgr(revenue_series):
    """Calculate Compound Monthly Growth Rate for a revenue time series."""
    if len(revenue_series) < 2:
        return None
    
    # Remove zero and negative values
    revenue_series = revenue_series[revenue_series > 0]
    if len(revenue_series) < 2:
        return None
    
    start_rev = revenue_series.iloc[0]
    end_rev = revenue_series.iloc[-1]
    months = len(revenue_series) - 1
    
    if start_rev <= 0 or months <= 0:
        return None
    
    cmgr = (end_rev / start_rev) ** (1/months) - 1
    return cmgr

def calculate_customer_cmgr_impact(df, customer_id, engagement_date, revenue_col, 
                                 pre_months=6, post_months=6):
    """Calculate CMGR before and after G3 engagement for a specific customer."""
    
    customer_data = df[df['sfdc_customer_id'] == customer_id].sort_values('ar_date')
    
    if len(customer_data) < 12:  # Need at least 12 months of data
        return None
    
    # Define pre/post periods
    pre_start = engagement_date - pd.DateOffset(months=pre_months)
    pre_end = engagement_date
    post_start = engagement_date
    post_end = engagement_date + pd.DateOffset(months=post_months)
    
    # Get pre/post revenue data
    pre_data = customer_data[
        (customer_data['ar_date'] >= pre_start) & 
        (customer_data['ar_date'] < pre_end)
    ][revenue_col]
    
    post_data = customer_data[
        (customer_data['ar_date'] >= post_start) & 
        (customer_data['ar_date'] < post_end)
    ][revenue_col]
    
    # Calculate CMGR for each period
    pre_cmgr = calculate_cmgr(pre_data)
    post_cmgr = calculate_cmgr(post_data)
    
    if pre_cmgr is None or post_cmgr is None:
        return None
    
    return {
        'customer_id': customer_id,
        'engagement_date': engagement_date,
        'pre_cmgr': pre_cmgr,
        'post_cmgr': post_cmgr,
        'cmgr_acceleration': post_cmgr - pre_cmgr,
        'pre_months': len(pre_data),
        'post_months': len(post_data)
    }

def analyze_treatment_group_cmgr(df, revenue_cols):
    """Analyze CMGR for treatment group customers."""
    treatment_results = []
    
    # Get customers with engagements
    engaged_customers = df[df['firstactivitydate'].notna()]['sfdc_customer_id'].unique()
    
    print(f"Analyzing CMGR for {len(engaged_customers)} engaged customers...")
    
    for customer_id in engaged_customers:
        customer_data = df[df['sfdc_customer_id'] == customer_id]
        engagement_date = customer_data['firstactivitydate'].iloc[0]
        
        if pd.notna(engagement_date):
            for revenue_col in revenue_cols:
                result = calculate_customer_cmgr_impact(
                    df, customer_id, engagement_date, revenue_col
                )
                if result:
                    result['revenue_type'] = revenue_col
                    treatment_results.append(result)
    
    return pd.DataFrame(treatment_results)

def analyze_control_group_cmgr(df, treatment_customers, revenue_cols):
    """Analyze CMGR for control group using pseudo-engagement dates."""
    control_results = []
    
    # Get control customers (no engagement) with sufficient data
    control_customers = df[df['firstactivitydate'].isna()]['sfdc_customer_id'].unique()
    
    # Filter control customers to those with sufficient data
    suitable_control = []
    median_engagement_date = df[df['firstactivitydate'].notna()]['firstactivitydate'].median()
    
    for customer_id in control_customers:
        customer_data = df[df['sfdc_customer_id'] == customer_id]
        if len(customer_data) >= 12:  # At least 12 months of data
            pre_months = len(customer_data[customer_data['ar_date'] < median_engagement_date])
            post_months = len(customer_data[customer_data['ar_date'] >= median_engagement_date])
            if pre_months >= 6 and post_months >= 6:
                suitable_control.append(customer_id)
    
    print(f"Found {len(suitable_control)} suitable control customers out of {len(control_customers)} total")
    print(f"Using pseudo-engagement date: {median_engagement_date}")
    
    # Sample control customers to match treatment group size
    np.random.seed(42)
    if len(suitable_control) > 0:
        sampled_control = np.random.choice(
            suitable_control, 
            size=min(len(treatment_customers), len(suitable_control)), 
            replace=False
        )
        
        for customer_id in sampled_control:
            for revenue_col in revenue_cols:
                result = calculate_customer_cmgr_impact(
                    df, customer_id, median_engagement_date, revenue_col
                )
                if result:
                    result['revenue_type'] = revenue_col
                    control_results.append(result)
    
    return pd.DataFrame(control_results)

def validate_data_for_cmgr(df):
    """Validate data sufficiency for CMGR analysis."""
    print("Data Validation for CMGR Analysis")
    print("=" * 50)
    
    # Check engagement customers data quality
    engaged_customers = df[df['firstactivitydate'].notna()]
    
    data_quality = []
    for customer_id in engaged_customers['sfdc_customer_id'].unique():
        customer_data = df[df['sfdc_customer_id'] == customer_id]
        engagement_date = customer_data['firstactivitydate'].iloc[0]
        
        pre_months = len(customer_data[customer_data['ar_date'] < engagement_date])
        post_months = len(customer_data[customer_data['ar_date'] >= engagement_date])
        total_months = len(customer_data)
        
        data_quality.append({
            'customer_id': customer_id,
            'engagement_date': engagement_date,
            'total_months': total_months,
            'pre_months': pre_months,
            'post_months': post_months,
            'suitable_for_cmgr': (pre_months >= 6 and post_months >= 6)
        })
    
    quality_df = pd.DataFrame(data_quality)
    
    print(f"Total engaged customers: {len(quality_df)}")
    print(f"Customers suitable for CMGR (6+ months pre/post): {quality_df['suitable_for_cmgr'].sum()}")
    print(f"Average months of data: {quality_df['total_months'].mean():.1f}")
    print(f"Average pre-engagement months: {quality_df['pre_months'].mean():.1f}")
    print(f"Average post-engagement months: {quality_df['post_months'].mean():.1f}")
    
    return quality_df

print("G3 Security Engagement CMGR Analysis")
print("=" * 50)

# Load data
print("Loading data from database...")
df = sql_pull('RACEleducse', """
SELECT
    a.sfdc_customer_id,
    a.ar_date,
    a.firstactivitydate,
    a.ttl_sls_rev,
    a.ttl_security_rev,
    a.ttl_resiliency_rev,
    b.sub_segment
FROM sa_ops_sb.g3_2025_kpi_3_increase_sec_adoption a
JOIN sa_ops_sb.tc_account b ON a.sfdc_customer_id = b.level_14_id
WHERE a.ar_date IS NOT NULL AND b.level_1 = 'WWPS'
ORDER BY a.sfdc_customer_id, a.ar_date;
""")

print(f"Loaded {len(df)} records from {df['sfdc_customer_id'].nunique()} customers")

# Data preparation
df['ar_date'] = pd.to_datetime(df['ar_date'])
df['firstactivitydate'] = pd.to_datetime(df['firstactivitydate'])

# Convert revenue columns to numeric
revenue_cols = ['ttl_sls_rev', 'ttl_security_rev', 'ttl_resiliency_rev']
for col in revenue_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# Validate data for CMGR analysis
quality_df = validate_data_for_cmgr(df)

# Filter to customers suitable for CMGR analysis
suitable_customers = quality_df[quality_df['suitable_for_cmgr']]['customer_id'].tolist()
df_suitable = df[df['sfdc_customer_id'].isin(suitable_customers)]

print(f"\nFiltered to {len(df_suitable)} records from {df_suitable['sfdc_customer_id'].nunique()} suitable customers")

# Analyze treatment group CMGR
print("\nAnalyzing Treatment Group CMGR...")
treatment_cmgr = analyze_treatment_group_cmgr(df_suitable, revenue_cols)

# Analyze control group CMGR
print("\nAnalyzing Control Group CMGR...")
control_cmgr = analyze_control_group_cmgr(df_suitable, treatment_cmgr['customer_id'].unique(), revenue_cols)

# Results Analysis
print("\n" + "=" * 50)
print("CMGR ANALYSIS RESULTS")
print("=" * 50)

if len(control_cmgr) == 0:
    print("WARNING: No suitable control customers found. Showing treatment group analysis only.")
    print("This limits our ability to establish causal inference.")
    
    # Treatment group only analysis
    for revenue_type in revenue_cols:
        treatment_data = treatment_cmgr[treatment_cmgr['revenue_type'] == revenue_type]
        
        if len(treatment_data) > 0:
            treat_pre_cmgr = treatment_data['pre_cmgr'].mean()
            treat_post_cmgr = treatment_data['post_cmgr'].mean()
            treat_acceleration = treatment_data['cmgr_acceleration'].mean()
            
            print(f"\n{revenue_type.replace('_', ' ').title()}:")
            print(f"  Treatment Group (G3 Engaged Customers):")
            print(f"    Pre-Engagement CMGR: {treat_pre_cmgr:.3f} ({treat_pre_cmgr*100:.1f}% monthly)")
            print(f"    Post-Engagement CMGR: {treat_post_cmgr:.3f} ({treat_post_cmgr*100:.1f}% monthly)")
            print(f"    CMGR Acceleration: {treat_acceleration:.3f} ({treat_acceleration*100:.1f}% monthly)")
            print(f"    Sample Size: {len(treatment_data)} customers")
            
            # Test if acceleration is significantly different from zero
            from scipy.stats import ttest_1samp
            t_stat, p_val = ttest_1samp(treatment_data['cmgr_acceleration'], 0)
            print(f"    P-value (vs zero acceleration): {p_val:.4f}")
            significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
            print(f"    Significance: {significance}")
else:
    # Full treatment vs control analysis
    for revenue_type in revenue_cols:
        treatment_data = treatment_cmgr[treatment_cmgr['revenue_type'] == revenue_type]
        control_data = control_cmgr[control_cmgr['revenue_type'] == revenue_type]
        
        if len(treatment_data) > 0 and len(control_data) > 0:
            # Calculate statistics
            treat_pre_cmgr = treatment_data['pre_cmgr'].mean()
            treat_post_cmgr = treatment_data['post_cmgr'].mean()
            treat_acceleration = treatment_data['cmgr_acceleration'].mean()
            
            control_pre_cmgr = control_data['pre_cmgr'].mean()
            control_post_cmgr = control_data['post_cmgr'].mean()
            control_acceleration = control_data['cmgr_acceleration'].mean()
            
            # Statistical test
            t_stat, p_val = ttest_ind(treatment_data['cmgr_acceleration'], control_data['cmgr_acceleration'])
            
            print(f"\n{revenue_type.replace('_', ' ').title()}:")
            print(f"  Treatment Group:")
            print(f"    Pre-Engagement CMGR: {treat_pre_cmgr:.3f} ({treat_pre_cmgr*100:.1f}% monthly)")
            print(f"    Post-Engagement CMGR: {treat_post_cmgr:.3f} ({treat_post_cmgr*100:.1f}% monthly)")
            print(f"    CMGR Acceleration: {treat_acceleration:.3f} ({treat_acceleration*100:.1f}% monthly)")
            print(f"  Control Group:")
            print(f"    Pre-Period CMGR: {control_pre_cmgr:.3f} ({control_pre_cmgr*100:.1f}% monthly)")
            print(f"    Post-Period CMGR: {control_post_cmgr:.3f} ({control_post_cmgr*100:.1f}% monthly)")
            print(f"    CMGR Acceleration: {control_acceleration:.3f} ({control_acceleration*100:.1f}% monthly)")
            print(f"  Net Impact: {treat_acceleration - control_acceleration:.3f} ({(treat_acceleration - control_acceleration)*100:.1f}% monthly)")
            print(f"  P-value: {p_val:.4f}")
            print(f"  Sample Size: {len(treatment_data)} treatment, {len(control_data)} control")

# Segment Analysis
print("\n" + "=" * 50)
print("SEGMENT ANALYSIS")
print("=" * 50)

# Add segment information to CMGR results
treatment_cmgr_with_segment = treatment_cmgr.merge(
    df_suitable[['sfdc_customer_id', 'sub_segment']].drop_duplicates(),
    left_on='customer_id', right_on='sfdc_customer_id', how='left'
)

for segment in treatment_cmgr_with_segment['sub_segment'].unique():
    if pd.notna(segment):
        segment_data = treatment_cmgr_with_segment[
            (treatment_cmgr_with_segment['sub_segment'] == segment) &
            (treatment_cmgr_with_segment['revenue_type'] == 'ttl_sls_rev')
        ]
        
        if len(segment_data) > 5:  # Minimum sample size
            avg_acceleration = segment_data['cmgr_acceleration'].mean()
            print(f"{segment}: {avg_acceleration:.3f} ({avg_acceleration*100:.1f}% monthly) - {len(segment_data)} customers")

# Time-to-Impact Analysis
print("\n" + "=" * 50)
print("TIME-TO-IMPACT ANALYSIS")
print("=" * 50)

# Analyze CMGR acceleration over different post-engagement periods
time_periods = [3, 6, 9, 12]
time_impact_results = {}

for months_post in time_periods:
    monthly_impacts = []
    
    suitable_customers_list = df_suitable[df_suitable['firstactivitydate'].notna()]['sfdc_customer_id'].unique()
    
    for customer_id in suitable_customers_list[:50]:  # Sample for performance
        customer_data = df_suitable[df_suitable['sfdc_customer_id'] == customer_id]
        engagement_date = customer_data['firstactivitydate'].iloc[0]
        
        result = calculate_customer_cmgr_impact(
            df_suitable, customer_id, engagement_date, 'ttl_sls_rev',
            pre_months=6, post_months=months_post
        )
        
        if result and result['cmgr_acceleration'] is not None:
            monthly_impacts.append(result['cmgr_acceleration'])
    
    if len(monthly_impacts) > 0:
        time_impact_results[months_post] = {
            'mean_acceleration': np.mean(monthly_impacts),
            'sample_size': len(monthly_impacts)
        }

print("CMGR Acceleration by Post-Engagement Period:")
for months, result in time_impact_results.items():
    print(f"  {months} months post: {result['mean_acceleration']:.3f} ({result['mean_acceleration']*100:.1f}% monthly) - {result['sample_size']} customers")

# Summary
print("\n" + "=" * 70)
print("CMGR ANALYSIS SUMMARY")
print("=" * 70)

print(f"Customers analyzed: {len(treatment_cmgr['customer_id'].unique())} treatment")
print(f"Data quality: {quality_df['suitable_for_cmgr'].sum()} out of {len(quality_df)} customers suitable for CMGR")

print("\nKey Findings:")
for revenue_type in revenue_cols:
    treatment_data = treatment_cmgr[treatment_cmgr['revenue_type'] == revenue_type]
    
    if len(treatment_data) > 0:
        if len(control_cmgr) > 0:
            control_data = control_cmgr[control_cmgr['revenue_type'] == revenue_type]
            if len(control_data) > 0:
                net_impact = treatment_data['cmgr_acceleration'].mean() - control_data['cmgr_acceleration'].mean()
                t_stat, p_val = ttest_ind(treatment_data['cmgr_acceleration'], control_data['cmgr_acceleration'])
            else:
                net_impact = treatment_data['cmgr_acceleration'].mean()
                from scipy.stats import ttest_1samp
                t_stat, p_val = ttest_1samp(treatment_data['cmgr_acceleration'], 0)
        else:
            net_impact = treatment_data['cmgr_acceleration'].mean()
            from scipy.stats import ttest_1samp
            t_stat, p_val = ttest_1samp(treatment_data['cmgr_acceleration'], 0)
        
        significance = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"  {revenue_type}: {net_impact*100:+.1f}% monthly acceleration {significance}")

print("\nMethodological Notes:")
print("  ✓ CMGR measures growth rate changes before vs after engagement")
print("  ✓ Control group uses pseudo-engagement dates for comparison")
print("  ✓ Minimum 6 months pre/post engagement data required")
print("  ✓ Analysis captures momentum effects beyond static comparisons")

# Save results
results_summary = []
for revenue_type in revenue_cols:
    treatment_data = treatment_cmgr[treatment_cmgr['revenue_type'] == revenue_type]
    
    if len(treatment_data) > 0:
        if len(control_cmgr) > 0:
            control_data = control_cmgr[control_cmgr['revenue_type'] == revenue_type]
            if len(control_data) > 0:
                net_impact = treatment_data['cmgr_acceleration'].mean() - control_data['cmgr_acceleration'].mean()
                t_stat, p_val = ttest_ind(treatment_data['cmgr_acceleration'], control_data['cmgr_acceleration'])
                control_acceleration = control_data['cmgr_acceleration'].mean()
            else:
                net_impact = treatment_data['cmgr_acceleration'].mean()
                from scipy.stats import ttest_1samp
                t_stat, p_val = ttest_1samp(treatment_data['cmgr_acceleration'], 0)
                control_acceleration = 0
        else:
            net_impact = treatment_data['cmgr_acceleration'].mean()
            from scipy.stats import ttest_1samp
            t_stat, p_val = ttest_1samp(treatment_data['cmgr_acceleration'], 0)
            control_acceleration = 0
        
        results_summary.append({
            'revenue_type': revenue_type,
            'treatment_acceleration': treatment_data['cmgr_acceleration'].mean(),
            'control_acceleration': control_acceleration,
            'net_impact_monthly': net_impact,
            'net_impact_annual': (1 + net_impact)**12 - 1,
            'p_value': p_val,
            'sample_size': len(treatment_data)
        })

results_df = pd.DataFrame(results_summary)
results_df.to_csv('/Users/leducse/Documents/Projects/G3 Lift Analysis/G3_CMGR_analysis_results.csv', index=False)
print(f"\nResults saved to: G3_CMGR_analysis_results.csv")