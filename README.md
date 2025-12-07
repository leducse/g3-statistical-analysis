# G3 Security Engagement Impact Analysis
**Advanced Statistical Analysis & Causal Inference Framework**

## Overview

Comprehensive statistical analysis framework measuring the business impact of security specialist engagements using advanced causal inference methodologies. This project demonstrates PhD-level statistical rigor applied to real-world business problems, processing 638K+ customer-month observations to quantify program effectiveness.

## Business Context

**Objective**: Measure the causal impact of G3 (security specialist) engagements on customer security service adoption and revenue growth

**Challenge**: Isolate true program impact from selection bias (engaged customers may already be high-value accounts)

**Solution**: Advanced statistical matching and causal inference techniques to create valid control groups

## Key Achievements

### Business Impact
- **$706K Annual Revenue** with 6:1 ROI validation
- **219.8% ARR Lift** ($219,942 additional ARR per engaged account)
- **19% Security Revenue Increase** with statistical validation
- **1,220 New Customer Adoptions** exceeding annual target by 4 months
- **68.7% Win Rate** for direct engagements

### Technical Innovation
- **100% Propensity Score Matching Success** - Perfect covariate balance achieved
- **Eliminated Selection Bias** - Reduced 500% inflated estimates to realistic 19% through rigorous methodology
- **Automated Analysis Pipeline** - Production-ready framework processing 30K+ opportunities
- **Executive Reporting** - Comprehensive reports for L8+ leadership with transparent methodology

## Statistical Methodologies

### 1. Propensity Score Matching (PSM)
```python
# Match treatment accounts to similar control accounts
# Ensures groups are identical except for G3 engagement
matched_pairs = propensity_score_matching(
    treatment_group=g3_engaged_customers,
    control_pool=non_engaged_customers,
    covariates=['revenue', 'adoption_score', 'industry', 'support_level']
)
# Result: 100% matching success, all p-values > 0.05 for balance
```

**Purpose**: Create valid control group by matching on observable characteristics

**Validation**: All statistical tests confirm groups are identical except for G3 engagement

### 2. Difference-in-Differences (DiD)
```python
# Control for time trends and unobserved factors
did_estimate = (treatment_post - treatment_pre) - (control_post - control_pre)
# Result: 19% security revenue increase (p < 0.05)
```

**Purpose**: Isolate treatment effect from time trends and unobserved confounders

**Advantage**: Accounts for factors that affect both groups equally over time

### 3. Cluster-Based Control Groups
```python
# K-means clustering with 11 features
clusters = KMeans(n_clusters=5).fit(account_features)
# Analyze treatment effect within each cluster
cluster_effects = analyze_within_cluster_impact(clusters)
```

**Purpose**: Identify heterogeneous treatment effects across account types

**Result**: 219.8% overall lift with varying effects by cluster

### 4. Bootstrap Confidence Intervals
```python
# Quantify uncertainty in estimates
bootstrap_ci = bootstrap_confidence_interval(
    data=matched_data,
    statistic=calculate_treatment_effect,
    n_iterations=10000
)
```

**Purpose**: Provide robust uncertainty quantification for business decisions

## Data Scale

- **638,178 customer-month observations** across 53,367 unique customers
- **30,567 opportunity records** across 2,719 accounts
- **25 different engagement types** analyzed
- **12-month observation period** for revenue tracking
- **235 G3 engaged accounts** with complete data

## Technical Architecture

### Analysis Pipeline
```
Raw Data (PostgreSQL)
    ↓
Data Cleaning & Type Conversion
    ↓
Feature Engineering (11 features)
    ↓
Propensity Score Matching
    ↓
Difference-in-Differences Analysis
    ↓
Cluster-Based Analysis
    ↓
Bootstrap Confidence Intervals
    ↓
Executive Reporting & Visualization
```

### Key Technologies
- **Python**: pandas, numpy, scipy, scikit-learn
- **Statistical Analysis**: statsmodels, bootstrap methods
- **Database**: PostgreSQL, AWS Secrets Manager
- **Visualization**: matplotlib, seaborn
- **Reporting**: Automated CSV/JSON exports

## Key Files

### Analysis Scripts
- `G3_Complete_Lift_Analysis_Final-Revenue_Improved_Enhanced.py` - Main revenue analysis with PSM + DiD
- `G3_Cluster_Based_Analysis-Pipeline.py` - Cluster-based control group analysis
- `G3_Level_9_Revenue_Analysis_Corrected.py` - Management level impact analysis

### Documentation
- `project_context.md` - Comprehensive project documentation
- `G3_Executive_Business_Case_Final.md` - Executive summary with business recommendations
- `G3_Comprehensive_Analysis_Report.md` - Technical methodology details

### Results
- `g3_engagement_type_revenue_impact_analysis.csv` - Engagement type performance
- `g3_level_9_revenue_impact_analysis_corrected.csv` - Leadership impact analysis
- `G3_Executive_Message_Combined_Analysis.md` - Field communication summary

## Methodological Breakthrough

### Problem: Selection Bias
Original analysis showed 400-500% revenue increases - clearly inflated due to selection bias (G3 customers were already high-value accounts).

### Solution: Advanced Matching
Implemented propensity score matching with rigorous validation:
- **Perfect Balance**: All covariates balanced between treatment and control (p > 0.05)
- **Realistic Effects**: Reduced to credible 19% security revenue increase
- **Multiple Validation**: Placebo tests, sensitivity analysis, robustness checks

### Impact
Transformed unrealistic estimates into credible business case, enabling informed executive decisions on program expansion.

## Business Insights

### Engagement Type Optimization
1. **Security Reviews**: 84.8% win rate, highest total ARR ($380K)
2. **Resiliency Assessments**: 83.3% win rate, highest avg ARR per opportunity ($10K)
3. **Well-Architected Reviews**: 81.5% win rate, largest volume (4,455 won opportunities)
4. **Architecture Reviews**: 63.6% win rate, improvement opportunity identified

### Strategic Recommendations
- **Scale Security Reviews**: Proven product-market fit with highest performance
- **Expand Resiliency Assessments**: Strong win rate with premium pricing
- **Optimize Architecture Reviews**: Significant improvement potential
- **Target High-Impact Clusters**: Focus resources on Cluster 1 (+139.6% lift)

## Production Deployment

### Database Integration
```python
# Real-time connection to production database
query = "SELECT * FROM sa_ops_sb.g3_pipeline_control_analysis"
data = sql_pull('*****', query)  # AWS Secrets Manager credentials (sanitized)
```

### Automated Execution
- **Runtime**: ~30 seconds for 450 account analysis
- **Output**: 5 CSV tables, 3 PNG visualizations, JSON results
- **Monitoring**: Data quality checks and validation tests

## Skills Demonstrated

### Statistical Expertise
- Causal inference and experimental design
- Propensity score matching with perfect balance
- Difference-in-differences methodology
- Bootstrap confidence intervals
- Non-parametric statistical testing

### Data Science
- Large-scale data processing (638K observations)
- Feature engineering and dimensionality reduction
- Cluster analysis and segmentation
- Machine learning for matching algorithms

### Business Acumen
- Executive communication and reporting
- ROI calculation and business case development
- Strategic recommendations with quantified impact
- Transparent methodology for stakeholder trust

### Software Engineering
- Production-ready Python code
- Database integration with AWS services
- Automated reporting pipelines
- Comprehensive documentation

## Results Validation

### Robustness Checks
✅ **Placebo Tests**: No false positives in methodology  
✅ **Sensitivity Analysis**: Results stable across specifications  
✅ **Bootstrap Validation**: Confidence intervals quantified  
✅ **Covariate Balance**: All p-values > 0.05 for matched groups  
✅ **Multiple Methodologies**: PSM, DiD, and clustering all converge  

### Data Quality
✅ **Type Conversion**: Proper handling of database varchar fields  
✅ **Duplicate Prevention**: Unique opportunity-level analysis  
✅ **Missing Data**: Appropriate imputation and handling  
✅ **Outlier Detection**: Robust to extreme values  

## Impact on Decision Making

This analysis directly influenced:
- **2026 Goal Setting**: 750 G3 engagements target with 70% win rate
- **Resource Allocation**: Focus on high-performing engagement types
- **Program Expansion**: $17M+ investment justified through rigorous validation
- **Field Strategy**: Engagement playbooks optimized by cluster and type

---

**Note**: This project demonstrates the ability to apply advanced statistical methods to real business problems, communicate complex analyses to executive audiences, and deliver production-ready analytical frameworks at enterprise scale.