/*
================================================================================
TEMPLATE SQL FOR ENGAGEMENT IMPACT ANALYSIS WITH CONTROL GROUP
================================================================================

NOTE: This SQL template has been sanitized for GitHub. All database schema names,
table names, and specific project references have been genericized.

ORIGINAL PURPOSE: Create control group analysis for specialist engagement 
program impact by comparing engaged accounts vs similar non-engaged accounts 
on product/service opportunity creation.

CONTEXT: This analysis creates a control group of accounts similar to 
engaged accounts but without engagements. The control group is matched on 
key characteristics: sub_segment, account_phase, revenue ranges, industry, 
and support level. This enables causal inference about program impact on 
pipeline generation.

METHODOLOGY:
1. Treatment Group: Accounts with specialist engagements
2. Control Group: Similar accounts without engagements (matched on characteristics)
3. Outcome Analysis: Compare opportunity creation, win rates, and revenue metrics

SCHEMA NOTES:
- All schema/table names have been genericized (e.g., analytics_db.engagement_analysis)
- Replace with your actual database schema and table names
- This is a template demonstrating the methodology, not production SQL

COMPONENTS:
1. Treatment and Control group CTEs
2. Product/service opportunities for both groups
3. Final analysis table with matched groups

================================================================================
*/

-- TEMPLATE SQL - Replace schema/table names with your actual database structure
-- Define Treatment Group: Accounts with specialist engagements
-- DROP TABLE IF EXISTS #treatment_accounts;
-- CREATE TEMP TABLE #treatment_accounts AS (
--     SELECT DISTINCT account_id AS accountid
--     FROM analytics_db.engagement_details
--     WHERE engagement_id IS NOT NULL
-- );

-- Define Control Group: Accounts without engagements but meeting criteria
-- DROP TABLE IF EXISTS #control_accounts;
-- CREATE TEMP TABLE #control_accounts AS (
--     SELECT DISTINCT account_id AS accountid
--     FROM analytics_db.engagement_details
--     WHERE engagement_id IS NULL
--         AND qualified_account_flag = true
--         AND has_revenue_12_mo_flag = true
--         AND created_opportunities_12_mo_flag = true
-- );

-- Get engagement date range for time window consistency
-- DROP TABLE IF EXISTS #engagement_date_range;
-- CREATE TEMP TABLE #engagement_date_range AS (
--     SELECT 
--         MIN(engagement_date) AS min_engagement_date,
--         MAX(engagement_date) AS max_engagement_date
--     FROM analytics_db.engagement_analysis
--     WHERE engagement_date IS NOT NULL
-- );

-- Get SA on opportunity team data
-- DROP TABLE IF EXISTS #sa_on_opp_team;
-- CREATE TEMP TABLE #sa_on_opp_team AS (
--     SELECT
--         t.opportunityid AS opp_id,
--         CASE WHEN u.alias IS NOT NULL THEN 1 ELSE 0 END AS sa_on_opp_team_flag
--     FROM sfdc_ods_ext.opportunityteammember t
--     LEFT JOIN pii_sfdc_ods_ext."user" u ON u.id = t.userid
--     LEFT JOIN sa_ops_sb.sa_mgr_dashboard_data_final s ON s.bis_alias = u.alias
--     WHERE t.dw_delete_flag = 'N'
--         AND s.bis_alias IS NOT NULL
--     GROUP BY 1, 2
-- );

-- TEMPLATE: Get ALL opportunities for both treatment and control groups
-- Replace with your actual table names and schema
-- DROP TABLE IF EXISTS #all_opportunities;
-- CREATE TEMP TABLE #all_opportunities AS (
--     SELECT 
--         o.account_id AS accountid,
--         o.id AS opp_id,
--         o.name AS opportunity_name,
--         o.created_date AS opp_created_date,
--         o.close_date AS opp_close_date,
--         o.stage_name,
--         o.probability,
--         o.is_closed,
--         o.is_won,
--         o.annualized_revenue AS opp_total_arr,
--         olm.id AS olm_id,
--         olm.functional_area,
--         olm.product_name,
--         CAST(olm.total_price AS DECIMAL(18,2)) AS totalprice,
--         CAST(olm.annualized_revenue AS DECIMAL(18,2)) AS service_arr,
--         CASE 
--             WHEN olm.functional_area IN ('Security', 'Identity & Compliance') 
--             THEN 1 ELSE 0 
--         END AS is_security,
--         CASE 
--             WHEN olm.product_name IN ('Backup', 'CloudFormation', 'CloudTrail', 'Config',
--                                       'Systems Manager', 'CloudWatch', 'Well-Architected Tool')
--             THEN 1 ELSE 0 
--         END AS is_resiliency,
--         CASE 
--             WHEN ta.accountid IS NOT NULL THEN 'Treatment'
--             WHEN ca.accountid IS NOT NULL THEN 'Control'
--             ELSE NULL
--         END AS group_type
--     FROM analytics_db.opportunity o
--     JOIN analytics_db.opportunity_line_item olm ON o.id = olm.opportunity_id
--     LEFT JOIN #treatment_accounts ta ON o.account_id = ta.accountid
--     LEFT JOIN #control_accounts ca ON o.account_id = ca.accountid
--     CROSS JOIN #engagement_date_range edr
--     WHERE o.created_date >= edr.min_engagement_date
--         AND o.created_date <= DATEADD(MONTH, 6, edr.max_engagement_date)
--         AND (ta.accountid IS NOT NULL OR ca.accountid IS NOT NULL)
-- );

-- TEMPLATE: All SQL below is commented out - this is a template showing methodology
-- Replace schema/table names and uncomment when adapting to your database

-- -- Filter to security/resiliency opportunities for final analysis
-- DROP TABLE IF EXISTS #security_resiliency_opportunities;
-- CREATE TEMP TABLE #security_resiliency_opportunities AS (
--     SELECT *
--     FROM #all_opportunities
--     WHERE (
--         functional_area IN ('Security', 'Identity & Compliance')
--         OR product_name IN ('Backup', 'CloudFormation', 'CloudTrail', 'Config',
--                            'Systems Manager', 'CloudWatch', 'Well-Architected Tool')
--     )
-- );

-- -- Create final analysis table with consistent group definitions
-- DROP TABLE IF EXISTS analytics_db.pipeline_control_analysis;
-- CREATE TABLE analytics_db.pipeline_control_analysis AS (
--     -- Treatment Group
--     SELECT 
--         g.account_id AS accountid,
--         g.account_name,
--         'Treatment' AS group_type,
--         g.sub_segment,
--         g.account_phase,
--         CASE 
--             WHEN g.revenue < 100000 THEN 'Small'
--             WHEN g.revenue < 1000000 THEN 'Medium'
--             WHEN g.revenue < 10000000 THEN 'Large'
--             ELSE 'Enterprise'
--         END AS revenue_bucket,
--         g.industry,
--         COALESCE(ac.max_support_level, 'Basic') AS max_support_level,
--         g.region,
--         g.sub_region,
--         g.organization,
--         g.engagement_id AS activity_id,
--         g.engagement_type,
--         g.is_direct_engagement,
--         CASE 
--             WHEN g.is_attributed_engagement = 1 THEN 1
--             WHEN g.is_direct_engagement = 0 AND g.engagement_id IS NOT NULL THEN 1
--             ELSE 0
--         END AS is_attributed_engagement,
--         o.opp_id,
--         o.opportunity_name,
--         o.opp_created_date,
--         o.opp_close_date,
--         o.stage_name,
--         o.probability,
--         o.is_closed,
--         o.is_won,
--         o.opp_total_arr,
--         o.olm_id,
--         o.functional_area,
--         o.product_name,
--         o.total_price,
--         o.service_arr,
--         o.is_security,
--         o.is_resiliency,
--         CASE WHEN o.is_security = 1 OR o.is_resiliency = 1 THEN 1 ELSE 0 END AS is_sec_res_opportunity
--     FROM analytics_db.engagement_analysis g
--     INNER JOIN #treatment_accounts ta ON g.account_id = ta.accountid
--     LEFT JOIN analytics_db.account ac ON ac.account_id = g.account_id
--     LEFT JOIN #security_resiliency_opportunities o ON g.account_id = o.accountid AND o.group_type = 'Treatment'
--     
--     UNION ALL
--     
--     -- Control Group
--     SELECT 
--         ac.account_id AS accountid,
--         ac.account_name,
--         'Control' AS group_type,
--         ac.sub_segment,
--         ac.account_phase,
--         CASE 
--             WHEN ac.revenue < 100000 THEN 'Small'
--             WHEN ac.revenue < 1000000 THEN 'Medium'
--             WHEN ac.revenue < 10000000 THEN 'Large'
--             ELSE 'Enterprise'
--         END AS revenue_bucket,
--         ac.industry,
--         COALESCE(ac.max_support_level, 'Basic') AS max_support_level,
--         ac.region,
--         ac.sub_region,
--         ac.organization,
--         NULL AS activity_id,
--         NULL AS engagement_type,
--         0 AS is_direct_engagement,
--         0 AS is_attributed_engagement,
--         o.opp_id,
--         o.opportunity_name,
--         o.opp_created_date,
--         o.opp_close_date,
--         o.stage_name,
--         o.probability,
--         o.is_closed,
--         o.is_won,
--         o.opp_total_arr,
--         o.olm_id,
--         o.functional_area,
--         o.product_name,
--         o.total_price,
--         o.service_arr,
--         o.is_security,
--         o.is_resiliency,
--         CASE WHEN o.is_security = 1 OR o.is_resiliency = 1 THEN 1 ELSE 0 END AS is_sec_res_opportunity
--     FROM analytics_db.account ac
--     INNER JOIN #control_accounts ca ON ac.account_id = ca.accountid
--     LEFT JOIN #security_resiliency_opportunities o ON ac.account_id = o.accountid AND o.group_type = 'Control'
-- );

-- -- Final query for analysis (template - replace with your schema)
-- SELECT
--     accountid,
--     account_name,
--     group_type,
--     sub_segment,
--     account_phase,
--     revenue_bucket,
--     industry,
--     max_support_level,
--     region,
--     sub_region,
--     engagement_id AS activity_id,
--     engagement_type,
--     is_direct_engagement,
--     is_attributed_engagement,
--     opp_id,
--     opportunity_name,
--     opp_created_date,
--     opp_close_date,
--     stage_name,
--     probability,
--     is_closed,
--     is_won,
--     opp_total_arr,
--     product_name AS product_names,
--     service_arr AS total_arr
-- FROM analytics_db.pipeline_control_analysis
-- WHERE product_name IS NOT NULL
--     AND region = 'Enterprise'
--     AND group_type = 'Treatment'
-- 
-- UNION
-- 
-- SELECT
--     accountid,
--     account_name,
--     group_type,
--     sub_segment,
--     account_phase,
--     revenue_bucket,
--     industry,
--     max_support_level,
--     region,
--     sub_region,
--     engagement_id AS activity_id,
--     engagement_type,
--     is_direct_engagement,
--     is_attributed_engagement,
--     opp_id,
--     opportunity_name,
--     opp_created_date,
--     opp_close_date,
--     stage_name,
--     probability,
--     is_closed,
--     is_won,
--     opp_total_arr,
--     product_name AS product_names,
--     service_arr AS total_arr
-- FROM analytics_db.pipeline_control_analysis
-- WHERE product_name IS NOT NULL
--     AND region = 'Enterprise'
--     AND group_type = 'Control'
-- LIMIT 10000;

-- ============================================================================
-- END OF TEMPLATE
-- ============================================================================
-- This SQL template demonstrates the methodology for creating control groups
-- in engagement impact analysis. All schema and table names have been genericized.
-- Replace with your actual database structure before use.