-- Verification queries for the quantities reported in main.tex
-- Source: powerlaw_results.db (SQLite), produced by fitpl.py
--
-- Conventions (see fitpl.py, powerlaw.Fit.distribution_compare):
--   R > 0                 -> power law preferred over exponential
--   R < 0                 -> exponential preferred over power law
--   p                     -> significance of the comparison
--   xmin                  -> lower bound selected by KS minimisation
--   year = 0              -> POOLED all-years fit, NOT a calendar year

-- ---------------------------------------------------------------
-- 1. The main table (Table 1 of the manuscript)
-- ---------------------------------------------------------------
-- L1: weekly cases, city, full series
SELECT 'L1' AS level, COUNT(*) AS n,
       ROUND(AVG(alpha),3) AS alpha_mean,
       ROUND((SELECT alpha FROM powerlaw_fits ORDER BY alpha
              LIMIT 1 OFFSET (SELECT COUNT(*)/2 FROM powerlaw_fits)),3) AS alpha_median,
       ROUND(AVG(xmin),1) AS xmin_median,
       ROUND(100.0*AVG(CASE WHEN xmin <= 5 THEN 1 ELSE 0 END),1) AS pct_xmin_le5,
       ROUND(100.0*AVG(CASE WHEN R > 0 THEN 1 ELSE 0 END),1) AS pl_preferred,
       ROUND(100.0*AVG(CASE WHEN R > 0 AND p < 0.05 THEN 1 ELSE 0 END),1) AS pl_significant,
       ROUND(100.0*AVG(CASE WHEN R < 0 AND p < 0.05 THEN 1 ELSE 0 END),1) AS exp_significant
FROM powerlaw_fits
UNION ALL
SELECT 'L2', COUNT(*), ROUND(AVG(alpha),3), NULL, ROUND(AVG(xmin),1),
       ROUND(100.0*AVG(CASE WHEN xmin <= 5 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 AND p < 0.05 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R < 0 AND p < 0.05 THEN 1 ELSE 0 END),1)
FROM powerlaw_fits_yearly
UNION ALL
SELECT 'L3', COUNT(*), ROUND(AVG(alpha),3), NULL, ROUND(AVG(xmin),1),
       ROUND(100.0*AVG(CASE WHEN xmin <= 5 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 AND p < 0.05 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R < 0 AND p < 0.05 THEN 1 ELSE 0 END),1)
FROM powerlaw_fits_total_cases
UNION ALL
SELECT 'L4', COUNT(*), ROUND(AVG(alpha),3), NULL, ROUND(AVG(xmin),1),
       ROUND(100.0*AVG(CASE WHEN xmin <= 5 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 AND p < 0.05 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R < 0 AND p < 0.05 THEN 1 ELSE 0 END),1)
FROM episcanner_state_fits WHERE metric='total_cases' AND year > 0
UNION ALL
SELECT 'L5', COUNT(*), ROUND(AVG(alpha),3), NULL, ROUND(AVG(xmin),1),
       ROUND(100.0*AVG(CASE WHEN xmin <= 5 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 AND p < 0.05 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R < 0 AND p < 0.05 THEN 1 ELSE 0 END),1)
FROM episcanner_state_fits WHERE metric='ep_dur' AND year > 0
UNION ALL
SELECT 'L6', COUNT(*), ROUND(AVG(alpha),3), NULL, ROUND(AVG(xmin),1),
       ROUND(100.0*AVG(CASE WHEN xmin <= 5 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 AND p < 0.05 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R < 0 AND p < 0.05 THEN 1 ELSE 0 END),1)
FROM episcanner_fits WHERE metric='total_cases' AND year > 0
UNION ALL
SELECT 'L7', COUNT(*), ROUND(AVG(alpha),3), NULL, ROUND(AVG(xmin),1),
       ROUND(100.0*AVG(CASE WHEN xmin <= 5 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R > 0 AND p < 0.05 THEN 1 ELSE 0 END),1),
       ROUND(100.0*AVG(CASE WHEN R < 0 AND p < 0.05 THEN 1 ELSE 0 END),1)
FROM episcanner_fits WHERE metric='ep_dur' AND year > 0;

-- ---------------------------------------------------------------
-- 2. Stability of the exponent across states (L4)
-- ---------------------------------------------------------------
SELECT state, COUNT(*) AS n_years, ROUND(AVG(alpha),3) AS alpha_state,
       ROUND(100.0*AVG(CASE WHEN R > 0 THEN 1 ELSE 0 END),1) AS pl_preferred,
       ROUND(100.0*AVG(CASE WHEN R > 0 AND p < 0.05 THEN 1 ELSE 0 END),1) AS pl_significant
FROM episcanner_state_fits
WHERE metric='total_cases' AND year > 0
GROUP BY state ORDER BY alpha_state;

-- ---------------------------------------------------------------
-- 3. Year-by-year national fits (L6 vs L7) -- the headline contrast
-- ---------------------------------------------------------------
SELECT year, metric, ROUND(alpha,3) AS alpha, ROUND(xmin,1) AS xmin,
       ROUND(R,3) AS R, ROUND(p,4) AS p
FROM episcanner_fits
WHERE year > 0
ORDER BY metric, year;

-- ---------------------------------------------------------------
-- 4. Correlation of alpha with size (L1)
-- ---------------------------------------------------------------
SELECT COUNT(*) AS n,
       ROUND((AVG(lnpop*alpha) - AVG(lnpop)*AVG(alpha)) /
             (SQRT(AVG(lnpop*lnpop)-AVG(lnpop)*AVG(lnpop)) *
              SQRT(AVG(alpha*alpha)-AVG(alpha)*AVG(alpha))), 4) AS pearson_r
FROM (SELECT f.alpha, LN(CAST(c.population AS REAL)) AS lnpop
      FROM powerlaw_fits f JOIN city_stats c USING (geocode)
      WHERE c.population > 0 AND f.alpha IS NOT NULL);
