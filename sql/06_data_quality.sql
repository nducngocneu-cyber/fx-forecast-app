-- What the cleaning steps changed, by dataset and year.
SELECT 'USD/VND daily (Yahoo)' AS dataset, year(date) AS year,
       count(*) AS observations, count(*) FILTER (WHERE flagged) AS values_replaced
FROM fx_usdvnd GROUP BY ALL
UNION ALL
SELECT 'VN govt yields monthly (VBMA)', year(month_end),
       count(*), count(*) FILTER (WHERE abs(yield_pct - published_pct) > 1e-9)
FROM vn_yields GROUP BY ALL
ORDER BY dataset, year;
