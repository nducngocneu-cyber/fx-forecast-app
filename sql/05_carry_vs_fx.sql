-- Uncovered interest parity check: does the VN-US 1Y yield gap predict the next 12 months of USD/VND?
-- UIP says the higher-yielding currency should depreciate by about the yield gap.
WITH gap AS (
    SELECT v.month_end, v.yield_pct - u.yield_pct AS vn_minus_us_1y
    FROM vn_yields v
    JOIN us_month_end u ON u.month_end = v.month_end AND u.tenor_y = v.tenor_y
    WHERE v.tenor_y = 1
),
fx AS (
    SELECT last_day(date) AS month_end, arg_max(close_clean, date) AS spot
    FROM fx_usdvnd GROUP BY ALL
),
joined AS (
    SELECT g.month_end, g.vn_minus_us_1y, f.spot,
           100 * (lead(f.spot, 12) OVER (ORDER BY g.month_end) / f.spot - 1) AS usdvnd_next_12m_pct
    FROM gap g JOIN fx f USING (month_end)
)
SELECT month_end,
       round(vn_minus_us_1y, 3)        AS vn_minus_us_1y_pct,
       round(usdvnd_next_12m_pct, 2)   AS usdvnd_next_12m_pct,
       round(usdvnd_next_12m_pct - vn_minus_us_1y, 2) AS uip_deviation_pct
FROM joined
WHERE usdvnd_next_12m_pct IS NOT NULL
ORDER BY month_end;
