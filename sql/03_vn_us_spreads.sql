-- Curve shape and the VN-US government spread, month by month.
SELECT v.month_end,
       max(v.yield_pct) FILTER (WHERE v.tenor_y = 10)                                   AS vn_10y,
       max(u.yield_pct) FILTER (WHERE u.tenor_y = 10)                                   AS us_10y,
       round(100 * (max(v.yield_pct) FILTER (WHERE v.tenor_y = 10)
                  - max(u.yield_pct) FILTER (WHERE u.tenor_y = 10)), 1)                  AS vn_minus_us_10y_bp,
       round(100 * (max(v.yield_pct) FILTER (WHERE v.tenor_y = 10)
                  - max(v.yield_pct) FILTER (WHERE v.tenor_y = 2)), 1)                   AS vn_2s10s_bp,
       round(100 * (max(u.yield_pct) FILTER (WHERE u.tenor_y = 10)
                  - max(u.yield_pct) FILTER (WHERE u.tenor_y = 2)), 1)                   AS us_2s10s_bp
FROM vn_yields v
JOIN us_month_end u ON u.month_end = v.month_end AND u.tenor_y = v.tenor_y
GROUP BY v.month_end
ORDER BY v.month_end;
