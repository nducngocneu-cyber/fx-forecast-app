-- Last available US yield in each calendar month (to line up with VBMA's monthly VN data).
CREATE OR REPLACE VIEW us_month_end AS
SELECT last_day(date) AS month_end,
       tenor_y,
       arg_max(yield_pct, date) AS yield_pct,
       max(date)                AS obs_date
FROM us_treasury_long
GROUP BY ALL;
