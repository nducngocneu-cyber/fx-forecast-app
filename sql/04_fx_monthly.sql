-- Month-end USD/VND, monthly change and 3-month realised volatility from daily log returns.
WITH daily AS (
    SELECT date, close_clean,
           ln(close_clean / lag(close_clean) OVER (ORDER BY date)) AS log_ret
    FROM fx_usdvnd
),
monthly AS (
    SELECT last_day(date)              AS month_end,
           arg_max(close_clean, date)  AS usdvnd_close,
           stddev_samp(log_ret)        AS daily_vol,
           count(*)                    AS n_days
    FROM daily
    GROUP BY ALL
)
SELECT month_end,
       usdvnd_close,
       round(100 * (usdvnd_close / lag(usdvnd_close) OVER w - 1), 3)               AS change_pct,
       round(100 * sqrt(252) * sqrt(avg(daily_vol * daily_vol)
             OVER (w ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)), 2)                 AS vol_3m_ann_pct
FROM monthly
WINDOW w AS (ORDER BY month_end)
ORDER BY month_end;
