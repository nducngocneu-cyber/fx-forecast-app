-- US Treasury curve, wide (one column per maturity) -> long (one row per date x maturity).
CREATE OR REPLACE VIEW us_treasury_long AS
WITH maturities(col, tenor_y) AS (
    VALUES ('y_1m', 1.0 / 12), ('y_3m', 0.25), ('y_6m', 0.5), ('y_1y', 1.0), ('y_2y', 2.0),
           ('y_3y', 3.0), ('y_5y', 5.0), ('y_7y', 7.0), ('y_10y', 10.0), ('y_20y', 20.0),
           ('y_30y', 30.0)
)
SELECT t.date, m.tenor_y, t.yield_pct
FROM (UNPIVOT us_treasury ON COLUMNS('^y_') INTO NAME col VALUE yield_pct) AS t
JOIN maturities AS m USING (col)
WHERE t.yield_pct IS NOT NULL;
