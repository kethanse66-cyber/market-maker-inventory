import duckdb
f = "E:/hl_export/orders/SOL/2026-09-10.parquet"
print(duckdb.sql(f"SELECT COUNT(*) AS events FROM '{f}'"))
