import pandas as pd

df = pd.read_csv("../data/timesData.csv")

print("=" * 60)
print("GENERAL INFO")
print("=" * 60)
print(f"Lines:                        {df.shape[0]}")
print(f"Columns:                      {df.shape[1]}")
print(f"Years:                        {sorted(df['year'].unique())}")
print(f"Num of unique universities:   {df['university_name'].nunique()}")
print(f"Num of unique countries:      {df['country'].nunique()}")

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("FIRST FIVE LINES")
print("=" * 60)
print(df.head())

print("\n" + "=" * 60)
print("MISSED VALUES")
print("=" * 60)

for col in df.columns:
    nan_count   = df[col].isna().sum()
    dash_count  = (df[col].astype(str).str.strip() == "-").sum()
    total       = nan_count + dash_count
    pct         = total / len(df) * 100
    print(f"{col:<30} NaN: {nan_count:>4}   -: {dash_count:>4}   total: {total:>4}  ({pct:.1f}%)")

print("\n" + "=" * 60)
print("PROBLEM VALUES IN world_rank")
print("=" * 60)

mask_ranges = df["world_rank"].astype(str).str.contains("-", na=False)
print(f"Lines with ranges (e.g. '201-250'): {mask_ranges.sum()}")
print(df[mask_ranges]["world_rank"].value_counts().head(10))

print("\n" + "=" * 60)
print("PROBLEM VALUES IN num_students")
print("=" * 60)

mask_comma = df["num_students"].astype(str).str.contains(",", na=False)
print(f"Lines with commas (e.g. '20,152'): {mask_comma.sum()}")
print(df[mask_comma]["num_students"].head(5))

print("\n" + "=" * 60)
print("PROBLEM VALUES IN international_students")
print("=" * 60)

mask_pct = df["international_students"].astype(str).str.contains("%", na=False)
print(f"Lines with percents (e.g. '25%'): {mask_pct.sum()}")
print(df[mask_pct]["international_students"].head(5))

print("\n" + "=" * 60)
print("PROBLEM VALUES IN female_male_ratio")
print("=" * 60)

print(f"Num of unique formats:  {df['female_male_ratio'].nunique()}")
print(f"Gaps:                   {df['female_male_ratio'].isna().sum()}")
print("Examples:")
print(df["female_male_ratio"].dropna().head(5).values)

print("\n" + "=" * 60)
print("NUM OF RECORDS BY YEAR")
print("=" * 60)
print(df.groupby("year").size().reset_index(name="count"))

print("\n" + "=" * 60)
print("TOP 10 COUNTRIES BY RECORDS")
print("=" * 60)
print(df["country"].value_counts().head(10))