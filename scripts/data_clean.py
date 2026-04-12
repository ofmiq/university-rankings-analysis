import pandas as pd
import numpy as np

df = pd.read_csv("../data/timesData.csv")

mask_numeric = ~df["world_rank"].astype(str).str.contains("-", na=False)
df = df[mask_numeric].copy()

df["world_rank"] = pd.to_numeric(df["world_rank"], errors="coerce")

for col in df.columns:
    df[col] = df[col].replace("-", np.nan)

numeric_cols = ["teaching", "international", "research", 
                "citations", "income", "total_score"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["international_students"] = (
    df["international_students"]
    .astype(str)
    .str.replace("%", "", regex=False)
    .pipe(pd.to_numeric, errors="coerce")
)

keep_cols = [
    "world_rank",
    "university_name",
    "country",
    "teaching",
    "international",
    "research",
    "citations",
    "income",
    "total_score",
    "international_students",
    "year",
]

df = df[keep_cols].copy()

df_2015 = df[df["year"] == 2015].copy().reset_index(drop=True)

df_all = df.copy().reset_index(drop=True)

print("=" * 60)
print("CLEANING RESULT")
print("=" * 60)

for name, frame in [("df_all", df_all), ("df_2015", df_2015)]:
    print(f"\n{name}:")
    print(f"  Lines:    {len(frame)}")
    print(f"  Columns:  {frame.shape[1]}")
    print(f"  Gaps:")
    missing = frame.isnull().sum()
    missing = missing[missing > 0]
    for col, cnt in missing.items():
        pct = cnt / len(frame) * 100
        print(f"    {col:<30} {cnt:>4}  ({pct:.1f}%)")

print("\n" + "=" * 60)
print("DATA TYPES df_all")
print("=" * 60)
print(df_all.dtypes)

print("\n" + "=" * 60)
print("VALUES RANGE (df_2015)")
print("=" * 60)

metrics = ["teaching", "research", "citations", "international", "income"]
for col in metrics:
    print(f"  {col:<20} min: {df_2015[col].min():.1f}   "
          f"max: {df_2015[col].max():.1f}   "
          f"NaN: {df_2015[col].isna().sum()}")
df_all.to_csv("../data/times_clean_all.csv", index=False)
df_2015.to_csv("../data/times_clean_2015.csv", index=False)


print("\n" + "=" * 60)
print("TOP 10 COUNTRIES BY RECORDS (2015)")
print("=" * 60)
print(df_2015["country"].value_counts().head(10))

print("\n" + "=" * 60)
print("FILES WERE SAVED")
print("=" * 60)

mask_nan_rank = df_all["world_rank"].isna()
print(df_all[mask_nan_rank]["world_rank"].value_counts())
print(df_all[mask_nan_rank][["world_rank", "university_name", "year"]].head(10))
