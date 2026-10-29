import pandas as pd

df = pd.read_csv("combined_reduction_degree_dataset.csv")

KEEP = [
    "meta_source", "reductant", "heating_mode", "bed_type",
    "temperature_c", "time_min", "basicity", "gangue_pct",
    "reduction_degree_pct", "cpr",
    "h2_frac", "co_frac", "ch4_frac", "porosity_pct",
]

clean = df[KEEP].copy()
clean = clean.dropna(subset=["reduction_degree_pct"])
clean = clean[(clean["reduction_degree_pct"] >= 0) &
              (clean["reduction_degree_pct"] <= 100)]

clean.to_csv("DATA/combined_reduction_degree_dataset_CLEAN.csv", index=False)
print(f"Wrote {len(clean)} rows x {len(KEEP)} columns")
print(f"Papers: {clean['meta_source'].nunique()}")
print(f"Reductants: {clean['reductant'].value_counts().to_dict()}")