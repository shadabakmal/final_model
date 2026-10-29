import pandas as pd
import numpy as np
from pathlib import Path

# Load your data
data_path = Path(__file__).resolve().parent / "DATA" / "combined_reduction_degree_dataset_CLEAN.csv"
df = pd.read_csv(data_path)

# 1. Create the new column. Default everything to Hematite.
df['ore_type'] = 'Hematite'

# 2. Change the specific Magnetite papers
df.loc[df['meta_source'].isin(['Zheng2023', 'Lu2024']), 'ore_type'] = 'Magnetite'

# 3. Change the Titanomagnetite paper
df.loc[df['meta_source'] == 'Zhang2020', 'ore_type'] = 'Titanomagnetite'

# 4. Verify the counts
print("Ore Type Counts in your dataset:")
print(df['ore_type'].value_counts())

# Save it back
df.to_csv(data_path, index=False)
print("\nSuccessfully added 'ore_type' column and saved!")