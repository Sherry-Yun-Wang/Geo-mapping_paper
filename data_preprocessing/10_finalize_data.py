import pandas as pd

# -----------------------------
# Paths
# -----------------------------
input_file = "/home/stofer@chapman.edu/federated_analysis/tableau/payment_almost_all_filled.csv"
output_file = "/home/stofer@chapman.edu/federated_analysis/tableau/FINAL_DATA.csv"

# -----------------------------
# Mapping dictionary
# -----------------------------
insurance_map = {
    "A": "Medicare Part C",
    "C": "Commercial",
    "K": "State Children's Health Insurance Program (SCHIP)",
    "M": "Medicaid",
    "R": "Medicare Risk",
    "S": "Self-Insured",
    "T": "Medicare Cost",
    "U": "Unknown/Missing",
    "X": "RX Only"
}

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(input_file, dtype=str)
initial_rows = len(df)

# -----------------------------
# 1. Map pay_type
# -----------------------------
# Only replace if not already mapped
df['pay_type'] = df['pay_type'].map(insurance_map).fillna(df['pay_type'])

# -----------------------------
# 2. Drop rows with ANY NA values
# -----------------------------
before_drop = len(df)
df = df.dropna()
after_drop = len(df)

print(f"Dropped {before_drop - after_drop} rows due to NA values")
# -----------------------------
# 3. Drop 'zip' column if it exists
# -----------------------------
if 'zip' in df.columns:
    df = df.drop(columns=['zip'])
    print("Dropped column: zip")

# -----------------------------
# 4. Save final file
# -----------------------------
df.to_csv(output_file, index=False)

print(f"Final dataset saved: {output_file}")
print(f"Initial rows: {initial_rows}, Final rows: {after_drop}, Columns: {len(df.columns)}")
