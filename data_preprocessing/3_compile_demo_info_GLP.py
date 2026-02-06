import pandas as pd

# Your mapping dictionary
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

# -------------------------------
# Step 1: Load the payment_type.csv
# -------------------------------
payment_df = pd.read_csv("payment_type.csv", dtype=str)

# Step 2: Map pay_type codes to full descriptions
payment_df["pay_type"] = payment_df["pay_type"].map(insurance_map)

# -------------------------------
# Step 3: Load GLP1_pat_states.csv
# -------------------------------
glp1_df = pd.read_csv("/home/stofer@chapman.edu/federated_analysis/GLP1_pat_states.csv", dtype=str)

# -------------------------------
# Step 4: Merge on pat_key
# -------------------------------
merged_df = glp1_df.merge(payment_df[["pat_key", "pay_type"]], on="pat_key", how="left")


pat_conditions = pd.read_csv(
    "/home/stofer@chapman.edu/merck_proposal/icd_pats/patient_conditions.csv"
)  # columns: pat_key, condition

# Merge to attach condition to each patient
df = merged_df.merge(pat_conditions, on="pat_key", how="inner")


# -------------------------------
# Step 5: Save final results
# -------------------------------
df.to_csv("tableau_data_prep.csv", index=False)

print("✅ Saved merged dataset as tableau_data.csv")
