import pandas as pd

csv_files = [
    "patient_year_filled.csv",
    "payment_type_filled.csv",
    "payment_type_filled_with_condition.csv",
    "payment_almost_all_filled.csv",
    "FINAL_DATA.csv"
]

for file in csv_files:
    df = pd.read_csv(file)
    percent_by_year = df['year'].value_counts().sort_index() #/ df.shape[0] * 100
    print(f"\n{file} percentages by year:")
    print(percent_by_year)
