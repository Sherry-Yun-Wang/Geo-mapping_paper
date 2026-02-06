import pandas as pd
import os
import time
import concurrent.futures
from tqdm import tqdm
from read_iqvia import read_iqvia_header

# Load obesity and T2D codes
obesity_codes = set(pd.read_csv('/home/stofer@chapman.edu/merck_proposal/glp_pats/obesity_codes.csv')['code'].astype(str))
t2d_codes = set(pd.read_csv('/home/stofer@chapman.edu/merck_proposal/glp_pats/t2d_codes.csv')['code'].astype(str))

def classify_row(diags):
    patient_codes = set(filter(pd.notna, diags))  # drop NaNs
    has_obesity = not patient_codes.isdisjoint(obesity_codes)
    has_t2d = not patient_codes.isdisjoint(t2d_codes)

    if has_obesity and has_t2d:
        return "Both"
    elif has_obesity:
        return "Obesity"
    elif has_t2d:
        return "T2D"
    else:
        return None  # filter out later

def read_iqvia_claims(year):
    use_cols = [0, 19, *range(22, 34)]  # pat_id, to_dt, diag1–12
    claims_folder = '/sharefolder/IQVIA'
    year_folder = os.path.join(claims_folder, f'claims_{year}')
    csv_in_parts_folder = os.path.join(year_folder, 'csv_in_parts')
    csv_files = [file for file in os.listdir(csv_in_parts_folder) if file.endswith('.csv')]

    filtered_data_list = []

    for csv_file in tqdm(csv_files, desc=f"Year {year}"):
        file_path = os.path.join(csv_in_parts_folder, csv_file)
        data_part = pd.read_csv(file_path, sep='|', header=None, dtype=str, usecols=use_cols)
        data_part.columns = ["pat_id", "to_dt", "diag1", "diag2", "diag3", "diag4", 
                             "diag5", "diag6", "diag7", "diag8", "diag9", "diag10", "diag11", "diag12"]

        data_part['to_dt'] = pd.to_datetime(data_part['to_dt'], errors='coerce')

        # NEW: restrict to the current year only
        data_part = data_part[data_part['to_dt'].dt.year == int(year)]

        # classify each row
        diag_cols = [f"diag{i}" for i in range(1, 13)]
        data_part['condition'] = data_part[diag_cols].apply(lambda row: classify_row(row.values), axis=1)

        # keep only patients with a relevant condition
        data_part = data_part.dropna(subset=['condition'])

        if not data_part.empty:
            filtered_data_list.append(data_part)

    return pd.concat(filtered_data_list, ignore_index=True) if filtered_data_list else pd.DataFrame()

def process_year(year):
    start_time = time.time()
    iqvia_data = read_iqvia_claims(year)

    if iqvia_data.empty:
        print(f"No valid patients found for year {year}", flush=True)
        return

    selected_columns = ["pat_id", "to_dt", "condition"]
    iqvia_data = iqvia_data[selected_columns]

    output_path = f'/home/stofer@chapman.edu/federated_analysis/tableau/data/iqvia_pat_{year}.csv'
    iqvia_data.to_csv(output_path, index=False)

    elapsed_time = time.time() - start_time
    print(f"Year {year} completed in {elapsed_time:.2f} seconds", flush=True)

def main():
    years = [str(year) for year in range(2010, 2023)]
    read_iqvia_header()  # still call if needed for consistency

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_year, year): year for year in years}
        for future in concurrent.futures.as_completed(futures):
            year = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing year {year}: {e}", flush=True)

if __name__ == "__main__":
    main()
