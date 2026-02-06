import pandas as pd
import os
import concurrent.futures

def read_enroll(year_pat_date):
    year, pat_date_list = year_pat_date
    claims_folder = '/sharefolder/IQVIA'  
    year_folder = os.path.join(claims_folder, f'enroll2_{year}')
    csv_in_parts_folder = os.path.join(year_folder, 'csv_in_parts')
    csv_files = [file for file in os.listdir(csv_in_parts_folder) if file.endswith('.csv')]

    pat_date_df = pd.DataFrame(pat_date_list, columns=['pat_key', 'month_id'])
    pat_date_df['pat_id'] = pat_date_df['pat_key'].str.extract(r'^(.*)_\d+$')

    pat_date_set = set(pat_date_df[['pat_id', 'month_id']].apply(tuple, axis=1))
    header_data = ['pat_id', 'mstr_enroll_cd', 'prd_type', 'pay_type', 'pcob_type', 'mcob_type', 'month_id']
    filtered_data_list = []

    for i, csv_file in enumerate(csv_files, 1):
        file_path = os.path.join(csv_in_parts_folder, csv_file)
        data_part = pd.read_csv(file_path, sep='|', header=None, dtype=str)
        data_part.columns = header_data

        data_part = data_part[data_part[['pat_id', 'month_id']].apply(tuple, axis=1).isin(pat_date_set)]
        filtered_data_list.append(data_part)
        print(f"[{year}] Appended part {i}/{len(csv_files)}", flush=True)

    if filtered_data_list:
        result = pd.concat(filtered_data_list, ignore_index=True)
        # Merge back to restore pat_key
        result = result.merge(pat_date_df[['pat_key', 'pat_id', 'month_id']], on=['pat_id', 'month_id'], how='left')
        return result
    else:
        return pd.DataFrame(columns=header_data + ['pat_key'])
    
# Load condition + filter all_data
all_data = pd.read_csv('/home/stofer@chapman.edu/federated_analysis/GLP1_pat_states.csv')

# Extract index_date and year
all_data['index_date'] = pd.to_datetime(all_data['index_date'])
all_data['index_date'] = all_data['index_date'].dt.strftime('%Y%m')
all_data['index_year'] = all_data['index_date'].str[:4]

# Map year -> list of (pat_id, month_id)
year_month_pat_id = all_data[['index_year', 'index_date', 'pat_key']].drop_duplicates()
year_to_pat_date = (
    year_month_pat_id.groupby('index_year')
    .apply(lambda x: list(zip(x['pat_key'], x['index_date'])))
    .to_dict()
)

# Use multiprocessing to process each year
print("Starting parallel read of enrollment files...")
with concurrent.futures.ProcessPoolExecutor() as executor:
    results = list(executor.map(read_enroll, year_to_pat_date.items()))

# Combine all enrollment data
final_enroll_df = pd.concat(results, ignore_index=True)

# Get and save value counts
final_enroll_df.to_csv("/home/stofer@chapman.edu/federated_analysis/tableau/payment_type.csv", header=True, index = False)
