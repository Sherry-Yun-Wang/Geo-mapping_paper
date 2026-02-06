import pandas as pd
import os

def read_ndc_codes():
    file_path = '/home/stofer@chapman.edu/merck_proposal/ndc_codes.txt'

    with open(file_path, 'r') as file:
        codes_string = file.read()

        # Split the string into individual code numbers
        codes_list = codes_string.split(",")

        # Join the formatted codes into a single string with single quotes
        formatted_string = ", ".join("'{}'".format(code) for code in codes_list)
        ndc_codes = [code.strip("' ") for code in formatted_string.split(",")]
    return ndc_codes


def read_iqvia_header():
    header_folder = '/sharefolder/IQVIA/header'
    header_files = [file_name for file_name in os.listdir(header_folder) if file_name.startswith('header_claims_')]

    header_data = {}
    for file_name in header_files:
        year = file_name.split('_')[-1].split('.')[0]
        with open(os.path.join(header_folder, file_name), 'r') as file:
            header = file.readline().strip().split('|')
            header_data[year] = header
    return header_data

def read_iqvia_claims(year, header_data, ndc_codes):
    claims_folder = '/sharefolder/IQVIA'  
    year_folder = os.path.join(claims_folder, 'claims_{}'.format(year))
    csv_in_parts_folder = os.path.join(year_folder, 'csv_in_parts')
    csv_files = [file for file in os.listdir(csv_in_parts_folder) if file.endswith('.csv')]

    filtered_data_list =[]
    i = 0
    for csv_file in csv_files:
        i += 1
        file_path = os.path.join(csv_in_parts_folder, csv_file)
        data_part = pd.read_csv(file_path, sep='|', header=None, dtype=str)
        data_part.columns = header_data[year]

        # Filter observations where ndc code is in ndc_codes list
        filtered_data = data_part[data_part['ndc'].isin(ndc_codes)]
        filtered_data_list.append(filtered_data)
        print(f"Appended part {i} out of 200!", flush = True)
    combined_df = pd.concat(filtered_data_list, ignore_index = True)
    return combined_df


def read_header():
    header_folder = '/sharefolder/IQVIA/header'
    header_files = [file_name for file_name in os.listdir(header_folder) if file_name.startswith('header_claims_')]

    header_data = {}
    for file_name in header_files:
        year = file_name.split('_')[-1].split('.')[0]
        with open(os.path.join(header_folder, file_name), 'r') as file:
            header = file.readline().strip().split('|')
            header_data[year] = header
    return header_data