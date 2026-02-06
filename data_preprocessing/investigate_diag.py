import pandas as pd

# Load the dataset
file_path = 'data/iqvia_pat_2015.csv'  # Adjust the file path if needed
df = pd.read_csv(file_path)

# Convert the 'to_dt' column to datetime format
df['to_dt'] = pd.to_datetime(df['to_dt'], errors='coerce')

# Extract the month and year (you can adjust the format to include just the month or year as needed)
df['month_year'] = df['to_dt'].dt.to_period('M')

# Get the value counts of rows by month
monthly_counts = df['month_year'].value_counts().sort_index()

# Calculate the proportion of rows by month
monthly_proportions = monthly_counts / len(df)

# Print the results
print(monthly_proportions)
