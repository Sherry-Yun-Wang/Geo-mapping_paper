import pandas as pd

def compute_weighted_zip(input_csv, output_csv):
    # Load ZIP data
    df = pd.read_csv(input_csv)
    
    # Drop rows with missing ZIPs
    df = df.dropna(subset=['zip'])

    # Ensure ZIPs are strings, and store original for fallback logic
    df['zip'] = df['zip'].astype(str)
    df['original_zip'] = df['zip']  # Keep original ZIP for edge-case fallback

    # Pad ZIPs to 5 digits where needed
    df['zip'] = df['zip'].str.zfill(5)

    # Ensure population is numeric, fill NaNs with 0
    df['population'] = pd.to_numeric(df['population'], errors='coerce').fillna(0)

    # Convert ZIPs to int for weighted average calculation
    df['zip_int'] = df['zip'].astype(int)

    # Extract 3-digit ZIP prefix
    df['zip3'] = df['zip'].str[:3]

    # Build a set of all real ZIPs (for existence check)
    valid_zips = set(df['zip'])

    # Define weighted ZIP function with fallback logic
    def weighted_zip(group):
        total_pop = group['population'].sum()
        if total_pop == 0:
            example_zip = group['original_zip'].iloc[0]
            if len(example_zip) < 5:
                fallback_zip = example_zip.zfill(5)  # e.g., '5' → '00005'
            else:
                fallback_zip = example_zip[:3] + '00'  # e.g., '19245' → '19200'
            return pd.Series({'weighted_zip': fallback_zip})

        # Calculate weighted average
        weighted_avg = (group['zip_int'] * group['population']).sum() / total_pop
        candidate_zip = str(round(weighted_avg)).zfill(5)

        # Check if candidate exists
        if candidate_zip in valid_zips:
            return pd.Series({'weighted_zip': candidate_zip})
        else:
            # Find closest valid ZIP numerically
            candidate_int = int(candidate_zip)
            closest_zip = min(valid_zips, key=lambda z: abs(int(z) - candidate_int))
            return pd.Series({'weighted_zip': closest_zip})

    # Apply to each 3-digit group
    result = df.groupby('zip3').apply(weighted_zip).reset_index()

    # Save to output file
    result.to_csv(output_csv, index=False)
    print(f"Weighted ZIPs saved to: {output_csv}")

# Example usage
if __name__ == "__main__":
    compute_weighted_zip(input_csv="uszips.csv", output_csv="weighted_zip_by_zip3.csv")
