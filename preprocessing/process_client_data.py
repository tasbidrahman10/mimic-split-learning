import os
import pandas as pd
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder

def load_table(data_dir, filename):
    """
    Helper function to load compressed CSV safely.
    """
    path = os.path.join(data_dir, filename)
    print(f"Loading {filename}...")
    if os.path.exists(path):
        return pd.read_csv(path, compression='gzip')
    else:
        print(f"Warning: {filename} not found at {path}.")
        return None

def build_client_dataset(input_folder, output_file):
    """
    Loads raw MIMIC-IV tables from input_folder, performs preprocessing, feature engineering,
    and saves the resulting patient-level feature vectors to output_file.
    """
    print(f"Processing data from: {input_folder}")
    
    # 1. Load core tables
    # Note: Filenames might vary slightly depending on how the data was split (e.g., labevents shards)
    # Using generic handling or specific filenames as seen in the notebook.
    patients_df = load_table(input_folder, 'patients.csv.gz')
    admissions_df = load_table(input_folder, 'admissions.csv.gz')
    icustays_df = load_table(input_folder, 'icustays.csv.gz')
    
    # Logic to find labevents file since it might have a shard suffix like 'labevents.csv-003.gz'
    labevents_filename = None
    for f in os.listdir(input_folder):
        if f.startswith('labevents') and f.endswith('.gz'):
            labevents_filename = f
            break
            
    # labevents_df = load_table(input_folder, labevents_filename) if labevents_filename else None # removed to save memory

    if patients_df is None or admissions_df is None:
        raise ValueError("Critical tables (patients, admissions) missing.")

    # 2. Merge Admissions and Patients
    print("Merging Admissions and Patients...")
    full_df = pd.merge(admissions_df, patients_df, on='subject_id', how='inner')

    # 3. Calculate Age
    # Age = anchor_age + (admittime_year - anchor_year)
    full_df['admittime'] = pd.to_datetime(full_df['admittime'])
    full_df['admission_year'] = full_df['admittime'].dt.year
    full_df['age'] = full_df['anchor_age'] + (full_df['admission_year'] - full_df['anchor_year'])

    # 4. Merge with ICU Stays (Length of Stay)
    if icustays_df is not None:
        print("Merging ICU Stays...")
        # Dictionary of aggregations
        agg_dict = {'los': 'sum'}
        if 'first_careunit' in icustays_df.columns:
            agg_dict['first_careunit'] = 'first'
            
        icu_features = icustays_df.groupby('hadm_id').agg(agg_dict).reset_index()
        full_df = pd.merge(full_df, icu_features, on='hadm_id', how='left')
        full_df['los'] = full_df['los'].fillna(0)
    else:
        full_df['los'] = 0

    # 5. Lab Features (Optimized for Memory)
    if labevents_filename:
        print(f"Processing Lab Events from {labevents_filename} in chunks...")
        lab_path = os.path.join(input_folder, labevents_filename)
        
        # We need to process in chunks because labevents can be huge
        chunk_size = 100000
        
        # 1. First pass: Find top lab items (or just use a sample)
        # To save time/memory, let's just use the first few chunks to estimate common labs
        # or list standard critical care labs if we knew them. 
        # Let's count from the first 500k rows
        item_counts = pd.Series(dtype=int)
        
        try:
            # Use usecols to save memory
            for i, chunk in enumerate(pd.read_csv(lab_path, compression='gzip', chunksize=chunk_size, usecols=['itemid'])):
                if i > 5: break # only look at first ~500k rows for speed
                item_counts = item_counts.add(chunk['itemid'].value_counts(), fill_value=0)
            
            top_labs = item_counts.sort_values(ascending=False).head(10).index.tolist()
            print(f"Top 10 Lab Items (estimated from sample): {top_labs}")
            
            # 2. Second pass: Load only the relevant items
            relevant_chunks = []
            for chunk in pd.read_csv(lab_path, compression='gzip', chunksize=chunk_size, usecols=['hadm_id', 'itemid', 'valuenum']):
                # Filter
                filtered = chunk[chunk['itemid'].isin(top_labs)]
                if not filtered.empty:
                     relevant_chunks.append(filtered)
            
            if relevant_chunks:
                lab_subset = pd.concat(relevant_chunks)
                
                # Aggregate: Mean value per admission per item
                lab_features = lab_subset.groupby(['hadm_id', 'itemid'])['valuenum'].mean().unstack(fill_value=0)
                
                # Rename columns
                lab_features.columns = [f'lab_{c}' for c in lab_features.columns]
                
                full_df = pd.merge(full_df, lab_features, on='hadm_id', how='left')
            else:
                print("No matching lab events found in chunks.")
        
        except Exception as e:
            print(f"Error processing labevents: {e}. Skipping lab features.")


    # 6. Handle Missing Values & Encoding
    print("Finalizing features...")
    
    # Fill remaining numeric NaNs with 0 (e.g. age, los if missed)
    numeric_cols = full_df.select_dtypes(include=[np.number]).columns
    full_df[numeric_cols] = full_df[numeric_cols].fillna(0)

    # Encode Gender
    if 'gender' in full_df.columns:
        # Fill missing gender with 'Unknown'
        full_df['gender'] = full_df['gender'].fillna('Unknown')
        le_gender = LabelEncoder()
        full_df['gender_encoded'] = le_gender.fit_transform(full_df['gender'].astype(str))
    else:
        full_df['gender_encoded'] = 0
    
    # 7. Select Columns for Final Dataset
    # We keep identifiers (subject_id, hadm_id) just in case, plus features and target
    feature_cols = ['age', 'gender_encoded', 'los'] + [c for c in full_df.columns if c.startswith('lab_')]
    
    # Target: hospital_expire_flag
    if 'hospital_expire_flag' in full_df.columns:
        target_col = 'hospital_expire_flag'
    else:
        print("Target 'hospital_expire_flag' not found. Creating synthetic target for demo.")
        full_df['hospital_expire_flag'] = np.random.randint(0, 2, size=len(full_df))
        target_col = 'hospital_expire_flag'

    cols_to_save = ['subject_id', 'hadm_id'] + feature_cols + [target_col]
    final_df = full_df[cols_to_save]

    print(f"saving processed data to {output_file}...")
    print(f"Final shape: {final_df.shape}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if output_file.endswith('.csv'):
        final_df.to_csv(output_file, index=False)
    elif output_file.endswith('.parquet'):
        final_df.to_parquet(output_file, index=False)
    else:
        # Default to csv
        final_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Client Data for Federated Learning")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to folder containing raw CSV.GZ files")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the processed CSV or Parquet file")
    
    args = parser.parse_args()
    
    build_client_dataset(args.input_folder, args.output_file)
