#!/usr/bin/env python3
"""
Check data preprocessing and identify any issues
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, 'src')

from data_utils import DataPreprocessor

print("="*60)
print("Data Preprocessing Check")
print("="*60)

# Load raw data
print("\n1. Loading raw data...")
try:
    train_df = pd.read_csv('data/train.csv')
    test_df = pd.read_csv('data/test.csv')
    print(f"✓ Train: {len(train_df)} rows")
    print(f"✓ Test: {len(test_df)} rows")
except Exception as e:
    print(f"✗ Error loading data: {e}")
    sys.exit(1)

# Check raw data
print("\n2. Raw training data sample:")
print(train_df.head(2))
print("\nColumns:", train_df.columns.tolist())
print("\nData types:")
print(train_df.dtypes)

# Check for missing values
print("\n3. Missing values in training data:")
print(train_df.isnull().sum())

# Check catalog_content sample
print("\n4. Sample catalog_content:")
print(train_df['catalog_content'].iloc[0][:500])

# Test preprocessing
print("\n5. Testing preprocessing with feature extraction...")
try:
    preprocessor = DataPreprocessor(extract_features=True)
    
    # Test on small sample first
    sample_df = train_df.head(10).copy()
    print(f"\nProcessing {len(sample_df)} samples...")
    
    processed_df = preprocessor.preprocess_training_data(sample_df, fit_scaler=True)
    
    print(f"✓ Processed successfully: {len(processed_df)} rows")
    print("\nProcessed columns:", processed_df.columns.tolist())
    
    # Check extracted features
    print("\n6. Extracted features:")
    if 'value' in processed_df.columns:
        print(f"✓ value: dtype={processed_df['value'].dtype}, sample={processed_df['value'].iloc[0]}")
    if 'pack_count' in processed_df.columns:
        print(f"✓ pack_count: dtype={processed_df['pack_count'].dtype}, sample={processed_df['pack_count'].iloc[0]}")
    if 'unit' in processed_df.columns:
        print(f"✓ unit: sample={processed_df['unit'].iloc[0]}")
    
    # Check unit columns
    unit_cols = [col for col in processed_df.columns if col.startswith('unit_')]
    print(f"✓ Unit one-hot columns: {len(unit_cols)}")
    if unit_cols:
        print(f"  Sample: {unit_cols[:3]}")
        print(f"  Dtypes: {processed_df[unit_cols].dtypes.unique()}")
    
    # Check clean_text
    if 'clean_text' in processed_df.columns:
        print(f"✓ clean_text: sample length={len(processed_df['clean_text'].iloc[0])}")
        print(f"  Sample: {processed_df['clean_text'].iloc[0][:100]}...")
    
    # Check numerical feature array
    print("\n7. Testing numerical feature extraction for model:")
    numerical_cols = ['value', 'pack_count'] + unit_cols
    print(f"Total numerical features: {len(numerical_cols)}")
    
    # Extract as numpy array (like the model will do)
    num_features = processed_df[numerical_cols].values
    print(f"✓ Numpy array shape: {num_features.shape}")
    print(f"✓ Numpy array dtype: {num_features.dtype}")
    print(f"✓ Sample row: {num_features[0][:5]}...")
    
    # Check for any non-numeric values
    if num_features.dtype == np.object_:
        print("✗ ERROR: Array has object dtype!")
        print("  Checking individual columns...")
        for col in numerical_cols:
            col_dtype = processed_df[col].dtype
            if col_dtype == np.object_:
                print(f"  ✗ {col}: {col_dtype}")
                print(f"    Sample values: {processed_df[col].head()}")
            else:
                print(f"  ✓ {col}: {col_dtype}")
    else:
        print("✓ All numerical features are numeric!")
    
    # Test conversion to float32
    print("\n8. Testing float32 conversion:")
    try:
        num_features_f32 = num_features.astype(np.float32)
        print(f"✓ Converted to float32: {num_features_f32.dtype}")
        print(f"✓ Sample: {num_features_f32[0][:5]}")
    except Exception as e:
        print(f"✗ Error converting to float32: {e}")
        print("  Checking for problematic values...")
        for i, col in enumerate(numerical_cols):
            try:
                processed_df[col].astype(np.float32)
                print(f"  ✓ {col}: OK")
            except Exception as col_e:
                print(f"  ✗ {col}: {col_e}")
                print(f"    Values: {processed_df[col].unique()[:10]}")
    
    # Test on full training data
    print("\n9. Testing on full training data...")
    try:
        full_processed = preprocessor.preprocess_training_data(train_df, fit_scaler=True)
        print(f"✓ Full preprocessing successful: {len(full_processed)} rows")
        
        # Get unit columns
        unit_columns = [col for col in full_processed.columns if col.startswith('unit_')]
        
        # Test on test data
        print("\n10. Testing on test data...")
        test_processed = preprocessor.preprocess_test_data(test_df, unit_columns=unit_columns)
        print(f"✓ Test preprocessing successful: {len(test_processed)} rows")
        
        # Final check
        print("\n11. Final numerical feature check:")
        numerical_cols = ['value', 'pack_count'] + unit_columns
        
        train_features = full_processed[numerical_cols].values
        test_features = test_processed[numerical_cols].values
        
        print(f"✓ Train features: shape={train_features.shape}, dtype={train_features.dtype}")
        print(f"✓ Test features: shape={test_features.shape}, dtype={test_features.dtype}")
        
        # Check for NaN or inf
        train_has_nan = np.isnan(train_features).any()
        train_has_inf = np.isinf(train_features).any()
        test_has_nan = np.isnan(test_features).any()
        test_has_inf = np.isinf(test_features).any()
        
        if train_has_nan or train_has_inf:
            print(f"✗ Train features have NaN: {train_has_nan}, inf: {train_has_inf}")
        else:
            print("✓ Train features: No NaN or inf")
        
        if test_has_nan or test_has_inf:
            print(f"✗ Test features have NaN: {test_has_nan}, inf: {test_has_inf}")
        else:
            print("✓ Test features: No NaN or inf")
        
        print("\n" + "="*60)
        print("✅ ALL CHECKS PASSED!")
        print("="*60)
        print(f"\nReady to train with {len(numerical_cols)} numerical features")
        
    except Exception as e:
        print(f"✗ Error in full preprocessing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
except Exception as e:
    print(f"✗ Error in preprocessing: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
