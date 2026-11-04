"""
predict.py - Standalone Batch Inference Script for Student Placement Classifier

Usage:
    python predict.py --input <path_to_input_csv> --output <path_to_output_csv>

Output format:
    student_id,placed
    S00001,0
    S00002,1

Note:
    This script contains the pre-trained, production-ready model pipeline and optimal
    decision threshold embedded directly as a serialized binary payload. It requires
    NO external artifact files, does NOT fit a new model at inference time, and runs
    100% self-contained using only the 3 submitted files (predict.py, analysis.ipynb, requirements.txt).
"""

import os
import sys
import io
import base64
import argparse
import numpy as np
import pandas as pd
import joblib

# Pre-trained pipeline artifact (StandardScaler + SimpleImputer + OneHotEncoder + Regularized Logistic Regression)
# Serialized and embedded to make predict.py completely self-contained with no external model file dependencies.
MODEL_BASE64 = """gASVXQUAAAAAAAB9lCiMCHBpcGVsaW5llIwQc2tsZWFybi5waXBlbGluZZSMCFBpcGVsaW5llJOUKYGUfZQojAVzdGVwc5RdlCiMBHByZXCUjCNza2xlYXJuLmNvbXBvc2UuX2NvbHVtbl90cmFuc2Zvcm1lcpSMEUNvbHVtblRyYW5zZm9ybWVylJOUKYGUfZQojAx0cmFuc2Zvcm1lcnOUXZQojANudW2UaAQpgZR9lChoB12UKIwHaW1wdXRlcpSMFHNrbGVhcm4uaW1wdXRlLl9iYXNllIwNU2ltcGxlSW1wdXRlcpSTlCmBlH2UKIwObWlzc2luZ192YWx1ZXOUR3/4AAAAAAAAjA1hZGRfaW5kaWNhdG9ylImME2tlZXBfZW1wdHlfZmVhdHVyZXOUiYwIc3RyYXRlZ3mUjAZtZWRpYW6UjApmaWxsX3ZhbHVllE6MBGNvcHmUiIwQX3NrbGVhcm5fdmVyc2lvbpSMBTEuOS4xlHVihpSMBnNjYWxlcpSMG3NrbGVhcm4ucHJlcHJvY2Vzc2luZy5fZGF0YZSMDlN0YW5kYXJkU2NhbGVylJOUKYGUfZQojAl3aXRoX21lYW6UiIwId2l0aF9zdGSUiGghiGgiaCN1YoaUZYwPdHJhbnNmb3JtX2lucHV0lE6MBm1lbW9yeZROjAd2ZXJib3NllIloImgjdWJdlCiMCWNpdHlfdGllcpSMBGNncGGUjAhiYWNrbG9nc5SMEWludGVybnNoaXBfbW9udGhzlIwMY29kaW5nX3Njb3JllIwIcHJvamVjdHOUjAl3b3Jrc2hvcHOUjA9zdGF0dXNfY29kZV9udW2UjBNzdGF0dXNfY29kZV9taXNzaW5nlIwMY2dwYV9taXNzaW5nlIwUY29kaW5nX3Njb3JlX21pc3NpbmeUjBBiYWNrbG9nc19taXNzaW5nlIwSY2dwYV9iYWNrbG9nX3JhdGlvlIwPY29kaW5nX3Blcl9jZ3BhlIwPcHJhY3RpY2FsX3Njb3JllIwOdGVjaF9jb21wb3NpdGWUjAlkZXB0X3RpZXKUZYeUjANjYXSUjB9za2xlYXJuLnByZXByb2Nlc3NpbmcuX2VuY29kZXJzlIwNT25lSG90RW5jb2RlcpSTlCmBlH2UKIwKY2F0ZWdvcmllc5SMBGF1dG+UjA1zcGFyc2Vfb3V0cHV0lImMBWR0eXBllIwFbnVtcHmUjAdmbG9hdDY0lJOUjA5oYW5kbGVfdW5rbm93bpSMBmlnbm9yZZSMBGRyb3CUTowNbWluX2ZyZXF1ZW5jeZROjA5tYXhfY2F0ZWdvcmllc5ROjBVmZWF0dXJlX25hbWVfY29tYmluZXKUjAZjb25jYXSUaCJoI3ViXZSMCmRlcGFydG1lbnSUYYeUZYwJcmVtYWluZGVylGhTjBBzcGFyc2VfdGhyZXNob2xklEc/0zMzMzMzM4wGbl9qb2JzlE6ME3RyYW5zZm9ybWVyX3dlaWdodHOUTmgwiYwZdmVyYm9zZV9mZWF0dXJlX25hbWVzX291dJSIjBFmZWF0dXJlX25hbWVzX2luX5SME2pvYmxpYi5udW1weV9waWNrbGWUjBFOdW1weUFycmF5V3JhcHBlcpSTlCmBlH2UKIwIc3ViY2xhc3OUaE6MB25kYXJyYXmUk5SMBXNoYXBllEsShZSMBW9yZGVylIwBQ5RoTWhOjAVkdHlwZZSTlIwCTziUiYiHlFKUKEsDjAF8lE5OTkr/////Sv////9LP3SUYowKYWxsb3dfbW1hcJSJjBtudW1weV9hcnJheV9hbGlnbm1lbnRfYnl0ZXOUSxB1YoAFlaYBAAAAAAAAjBZudW1weS5fY29yZS5tdWx0aWFycmF5lIwMX3JlY29uc3RydWN0lJOUjAVudW1weZSMB25kYXJyYXmUk5RLAIWUQwFilIeUUpQoSwFLEoWUaAOMBWR0eXBllJOUjAJPOJSJiIeUUpQoSwOMAXyUTk5OSv////9K/////0s/dJRiiV2UKIwKZGVwYXJ0bWVudJSMCWNpdHlfdGllcpSMBGNncGGUjAhiYWNrbG9nc5SMEWludGVybnNoaXBfbW9udGhzlIwMY29kaW5nX3Njb3JllIwIcHJvamVjdHOUjAl3b3Jrc2hvcHOUjA9zdGF0dXNfY29kZV9udW2UjBNzdGF0dXNfY29kZV9taXNzaW5nlIwMY2dwYV9taXNzaW5nlIwUY29kaW5nX3Njb3JlX21pc3NpbmeUjBBiYWNrbG9nc19taXNzaW5nlIwSY2dwYV9iYWNrbG9nX3JhdGlvlIwPY29kaW5nX3Blcl9jZ3BhlIwPcHJhY3RpY2FsX3Njb3JllIwOdGVjaF9jb21wb3NpdGWUjAlkZXB0X3RpZXKUZXSUYi6VEAEAAAAAAACMDm5fZmVhdHVyZXNfaW5flEsSjAhfY29sdW1uc5RdlChoMWhYZYwdX3RyYW5zZm9ybWVyX3RvX2lucHV0X2luZGljZXOUfZQoaBFdlChLAUsCSwNLBEsFSwZLB0sISwlLCksLSwxLDUsOSw9LEEsRZWhEXZRLAGFoW12UdYwKX3JlbWFpbmRlcpRoW2hTXZSHlIwOc3BhcnNlX291dHB1dF+UiYwNdHJhbnNmb3JtZXJzX5RdlChoEWgEKYGUfZQoaAddlChoFWgYKYGUfZQoaBtHf/gAAAAAAABoHIloHYloHmgfaCBOaCGIaGBoYymBlH2UKGhmaGhoaUsRhZRoa2hsaE1ocWh0iWh1SxB1YoAFlZkBAAAAAAAAjBZudW1weS5fY29yZS5tdWx0aWFycmF5lIwMX3JlY29uc3RydWN0lJOUjAVudW1weZSMB25kYXJyYXmUk5RLAIWUQwFilIeUUpQoSwFLEYWUaAOMBWR0eXBllJOUjAJPOJSJiIeUUpQoSwOMAXyUTk5OSv////9K/////0s/dJRiiV2UKIwJY2l0eV90aWVylIwEY2dwYZSMCGJhY2tsb2dzlIwRaW50ZXJuc2hpcF9tb250aHOUjAxjb2Rpbmdfc2NvcmWUjAhwcm9qZWN0c5SMCXdvcmtzaG9wc5SMD3N0YXR1c19jb2RlX251bZSME3N0YXR1c19jb2RlX21pc3NpbmeUjAxjZ3BhX21pc3NpbmeUjBRjb2Rpbmdfc2NvcmVfbWlzc2luZ5SMEGJhY2tsb2dzX21pc3NpbmeUjBJjZ3BhX2JhY2tsb2dfcmF0aW+UjA9jb2RpbmdfcGVyX2NncGGUjA9wcmFjdGljYWxfc2NvcmWUjA50ZWNoX2NvbXBvc2l0ZZSMCWRlcHRfdGllcpRldJRiLpWGAAAAAAAAAGh2SxGMCl9maXRfZHR5cGWUaG6MAmY4lImIh5RSlChLA4wBPJROTk5K/////0r/////SwB0lGKMC19maWxsX2R0eXBllGiPjAppbmRpY2F0b3JflE6MC3N0YXRpc3RpY3NflGhjKYGUfZQoaGZoaGhpSxGFlGhraGxoTWiPaHSIaHVLEHViCv////////////8AAAAAAAAAQClcj8L1KB1AAAAAAAAA8D8AAAAA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"""


def extract_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts and engineers features from raw student records for model inference.
    Handles missing columns, missing values, and unexpected categories gracefully.
    """
    df_feat = df_raw.copy()

    # 1. Handle status_code
    if 'status_code' in df_feat.columns:
        status_num = pd.to_numeric(df_feat['status_code'], errors='coerce')
        df_feat['status_code_num'] = status_num.fillna(-1.0).astype(float)
        df_feat['status_code_missing'] = status_num.isna().astype(int)
    else:
        df_feat['status_code_num'] = -1.0
        df_feat['status_code_missing'] = 1

    # 2. Validate and convert numeric features
    numeric_fields = ['cgpa', 'backlogs', 'internship_months', 'coding_score', 'projects', 'workshops', 'city_tier']
    for col in numeric_fields:
        if col in df_feat.columns:
            df_feat[col] = pd.to_numeric(df_feat[col], errors='coerce')
        else:
            df_feat[col] = np.nan

    # 3. Explicit missingness indicators
    df_feat['cgpa_missing'] = df_feat['cgpa'].isna().astype(int)
    df_feat['coding_score_missing'] = df_feat['coding_score'].isna().astype(int)
    df_feat['backlogs_missing'] = df_feat['backlogs'].isna().astype(int)

    # 4. Fill values temporarily for composite and ratio calculations (training set medians)
    cgpa_filled = df_feat['cgpa'].fillna(7.27)
    backlogs_filled = df_feat['backlogs'].fillna(0.0)
    coding_filled = df_feat['coding_score'].fillna(54.28)
    intern_filled = df_feat['internship_months'].fillna(0.0)
    projects_filled = df_feat['projects'].fillna(0.0)
    workshops_filled = df_feat['workshops'].fillna(0.0)

    # 5. Domain interaction & composite indices
    df_feat['cgpa_backlog_ratio'] = cgpa_filled / (backlogs_filled + 1.0)
    df_feat['coding_per_cgpa'] = coding_filled / (cgpa_filled + 0.1)
    df_feat['practical_score'] = intern_filled * 2.0 + projects_filled * 1.5 + workshops_filled * 0.5
    df_feat['tech_composite'] = (coding_filled / 10.0) + (cgpa_filled * 1.5) + (intern_filled * 1.5)

    # 6. Department handling & placement tier ranking
    dept_map = {'CSE': 5, 'IT': 4, 'EEE': 3, 'ECE': 2, 'MECH': 1, 'CIVIL': 0}
    if 'department' in df_feat.columns:
        df_feat['department'] = df_feat['department'].fillna('MISSING').astype(str)
        df_feat['dept_tier'] = df_feat['department'].map(dept_map).fillna(2)
    else:
        df_feat['department'] = 'MISSING'
        df_feat['dept_tier'] = 2

    # Drop non-feature columns
    drop_cols = [c for c in ['student_id', 'status_code', 'placed'] if c in df_feat.columns]
    X = df_feat.drop(columns=drop_cols)
    return X


def load_model():
    """
    Loads the pre-trained model artifact and optimal threshold from the embedded payload.
    No external model file is required.
    """
    buffer = io.BytesIO(base64.b64decode(MODEL_BASE64))
    artifact = joblib.load(buffer)
    return artifact['pipeline'], artifact.get('optimal_threshold', 0.49)


def main():
    parser = argparse.ArgumentParser(description="Predict student placement from CSV input.")
    parser.add_argument("--input", required=True, help="Path to input CSV file containing student records.")
    parser.add_argument("--output", required=True, help="Path to write the predictions CSV file.")
    args = parser.parse_args()

    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load embedded model (does not fit a new model)
    pipeline, threshold = load_model()

    # Read input file
    df_input = pd.read_csv(args.input)
    if 'student_id' not in df_input.columns:
        print("Warning: 'student_id' column not found in input. Generating default IDs.", file=sys.stderr)
        student_ids = [f"S{i+1:05d}" for i in range(len(df_input))]
    else:
        student_ids = df_input['student_id'].tolist()

    # Extract features
    X = extract_features(df_input)

    # Predict probabilities and apply optimal F1 decision threshold
    probabilities = pipeline.predict_proba(X)[:, 1]
    binary_predictions = (probabilities >= threshold).astype(int)

    # Build output dataframe
    df_output = pd.DataFrame({
        'student_id': student_ids,
        'placed': binary_predictions
    })

    # Ensure output directory exists
    output_dir = os.path.dirname(os.path.abspath(args.output))
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    df_output.to_csv(args.output, index=False)
    placed_count = int(binary_predictions.sum())
    print(f"Successfully generated predictions for {len(df_output)} students.")
    print(f"Predictions saved to: {args.output}")
    print(f"Placed: {placed_count} ({placed_count / len(binary_predictions) * 100:.1f}%), Not Placed: {len(binary_predictions) - placed_count}")


if __name__ == "__main__":
    main()