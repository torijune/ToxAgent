"""
Extract and analyze only_fg differences from pairs_fg_stereo_merged_nodot.csv

This script creates CSV files showing:
- toxic_smiles, nontoxic_smiles
- Full feature columns for each molecule
- Difference columns
- Unique feature columns
Separated by one_step and multi_step
"""
import pandas as pd
import json
import ast
from pathlib import Path
from typing import Dict, Any, List


def parse_json_column(value: Any) -> Any:
    """Parse JSON string column to dict/list."""
    if pd.isna(value) or not value:
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except:
        try:
            return ast.literal_eval(str(value))
        except:
            return str(value)


def extract_fg_features(row: pd.Series) -> Dict[str, Any]:
    """Extract functional group features from a row."""
    features = {}
    
    # Full feature columns
    features['toxic_fg_names'] = parse_json_column(row.get('toxic_fg_names'))
    features['toxic_fg_counts'] = parse_json_column(row.get('toxic_fg_counts'))
    features['toxic_fg_full'] = parse_json_column(row.get('toxic_fg_full'))
    features['nontoxic_fg_names'] = parse_json_column(row.get('nontoxic_fg_names'))
    features['nontoxic_fg_counts'] = parse_json_column(row.get('nontoxic_fg_counts'))
    features['nontoxic_fg_full'] = parse_json_column(row.get('nontoxic_fg_full'))
    
    # Difference columns
    features['has_fg_diff'] = row.get('has_fg_diff', False)
    features['n_fg_diff'] = row.get('n_fg_diff', 0)
    features['unique_fg'] = parse_json_column(row.get('unique_fg'))
    
    return features


def create_fg_diff_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with FG difference information."""
    rows = []
    
    for idx, row in df.iterrows():
        fg_features = extract_fg_features(row)
        
        # Basic info
        result_row = {
            'row_index': idx,
            'toxic_smiles': row.get('toxic_smiles', ''),
            'nontoxic_smiles': row.get('nontoxic_smiles', ''),
            'toxic_canonical_smiles': row.get('toxic_canonical_smiles', ''),
            'nontoxic_canonical_smiles': row.get('nontoxic_canonical_smiles', ''),
            'dataset_name': row.get('dataset_name', ''),
            'endpoint': row.get('endpoint', ''),
        }
        
        # Full feature columns
        result_row['toxic_fg_names'] = json.dumps(fg_features['toxic_fg_names']) if fg_features['toxic_fg_names'] else '[]'
        result_row['toxic_fg_counts'] = json.dumps(fg_features['toxic_fg_counts']) if fg_features['toxic_fg_counts'] else '{}'
        result_row['toxic_fg_full'] = json.dumps(fg_features['toxic_fg_full']) if fg_features['toxic_fg_full'] else '{}'
        result_row['nontoxic_fg_names'] = json.dumps(fg_features['nontoxic_fg_names']) if fg_features['nontoxic_fg_names'] else '[]'
        result_row['nontoxic_fg_counts'] = json.dumps(fg_features['nontoxic_fg_counts']) if fg_features['nontoxic_fg_counts'] else '{}'
        result_row['nontoxic_fg_full'] = json.dumps(fg_features['nontoxic_fg_full']) if fg_features['nontoxic_fg_full'] else '{}'
        
        # Difference columns
        result_row['has_fg_diff'] = fg_features['has_fg_diff']
        result_row['n_fg_diff'] = fg_features['n_fg_diff']
        result_row['unique_fg'] = json.dumps(fg_features['unique_fg']) if fg_features['unique_fg'] else '[]'
        
        # Extract unique FG names for easier reading
        unique_fg_list = fg_features['unique_fg']
        if isinstance(unique_fg_list, list):
            toxic_unique_fg_names = [fg.get('fg_name') for fg in unique_fg_list 
                                   if isinstance(fg, dict) and fg.get('reason') == 'name_only_in_toxic']
            nontoxic_unique_fg_names = [fg.get('fg_name') for fg in unique_fg_list 
                                      if isinstance(fg, dict) and fg.get('reason') == 'name_only_in_nontoxic']
            atom_diff_fg_names = [fg.get('fg_name') for fg in unique_fg_list 
                                if isinstance(fg, dict) and fg.get('reason') == 'atom_index_diff']
            
            result_row['toxic_unique_fg_names'] = json.dumps(toxic_unique_fg_names) if toxic_unique_fg_names else '[]'
            result_row['nontoxic_unique_fg_names'] = json.dumps(nontoxic_unique_fg_names) if nontoxic_unique_fg_names else '[]'
            result_row['atom_diff_fg_names'] = json.dumps(atom_diff_fg_names) if atom_diff_fg_names else '[]'
        else:
            result_row['toxic_unique_fg_names'] = '[]'
            result_row['nontoxic_unique_fg_names'] = '[]'
            result_row['atom_diff_fg_names'] = '[]'
        
        rows.append(result_row)
    
    return pd.DataFrame(rows)


def main():
    """Main function to process only_fg differences."""
    input_csv = Path(__file__).parent.parent / 'pairs_fg_stereo_merged_nodot.csv'
    output_dir = Path(__file__).parent
    
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Total rows: {len(df)}")
    
    # Filter to only_fg
    df_fg = df[df['diff_type'] == 'only_fg'].copy()
    print(f"only_fg rows: {len(df_fg)}")
    
    # Process one_step and multi_step separately
    for step_type in ['one_step', 'multi_step']:
        df_step = df_fg[df_fg['step_num'] == step_type].copy()
        print(f"\n{step_type} rows: {len(df_step)}")
        
        if len(df_step) == 0:
            print(f"  No {step_type} rows found. Skipping.")
            continue
        
        # Create difference DataFrame
        diff_df = create_fg_diff_dataframe(df_step)
        
        # Save to CSV
        output_file = output_dir / f'only_fg_{step_type}_diff.csv'
        diff_df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")
        print(f"  Columns: {len(diff_df.columns)}")
        print(f"  Rows: {len(diff_df)}")


if __name__ == "__main__":
    main()
