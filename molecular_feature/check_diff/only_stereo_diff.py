"""
Extract and analyze only_stereo differences from pairs_fg_stereo_merged_nodot.csv

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


def extract_stereo_features(row: pd.Series) -> Dict[str, Any]:
    """Extract stereochemistry features from a row."""
    features = {}
    
    # Full feature columns - toxic
    features['toxic_chiral_centers'] = parse_json_column(row.get('toxic_chiral_centers'))
    features['toxic_ez_bonds'] = parse_json_column(row.get('toxic_ez_bonds'))
    features['toxic_has_chirality'] = row.get('toxic_has_chirality', False)
    features['toxic_has_ez_bonds'] = row.get('toxic_has_ez_bonds', False)
    features['toxic_stereochemistry'] = parse_json_column(row.get('toxic_stereochemistry'))
    
    # Full feature columns - nontoxic
    features['nontoxic_chiral_centers'] = parse_json_column(row.get('nontoxic_chiral_centers'))
    features['nontoxic_ez_bonds'] = parse_json_column(row.get('nontoxic_ez_bonds'))
    features['nontoxic_has_chirality'] = row.get('nontoxic_has_chirality', False)
    features['nontoxic_has_ez_bonds'] = row.get('nontoxic_has_ez_bonds', False)
    features['nontoxic_stereochemistry'] = parse_json_column(row.get('nontoxic_stereochemistry'))
    
    # Difference columns
    features['chiral_diff_loose'] = row.get('chiral_diff_loose', False)
    features['ez_diff_loose'] = row.get('ez_diff_loose', False)
    features['stereo_diff_type_loose'] = row.get('stereo_diff_type_loose', '')
    features['n_stereo_diff'] = row.get('n_stereo_diff', 0)
    features['stereochemistry_difference'] = parse_json_column(row.get('stereochemistry_difference'))
    
    return features


def extract_unique_stereo(features: Dict[str, Any]) -> Dict[str, List[str]]:
    """Extract unique stereochemistry features."""
    unique = {
        'toxic_unique_stereo': [],
        'nontoxic_unique_stereo': []
    }
    
    # Try to get from stereochemistry_difference first
    stereo_diff = features.get('stereochemistry_difference')
    if isinstance(stereo_diff, dict):
        unique['toxic_unique_stereo'] = stereo_diff.get('toxic_unique_stereo', [])
        unique['nontoxic_unique_stereo'] = stereo_diff.get('nontoxic_unique_stereo', [])
    
    # If not found, extract from individual columns
    if not unique['toxic_unique_stereo'] and not unique['nontoxic_unique_stereo']:
        tx_chiral = features.get('toxic_chiral_centers', [])
        nt_chiral = features.get('nontoxic_chiral_centers', [])
        
        if isinstance(tx_chiral, list) and isinstance(nt_chiral, list):
            tx_chiral_map = {c.get('atom_idx'): c.get('config') 
                           for c in tx_chiral if isinstance(c, dict) and 'atom_idx' in c}
            nt_chiral_map = {c.get('atom_idx'): c.get('config') 
                           for c in nt_chiral if isinstance(c, dict) and 'atom_idx' in c}
            
            # Find toxic-unique chiral centers
            for atom_idx, config in tx_chiral_map.items():
                if atom_idx not in nt_chiral_map or nt_chiral_map[atom_idx] != config:
                    unique['toxic_unique_stereo'].append(f"Chiral center {atom_idx}: {config} configuration")
            
            # Find nontoxic-unique chiral centers
            for atom_idx, config in nt_chiral_map.items():
                if atom_idx not in tx_chiral_map or tx_chiral_map[atom_idx] != config:
                    unique['nontoxic_unique_stereo'].append(f"Chiral center {atom_idx}: {config} configuration")
        
        # Check E/Z bonds
        tx_ez = features.get('toxic_ez_bonds', [])
        nt_ez = features.get('nontoxic_ez_bonds', [])
        
        if isinstance(tx_ez, list) and isinstance(nt_ez, list):
            tx_ez_map = {tuple(bond.get('bond', [])): bond.get('geometry', '') 
                        for bond in tx_ez if isinstance(bond, dict) and 'bond' in bond}
            nt_ez_map = {tuple(bond.get('bond', [])): bond.get('geometry', '') 
                        for bond in nt_ez if isinstance(bond, dict) and 'bond' in bond}
            
            for bond, geometry in tx_ez_map.items():
                if bond not in nt_ez_map or nt_ez_map[bond] != geometry:
                    unique['toxic_unique_stereo'].append(f"E/Z bond {list(bond)}: {geometry} geometry")
            
            for bond, geometry in nt_ez_map.items():
                if bond not in tx_ez_map or tx_ez_map[bond] != geometry:
                    unique['nontoxic_unique_stereo'].append(f"E/Z bond {list(bond)}: {geometry} geometry")
    
    return unique


def create_stereo_diff_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Create a DataFrame with stereochemistry difference information."""
    rows = []
    
    for idx, row in df.iterrows():
        stereo_features = extract_stereo_features(row)
        unique_stereo = extract_unique_stereo(stereo_features)
        
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
        
        # Full feature columns - toxic
        result_row['toxic_chiral_centers'] = json.dumps(stereo_features['toxic_chiral_centers']) if stereo_features['toxic_chiral_centers'] else '[]'
        result_row['toxic_ez_bonds'] = json.dumps(stereo_features['toxic_ez_bonds']) if stereo_features['toxic_ez_bonds'] else '[]'
        result_row['toxic_has_chirality'] = stereo_features['toxic_has_chirality']
        result_row['toxic_has_ez_bonds'] = stereo_features['toxic_has_ez_bonds']
        result_row['toxic_stereochemistry'] = json.dumps(stereo_features['toxic_stereochemistry']) if stereo_features['toxic_stereochemistry'] else '{}'
        
        # Full feature columns - nontoxic
        result_row['nontoxic_chiral_centers'] = json.dumps(stereo_features['nontoxic_chiral_centers']) if stereo_features['nontoxic_chiral_centers'] else '[]'
        result_row['nontoxic_ez_bonds'] = json.dumps(stereo_features['nontoxic_ez_bonds']) if stereo_features['nontoxic_ez_bonds'] else '[]'
        result_row['nontoxic_has_chirality'] = stereo_features['nontoxic_has_chirality']
        result_row['nontoxic_has_ez_bonds'] = stereo_features['nontoxic_has_ez_bonds']
        result_row['nontoxic_stereochemistry'] = json.dumps(stereo_features['nontoxic_stereochemistry']) if stereo_features['nontoxic_stereochemistry'] else '{}'
        
        # Difference columns
        result_row['chiral_diff_loose'] = stereo_features['chiral_diff_loose']
        result_row['ez_diff_loose'] = stereo_features['ez_diff_loose']
        result_row['stereo_diff_type_loose'] = stereo_features['stereo_diff_type_loose']
        result_row['n_stereo_diff'] = stereo_features['n_stereo_diff']
        result_row['stereochemistry_difference'] = json.dumps(stereo_features['stereochemistry_difference']) if stereo_features['stereochemistry_difference'] else '{}'
        
        # Unique feature columns
        result_row['toxic_unique_stereo'] = json.dumps(unique_stereo['toxic_unique_stereo']) if unique_stereo['toxic_unique_stereo'] else '[]'
        result_row['nontoxic_unique_stereo'] = json.dumps(unique_stereo['nontoxic_unique_stereo']) if unique_stereo['nontoxic_unique_stereo'] else '[]'
        
        rows.append(result_row)
    
    return pd.DataFrame(rows)


def main():
    """Main function to process only_stereo differences."""
    input_csv = Path(__file__).parent.parent / 'pairs_fg_stereo_merged_nodot.csv'
    output_dir = Path(__file__).parent
    
    print(f"Loading data from: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Total rows: {len(df)}")
    
    # Filter to only_stereo
    df_stereo = df[df['diff_type'] == 'only_stereo'].copy()
    print(f"only_stereo rows: {len(df_stereo)}")
    
    # Process one_step and multi_step separately
    for step_type in ['one_step', 'multi_step']:
        df_step = df_stereo[df_stereo['step_num'] == step_type].copy()
        print(f"\n{step_type} rows: {len(df_step)}")
        
        if len(df_step) == 0:
            print(f"  No {step_type} rows found. Skipping.")
            continue
        
        # Create difference DataFrame
        diff_df = create_stereo_diff_dataframe(df_step)
        
        # Save to CSV
        output_file = output_dir / f'only_stereo_{step_type}_diff.csv'
        diff_df.to_csv(output_file, index=False)
        print(f"  Saved to: {output_file}")
        print(f"  Columns: {len(diff_df.columns)}")
        print(f"  Rows: {len(diff_df)}")


if __name__ == "__main__":
    main()
