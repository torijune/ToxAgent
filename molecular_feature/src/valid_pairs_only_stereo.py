"""
Validation-by-Reconstruction for only_stereo pairs.

This script validates that pairs differ ONLY in stereochemistry by:
1. Comparing non-isomeric canonical SMILES of both molecules
2. If they match, the pair is valid (only stereo differs, structure is identical)
3. If they don't match, the pair is invalid (structure differs beyond stereo)

Unlike FG validation, we don't "remove" stereo features. Instead, we compare
the underlying molecular structure ignoring stereochemistry.
"""
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from tqdm import tqdm
from rdkit import Chem

# Suppress RDKit warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')


def validate_stereo_pair(
    toxic_smiles: str,
    nontoxic_smiles: str,
    toxic_canonical: str,
    nontoxic_canonical: str
) -> Tuple[bool, str]:
    """Validate stereo pair by comparing non-isomeric canonical SMILES.
    
    For only_stereo pairs, the underlying molecular structure should be identical.
    The only difference should be in stereochemistry (R/S, E/Z).
    
    Args:
        toxic_smiles: Toxic SMILES
        nontoxic_smiles: Nontoxic SMILES
        toxic_canonical: Toxic canonical SMILES
        nontoxic_canonical: Nontoxic canonical SMILES
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # Parse molecules from canonical SMILES
        toxic_mol = Chem.MolFromSmiles(toxic_canonical)
        nontoxic_mol = Chem.MolFromSmiles(nontoxic_canonical)
        
        if not toxic_mol or not nontoxic_mol:
            return False, "Failed to parse molecules"
        
        # Get non-isomeric canonical SMILES (ignores stereochemistry)
        # This removes all stereo information (R/S, E/Z) and compares structure only
        toxic_noniso = Chem.MolToSmiles(toxic_mol, isomericSmiles=False, canonical=True)
        nontoxic_noniso = Chem.MolToSmiles(nontoxic_mol, isomericSmiles=False, canonical=True)
        
        # If non-isomeric SMILES match, the structures are identical
        # Only stereochemistry differs, which is valid for only_stereo pairs
        if toxic_noniso == nontoxic_noniso:
            return True, "Valid: Structures match (only stereo differs)"
        else:
            return False, f"Invalid: Structures differ beyond stereo. Toxic: {toxic_noniso[:50]}..., Nontoxic: {nontoxic_noniso[:50]}..."
        
    except Exception as e:
        return False, f"Error during validation: {str(e)}"


def validate_pairs(
    input_csv: Path,
    output_csv: Path,
    limit: Optional[int] = None
) -> None:
    """Validate only_stereo pairs by comparing non-isomeric structures.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        limit: Optional limit on number of pairs to process
    """
    print(f"Loading pairs from: {input_csv}")
    df = pd.read_csv(input_csv, low_memory=False)
    print(f"Total rows: {len(df)}")
    
    # Filter to only_stereo
    df_stereo = df[df['diff_type'] == 'only_stereo'].copy()
    print(f"only_stereo rows: {len(df_stereo)}")
    
    if len(df_stereo) == 0:
        print("No only_stereo pairs found. Exiting.")
        return
    
    # Process pairs
    valid_rows = []
    invalid_rows = []
    
    df_to_process = df_stereo.head(limit) if limit else df_stereo
    
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="Validating pairs"):
        try:
            # Extract data
            toxic_smiles = str(row.get('toxic_smiles', '')).strip()
            nontoxic_smiles = str(row.get('nontoxic_smiles', '')).strip()
            toxic_canonical = str(row.get('toxic_canonical_smiles', '')).strip()
            nontoxic_canonical = str(row.get('nontoxic_canonical_smiles', '')).strip()
            
            if not toxic_smiles or not nontoxic_smiles or pd.isna(toxic_smiles) or pd.isna(nontoxic_smiles):
                invalid_rows.append({
                    'row_index': idx,
                    'reason': 'Missing SMILES',
                    'original_row': row.to_dict()
                })
                continue
            
            # Use canonical SMILES if available, otherwise use original
            if not toxic_canonical or pd.isna(toxic_canonical):
                toxic_canonical = toxic_smiles
            if not nontoxic_canonical or pd.isna(nontoxic_canonical):
                nontoxic_canonical = nontoxic_smiles
            
            # Validate
            is_valid, error_msg = validate_stereo_pair(
                toxic_smiles,
                nontoxic_smiles,
                toxic_canonical,
                nontoxic_canonical
            )
            
            if is_valid:
                # Add validation status to row
                row_dict = row.to_dict()
                row_dict['validation_status'] = 'valid'
                row_dict['validation_message'] = error_msg
                valid_rows.append(row_dict)
            else:
                invalid_rows.append({
                    'row_index': idx,
                    'reason': error_msg,
                    'original_row': row.to_dict()
                })
                
        except Exception as e:
            invalid_rows.append({
                'row_index': idx,
                'reason': f'Exception: {str(e)}',
                'original_row': row.to_dict()
            })
    
    # Create output DataFrame
    if valid_rows:
        df_valid = pd.DataFrame(valid_rows)
        df_valid.to_csv(output_csv, index=False)
        print(f"\n✅ Validated {len(valid_rows)} pairs")
        print(f"✅ Saved to: {output_csv}")
    else:
        print(f"\n⚠️ No valid pairs found")
    
    if invalid_rows:
        print(f"❌ Invalid pairs: {len(invalid_rows)}")
        # Optionally save invalid pairs for debugging
        invalid_csv = output_csv.parent / f"{output_csv.stem}_invalid.csv"
        df_invalid = pd.DataFrame(invalid_rows)
        df_invalid.to_csv(invalid_csv, index=False)
        print(f"   Invalid pairs saved to: {invalid_csv}")


def main():
    """Main function."""
    input_csv = Path(__file__).parent.parent / 'pairs_fg_stereo_merged_nodot.csv'
    output_csv = Path(__file__).parent.parent / 'pairs_fg_stereo_merged_nodot_stereo_validated.csv'
    
    validate_pairs(input_csv, output_csv)


if __name__ == "__main__":
    main()
