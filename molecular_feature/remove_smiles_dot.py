"""
Remove pairs where toxic_smiles or nontoxic_smiles contain multiple molecules separated by "."

This script filters out rows where the SMILES strings contain multiple molecules
(e.g., salt forms, metal complexes) separated by periods.
"""
import csv
from pathlib import Path
from typing import List, Dict, Any


def has_multiple_molecules(smiles: str) -> bool:
    """
    Check if a SMILES string contains multiple molecules separated by "."
    
    Args:
        smiles: SMILES string to check
        
    Returns:
        True if multiple molecules are detected, False otherwise
    """
    if not smiles or not isinstance(smiles, str):
        return False
    
    smiles = smiles.strip()
    
    # Check if "." is present
    if '.' not in smiles:
        return False
    
    # Split by "." and check for multiple valid molecules
    parts = [x.strip() for x in smiles.split('.') if x.strip()]
    
    # Filter out very short parts (likely not valid molecules)
    # Also filter out single atoms/elements like [Co], [Na], etc.
    valid_parts = []
    for part in parts:
        part = part.strip()
        # Skip very short parts or single element notations
        if len(part) >= 3 and not (part.startswith('[') and part.endswith(']') and len(part) < 10):
            valid_parts.append(part)
    
    # If we have more than one valid molecule part, it's multiple molecules
    return len(valid_parts) > 1


def filter_pairs_without_dot(
    input_csv: str,
    output_csv: str,
    toxic_col: str = "toxic_smiles",
    nontoxic_col: str = "nontoxic_smiles"
) -> tuple[int, int]:
    """
    Remove pairs where toxic_smiles or nontoxic_smiles contain multiple molecules.
    
    Args:
        input_csv: Path to input CSV file
        output_csv: Path to output CSV file
        toxic_col: Name of toxic SMILES column
        nontoxic_col: Name of nontoxic SMILES column
        
    Returns:
        Tuple of (total_rows, filtered_rows)
    """
    input_path = Path(input_csv)
    output_path = Path(output_csv)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    total_rows = 0
    filtered_rows = 0
    removed_rows = 0
    
    print(f"Reading from: {input_csv}")
    print(f"Writing to: {output_csv}")
    print()
    
    with open(input_path, 'r', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames
        
        if fieldnames is None:
            raise ValueError("CSV file has no header")
        
        if toxic_col not in fieldnames:
            raise ValueError(f"Column '{toxic_col}' not found in CSV")
        if nontoxic_col not in fieldnames:
            raise ValueError(f"Column '{nontoxic_col}' not found in CSV")
        
        with open(output_path, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in reader:
                total_rows += 1
                
                toxic_smiles = row.get(toxic_col, '')
                nontoxic_smiles = row.get(nontoxic_col, '')
                
                # Check if either SMILES contains multiple molecules
                toxic_has_multiple = has_multiple_molecules(toxic_smiles)
                nontoxic_has_multiple = has_multiple_molecules(nontoxic_smiles)
                
                if toxic_has_multiple or nontoxic_has_multiple:
                    removed_rows += 1
                    if removed_rows <= 10:  # Print first 10 removed rows
                        reason = []
                        if toxic_has_multiple:
                            reason.append(f"{toxic_col} has multiple molecules")
                        if nontoxic_has_multiple:
                            reason.append(f"{nontoxic_col} has multiple molecules")
                        print(f"  Removing row {total_rows}: {', '.join(reason)}")
                else:
                    writer.writerow(row)
                    filtered_rows += 1
    
    print()
    print("=" * 60)
    print("Filtering Summary")
    print("=" * 60)
    print(f"Total rows processed: {total_rows:,}")
    print(f"Rows removed: {removed_rows:,} ({removed_rows/total_rows*100:.2f}%)")
    print(f"Rows kept: {filtered_rows:,} ({filtered_rows/total_rows*100:.2f}%)")
    print("=" * 60)
    
    return total_rows, filtered_rows


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Remove pairs with multiple molecules in SMILES columns"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="pairs_fg_stereo_merged.csv",
        help="Input CSV file path (default: pairs_fg_stereo_merged.csv)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pairs_fg_stereo_merged_nodot.csv",
        help="Output CSV file path (default: pairs_fg_stereo_merged_nodot.csv)"
    )
    parser.add_argument(
        "--toxic-col",
        type=str,
        default="toxic_smiles",
        help="Name of toxic SMILES column (default: toxic_smiles)"
    )
    parser.add_argument(
        "--nontoxic-col",
        type=str,
        default="nontoxic_smiles",
        help="Name of nontoxic SMILES column (default: nontoxic_smiles)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input
    output_path = script_dir / args.output
    
    try:
        total, filtered = filter_pairs_without_dot(
            str(input_path),
            str(output_path),
            toxic_col=args.toxic_col,
            nontoxic_col=args.nontoxic_col
        )
        print(f"\n✅ Successfully created filtered CSV: {output_path}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
