"""Add type and step_num columns to CSV files based on n_fg_diff, n_stereo_diff, and n_diff_features.

Type classification (diff_type):
- only_fg: n_fg_diff > 0 and n_stereo_diff == 0
- only_stereo: n_stereo_diff > 0 and n_fg_diff == 0
- both: n_fg_diff > 0 and n_stereo_diff > 0
- none: n_fg_diff == 0 and n_stereo_diff == 0

Step classification (step_num):
- one_step: n_diff_features == 1
- multi_step: n_diff_features > 1
"""
import pandas as pd
from pathlib import Path


def add_type_column(
    csv_path: str,
    output_path: str = None,
    type_column_name: str = "diff_type"
) -> pd.DataFrame:
    """Add type column to CSV based on n_fg_diff and n_stereo_diff.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save output CSV (if None, overwrites input file)
        type_column_name: Name of the type column to add
        
    Returns:
        DataFrame with added type column
    """
    # Load CSV
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"Total rows: {len(df)}")
    
    # Check if columns exist
    if 'n_fg_diff' not in df.columns:
        raise ValueError("Column 'n_fg_diff' not found in CSV")
    if 'n_stereo_diff' not in df.columns:
        raise ValueError("Column 'n_stereo_diff' not found in CSV")
    
    # Classify based on n_fg_diff and n_stereo_diff
    def classify_type(row):
        n_fg = row['n_fg_diff']
        n_stereo = row['n_stereo_diff']
        
        if n_fg > 0 and n_stereo == 0:
            return 'only_fg'
        elif n_fg == 0 and n_stereo > 0:
            return 'only_stereo'
        elif n_fg > 0 and n_stereo > 0:
            return 'both'
        else:
            return 'none'  # Should not happen in pairs_one_diff_only.csv
    
    # Add type column
    df[type_column_name] = df.apply(classify_type, axis=1)
    
    # Print statistics
    print(f"\nType distribution:")
    type_counts = df[type_column_name].value_counts()
    for type_name, count in type_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {type_name}: {count} ({percentage:.2f}%)")
    
    # Save output
    if output_path is None:
        output_path = csv_path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return df


def add_type_and_step_columns(
    csv_path: str,
    output_path: str = None,
    type_column_name: str = "diff_type",
    step_column_name: str = "step_num"
) -> pd.DataFrame:
    """Add diff_type and step_num columns to CSV based on n_fg_diff, n_stereo_diff, and n_diff_features.
    
    Args:
        csv_path: Path to input CSV file
        output_path: Optional path to save output CSV (if None, overwrites input file)
        type_column_name: Name of the diff_type column to add
        step_column_name: Name of the step_num column to add
        
    Returns:
        DataFrame with added columns
    """
    # Load CSV
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total rows: {len(df)}")
    
    # Check if required columns exist
    required_columns = ['n_fg_diff', 'n_stereo_diff', 'n_diff_features']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Classify diff_type based on n_fg_diff and n_stereo_diff
    def classify_type(row):
        n_fg = row['n_fg_diff']
        n_stereo = row['n_stereo_diff']
        
        if n_fg > 0 and n_stereo == 0:
            return 'only_fg'
        elif n_fg == 0 and n_stereo > 0:
            return 'only_stereo'
        elif n_fg > 0 and n_stereo > 0:
            return 'both'
        else:
            return 'none'
    
    # Classify step_num based on n_diff_features
    def classify_step(row):
        n_diff = row['n_diff_features']
        if n_diff == 1:
            return 'one_step'
        else:
            return 'multi_step'
    
    # Add columns
    df[type_column_name] = df.apply(classify_type, axis=1)
    df[step_column_name] = df.apply(classify_step, axis=1)
    
    # Print statistics
    print(f"\n{type_column_name} distribution:")
    type_counts = df[type_column_name].value_counts()
    for type_name, count in type_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {type_name}: {count} ({percentage:.2f}%)")
    
    print(f"\n{step_column_name} distribution:")
    step_counts = df[step_column_name].value_counts()
    for step_name, count in step_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {step_name}: {count} ({percentage:.2f}%)")
    
    # Print cross-tabulation
    print(f"\nCross-tabulation ({type_column_name} x {step_column_name}):")
    cross_tab = pd.crosstab(df[type_column_name], df[step_column_name])
    print(cross_tab)
    
    # Save output
    if output_path is None:
        output_path = csv_path
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    
    return df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Add type and step_num columns to CSV files based on n_fg_diff, n_stereo_diff, and n_diff_features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add only diff_type column (for pairs_one_diff_only.csv)
  python add_type.py --input pairs_one_diff_only.csv --type-only
  
  # Add both diff_type and step_num columns (for pairs_fg_stereo_merged.csv)
  python add_type.py --input pairs_fg_stereo_merged.csv --both
        """
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input CSV file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: overwrites input file)"
    )
    parser.add_argument(
        "--type-only",
        action="store_true",
        help="Add only diff_type column (for pairs_one_diff_only.csv)"
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="Add both diff_type and step_num columns (for pairs_fg_stereo_merged.csv)"
    )
    parser.add_argument(
        "--type-column-name",
        type=str,
        default="diff_type",
        help="Name of the diff_type column (default: diff_type)"
    )
    parser.add_argument(
        "--step-column-name",
        type=str,
        default="step_num",
        help="Name of the step_num column (default: step_num)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths relative to script directory
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = script_dir / args.output if args.output and not Path(args.output).is_absolute() else (Path(args.output) if args.output else None)
    
    if args.both:
        # Add both columns
        df = add_type_and_step_columns(
            csv_path=str(input_path),
            output_path=str(output_path) if output_path else None,
            type_column_name=args.type_column_name,
            step_column_name=args.step_column_name
        )
        print(f"\n✓ Successfully added '{args.type_column_name}' and '{args.step_column_name}' columns to {len(df)} rows")
    else:
        # Add only diff_type column (default behavior)
        df = add_type_column(
            csv_path=str(input_path),
            output_path=str(output_path) if output_path else None,
            type_column_name=args.type_column_name
        )
        print(f"\n✓ Successfully added '{args.type_column_name}' column to {len(df)} rows")
