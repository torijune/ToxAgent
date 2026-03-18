"""Generate statistics for pairs_fg_stereo_merged.csv.

This script generates comprehensive statistics about:
- Total number of pairs
- Number of unique datasets
- Number of unique endpoints
- Pairs per dataset
- Pairs per endpoint
- Pairs per dataset-endpoint combination
- Distribution by diff_type and step_num
"""
import pandas as pd
from pathlib import Path
from collections import Counter


def generate_pair_statistics(csv_path: str) -> dict:
    """Generate comprehensive statistics for pairs.
    
    Args:
        csv_path: Path to pairs_fg_stereo_merged.csv
        
    Returns:
        Dictionary containing all statistics
    """
    print(f"Loading CSV from {csv_path}...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"Total rows: {len(df):,}")
    print()
    
    stats = {}
    
    # 1. 기본 통계
    stats['total_pairs'] = len(df)
    stats['unique_datasets'] = df['dataset_name'].nunique()
    stats['unique_endpoints'] = df['endpoint'].nunique()
    
    # 2. Dataset별 통계
    dataset_stats = df['dataset_name'].value_counts().to_dict()
    stats['pairs_per_dataset'] = dataset_stats
    
    # 3. Endpoint별 통계
    endpoint_stats = df['endpoint'].value_counts().to_dict()
    stats['pairs_per_endpoint'] = endpoint_stats
    
    # 4. Dataset-Endpoint 조합별 통계
    dataset_endpoint_stats = df.groupby(['dataset_name', 'endpoint']).size().to_dict()
    stats['pairs_per_dataset_endpoint'] = dataset_endpoint_stats
    
    # 5. diff_type별 통계
    if 'diff_type' in df.columns:
        diff_type_stats = df['diff_type'].value_counts().to_dict()
        stats['pairs_per_diff_type'] = diff_type_stats
    else:
        stats['pairs_per_diff_type'] = {}
    
    # 6. step_num별 통계
    if 'step_num' in df.columns:
        step_num_stats = df['step_num'].value_counts().to_dict()
        stats['pairs_per_step_num'] = step_num_stats
    else:
        stats['pairs_per_step_num'] = {}
    
    # 7. diff_type x step_num 교차 통계
    if 'diff_type' in df.columns and 'step_num' in df.columns:
        cross_stats = pd.crosstab(df['diff_type'], df['step_num']).to_dict()
        stats['pairs_per_diff_type_step_num'] = cross_stats
    else:
        stats['pairs_per_diff_type_step_num'] = {}
    
    # 8. Dataset별 diff_type 분포
    if 'diff_type' in df.columns:
        dataset_diff_type = df.groupby(['dataset_name', 'diff_type']).size().to_dict()
        stats['pairs_per_dataset_diff_type'] = dataset_diff_type
    else:
        stats['pairs_per_dataset_diff_type'] = {}
    
    # 9. Dataset별 step_num 분포
    if 'step_num' in df.columns:
        dataset_step_num = df.groupby(['dataset_name', 'step_num']).size().to_dict()
        stats['pairs_per_dataset_step_num'] = dataset_step_num
    else:
        stats['pairs_per_dataset_step_num'] = {}
    
    # 10. Endpoint별 diff_type 분포
    if 'diff_type' in df.columns:
        endpoint_diff_type = df.groupby(['endpoint', 'diff_type']).size().to_dict()
        stats['pairs_per_endpoint_diff_type'] = endpoint_diff_type
    else:
        stats['pairs_per_endpoint_diff_type'] = {}
    
    # 11. Endpoint별 step_num 분포
    if 'step_num' in df.columns:
        endpoint_step_num = df.groupby(['endpoint', 'step_num']).size().to_dict()
        stats['pairs_per_endpoint_step_num'] = endpoint_step_num
    else:
        stats['pairs_per_endpoint_step_num'] = {}
    
    # 12. n_fg_diff, n_stereo_diff, n_diff_features 분포
    if 'n_fg_diff' in df.columns:
        stats['n_fg_diff_distribution'] = df['n_fg_diff'].value_counts().sort_index().to_dict()
    if 'n_stereo_diff' in df.columns:
        stats['n_stereo_diff_distribution'] = df['n_stereo_diff'].value_counts().sort_index().to_dict()
    if 'n_diff_features' in df.columns:
        stats['n_diff_features_distribution'] = df['n_diff_features'].value_counts().sort_index().to_dict()
    
    return stats


def print_statistics(stats: dict):
    """Print statistics in a readable format.
    
    Args:
        stats: Statistics dictionary from generate_pair_statistics
    """
    print("=" * 80)
    print("PAIR STATISTICS")
    print("=" * 80)
    print()
    
    # 1. 기본 통계
    print("1. 기본 통계")
    print("-" * 80)
    print(f"  총 Pair 수: {stats['total_pairs']:,}")
    print(f"  고유 Dataset 수: {stats['unique_datasets']}")
    print(f"  고유 Endpoint 수: {stats['unique_endpoints']}")
    print()
    
    # 2. Dataset별 통계
    print("2. Dataset별 Pair 수")
    print("-" * 80)
    for dataset, count in sorted(stats['pairs_per_dataset'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_pairs']) * 100
        print(f"  {dataset:30s}: {count:6,} ({percentage:5.2f}%)")
    print()
    
    # 3. Endpoint별 통계
    print("3. Endpoint별 Pair 수")
    print("-" * 80)
    for endpoint, count in sorted(stats['pairs_per_endpoint'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_pairs']) * 100
        print(f"  {endpoint:50s}: {count:6,} ({percentage:5.2f}%)")
    print()
    
    # 4. Dataset-Endpoint 조합별 통계 (상위 20개)
    print("4. Dataset-Endpoint 조합별 Pair 수 (상위 20개)")
    print("-" * 80)
    sorted_combinations = sorted(stats['pairs_per_dataset_endpoint'].items(), key=lambda x: x[1], reverse=True)
    for (dataset, endpoint), count in sorted_combinations[:20]:
        print(f"  {dataset:20s} | {endpoint:40s}: {count:6,}")
    print()
    
    # 5. diff_type별 통계
    if stats['pairs_per_diff_type']:
        print("5. diff_type별 Pair 수")
        print("-" * 80)
        for diff_type, count in sorted(stats['pairs_per_diff_type'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_pairs']) * 100
            print(f"  {diff_type:20s}: {count:6,} ({percentage:5.2f}%)")
        print()
    
    # 6. step_num별 통계
    if stats['pairs_per_step_num']:
        print("6. step_num별 Pair 수")
        print("-" * 80)
        for step_num, count in sorted(stats['pairs_per_step_num'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_pairs']) * 100
            print(f"  {step_num:20s}: {count:6,} ({percentage:5.2f}%)")
        print()
    
    # 7. diff_type x step_num 교차 통계
    if stats['pairs_per_diff_type_step_num']:
        print("7. diff_type x step_num 교차 통계")
        print("-" * 80)
        # Convert nested dict to readable format
        if isinstance(list(stats['pairs_per_diff_type_step_num'].values())[0], dict):
            for diff_type in sorted(stats['pairs_per_diff_type_step_num'].keys()):
                print(f"  {diff_type}:")
                for step_num in sorted(stats['pairs_per_diff_type_step_num'][diff_type].keys()):
                    count = stats['pairs_per_diff_type_step_num'][diff_type][step_num]
                    print(f"    {step_num:20s}: {count:6,}")
        print()
    
    # 8. n_fg_diff, n_stereo_diff, n_diff_features 분포
    if 'n_fg_diff_distribution' in stats:
        print("8. n_fg_diff 분포")
        print("-" * 80)
        for n_diff, count in sorted(stats['n_fg_diff_distribution'].items()):
            percentage = (count / stats['total_pairs']) * 100
            print(f"  n_fg_diff = {n_diff:2d}: {count:6,} ({percentage:5.2f}%)")
        print()
    
    if 'n_stereo_diff_distribution' in stats:
        print("9. n_stereo_diff 분포")
        print("-" * 80)
        for n_diff, count in sorted(stats['n_stereo_diff_distribution'].items()):
            percentage = (count / stats['total_pairs']) * 100
            print(f"  n_stereo_diff = {n_diff:2d}: {count:6,} ({percentage:5.2f}%)")
        print()
    
    if 'n_diff_features_distribution' in stats:
        print("10. n_diff_features 분포")
        print("-" * 80)
        for n_diff, count in sorted(stats['n_diff_features_distribution'].items()):
            percentage = (count / stats['total_pairs']) * 100
            print(f"  n_diff_features = {n_diff:2d}: {count:6,} ({percentage:5.2f}%)")
        print()


def generate_markdown_report(stats: dict, output_path: str):
    """Generate a markdown report from statistics.
    
    Args:
        stats: Statistics dictionary from generate_pair_statistics
        output_path: Path to save the markdown file
    """
    md_lines = []
    
    md_lines.append("# Pair 통계 보고서")
    md_lines.append("")
    md_lines.append("## 1. 기본 통계")
    md_lines.append("")
    md_lines.append(f"- **총 Pair 수**: {stats['total_pairs']:,}개")
    md_lines.append(f"- **고유 Dataset 수**: {stats['unique_datasets']}개")
    md_lines.append(f"- **고유 Endpoint 수**: {stats['unique_endpoints']}개")
    md_lines.append("")
    
    md_lines.append("## 2. Dataset별 Pair 수")
    md_lines.append("")
    md_lines.append("| Dataset | Pair 수 | 비율 (%) |")
    md_lines.append("|---------|---------|----------|")
    for dataset, count in sorted(stats['pairs_per_dataset'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_pairs']) * 100
        md_lines.append(f"| {dataset} | {count:,} | {percentage:.2f} |")
    md_lines.append("")
    
    md_lines.append("## 3. Endpoint별 Pair 수")
    md_lines.append("")
    md_lines.append("| Endpoint | Pair 수 | 비율 (%) |")
    md_lines.append("|----------|---------|----------|")
    for endpoint, count in sorted(stats['pairs_per_endpoint'].items(), key=lambda x: x[1], reverse=True):
        percentage = (count / stats['total_pairs']) * 100
        # Endpoint 이름이 너무 길면 자르기
        endpoint_display = endpoint[:60] + "..." if len(endpoint) > 60 else endpoint
        md_lines.append(f"| {endpoint_display} | {count:,} | {percentage:.2f} |")
    md_lines.append("")
    
    md_lines.append("## 4. Dataset-Endpoint 조합별 Pair 수 (상위 30개)")
    md_lines.append("")
    md_lines.append("| Dataset | Endpoint | Pair 수 |")
    md_lines.append("|---------|----------|---------|")
    sorted_combinations = sorted(stats['pairs_per_dataset_endpoint'].items(), key=lambda x: x[1], reverse=True)
    for (dataset, endpoint), count in sorted_combinations[:30]:
        endpoint_display = endpoint[:50] + "..." if len(endpoint) > 50 else endpoint
        md_lines.append(f"| {dataset} | {endpoint_display} | {count:,} |")
    md_lines.append("")
    
    if stats['pairs_per_diff_type']:
        md_lines.append("## 5. diff_type별 Pair 수")
        md_lines.append("")
        md_lines.append("| diff_type | Pair 수 | 비율 (%) |")
        md_lines.append("|-----------|---------|----------|")
        for diff_type, count in sorted(stats['pairs_per_diff_type'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_pairs']) * 100
            md_lines.append(f"| {diff_type} | {count:,} | {percentage:.2f} |")
        md_lines.append("")
    
    if stats['pairs_per_step_num']:
        md_lines.append("## 6. step_num별 Pair 수")
        md_lines.append("")
        md_lines.append("| step_num | Pair 수 | 비율 (%) |")
        md_lines.append("|----------|---------|----------|")
        for step_num, count in sorted(stats['pairs_per_step_num'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_pairs']) * 100
            md_lines.append(f"| {step_num} | {count:,} | {percentage:.2f} |")
        md_lines.append("")
    
    if stats['pairs_per_diff_type_step_num']:
        md_lines.append("## 7. diff_type x step_num 교차 통계")
        md_lines.append("")
        md_lines.append("| diff_type | step_num | Pair 수 |")
        md_lines.append("|-----------|----------|---------|")
        if isinstance(list(stats['pairs_per_diff_type_step_num'].values())[0], dict):
            for diff_type in sorted(stats['pairs_per_diff_type_step_num'].keys()):
                for step_num in sorted(stats['pairs_per_diff_type_step_num'][diff_type].keys()):
                    count = stats['pairs_per_diff_type_step_num'][diff_type][step_num]
                    md_lines.append(f"| {diff_type} | {step_num} | {count:,} |")
        md_lines.append("")
    
    if 'n_fg_diff_distribution' in stats:
        md_lines.append("## 8. n_fg_diff 분포")
        md_lines.append("")
        md_lines.append("| n_fg_diff | Pair 수 | 비율 (%) |")
        md_lines.append("|-----------|---------|----------|")
        for n_diff, count in sorted(stats['n_fg_diff_distribution'].items()):
            percentage = (count / stats['total_pairs']) * 100
            md_lines.append(f"| {n_diff} | {count:,} | {percentage:.2f} |")
        md_lines.append("")
    
    if 'n_stereo_diff_distribution' in stats:
        md_lines.append("## 9. n_stereo_diff 분포")
        md_lines.append("")
        md_lines.append("| n_stereo_diff | Pair 수 | 비율 (%) |")
        md_lines.append("|--------------|---------|----------|")
        for n_diff, count in sorted(stats['n_stereo_diff_distribution'].items()):
            percentage = (count / stats['total_pairs']) * 100
            md_lines.append(f"| {n_diff} | {count:,} | {percentage:.2f} |")
        md_lines.append("")
    
    if 'n_diff_features_distribution' in stats:
        md_lines.append("## 10. n_diff_features 분포")
        md_lines.append("")
        md_lines.append("| n_diff_features | Pair 수 | 비율 (%) |")
        md_lines.append("|----------------|---------|----------|")
        for n_diff, count in sorted(stats['n_diff_features_distribution'].items()):
            percentage = (count / stats['total_pairs']) * 100
            md_lines.append(f"| {n_diff} | {count:,} | {percentage:.2f} |")
        md_lines.append("")
    
    # Dataset별 상세 통계
    if stats['pairs_per_dataset_diff_type']:
        md_lines.append("## 11. Dataset별 diff_type 분포")
        md_lines.append("")
        # Group by dataset
        dataset_diff_types = {}
        for (dataset, diff_type), count in stats['pairs_per_dataset_diff_type'].items():
            if dataset not in dataset_diff_types:
                dataset_diff_types[dataset] = {}
            dataset_diff_types[dataset][diff_type] = count
        
        for dataset in sorted(dataset_diff_types.keys()):
            md_lines.append(f"### {dataset}")
            md_lines.append("")
            md_lines.append("| diff_type | Pair 수 |")
            md_lines.append("|-----------|---------|")
            for diff_type, count in sorted(dataset_diff_types[dataset].items(), key=lambda x: x[1], reverse=True):
                md_lines.append(f"| {diff_type} | {count:,} |")
            md_lines.append("")
    
    if stats['pairs_per_dataset_step_num']:
        md_lines.append("## 12. Dataset별 step_num 분포")
        md_lines.append("")
        # Group by dataset
        dataset_step_nums = {}
        for (dataset, step_num), count in stats['pairs_per_dataset_step_num'].items():
            if dataset not in dataset_step_nums:
                dataset_step_nums[dataset] = {}
            dataset_step_nums[dataset][step_num] = count
        
        for dataset in sorted(dataset_step_nums.keys()):
            md_lines.append(f"### {dataset}")
            md_lines.append("")
            md_lines.append("| step_num | Pair 수 |")
            md_lines.append("|----------|---------|")
            for step_num, count in sorted(dataset_step_nums[dataset].items(), key=lambda x: x[1], reverse=True):
                md_lines.append(f"| {step_num} | {count:,} |")
            md_lines.append("")
    
    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))
    
    print(f"\nMarkdown report saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate statistics for pairs_fg_stereo_merged.csv"
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
        default="Pair_Stat.md",
        help="Output markdown file path (default: Pair_Stat.md)"
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    script_dir = Path(__file__).parent
    input_path = script_dir / args.input if not Path(args.input).is_absolute() else Path(args.input)
    output_path = script_dir / args.output if not Path(args.output).is_absolute() else Path(args.output)
    
    # Generate statistics
    stats = generate_pair_statistics(str(input_path))
    
    # Print statistics
    print_statistics(stats)
    
    # Generate markdown report
    generate_markdown_report(stats, str(output_path))
    
    print(f"\n✓ Statistics generation completed!")
