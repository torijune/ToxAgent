from typing import Tuple, Optional
import pandas as pd
import os
import pickle
from pathlib import Path
import ast

def map_existing_fg_data(merged_df: pd.DataFrame, 
                        existing_fg_file: str,
                        smiles_column: str = 'X',
                        id_column: str = 'Drug_ID') -> pd.DataFrame:
    """
    기존 FG 정보 파일을 사용하여 DataFrame에 FG 정보를 매핑
    
    Args:
        merged_df: 입력 DataFrame
        existing_fg_file: 기존 FG 정보 파일 경로
        smiles_column: SMILES 컬럼명
        id_column: ID 컬럼명
        
    Returns:
        FG 정보가 매핑된 DataFrame
    """
    print(f"📂 기존 FG 정보 파일 로드: {existing_fg_file}")
    
    # 기존 FG 파일 로드 (안전한 방식)
    try:
        if existing_fg_file.endswith('.parquet'):
            fg_df = pd.read_parquet(existing_fg_file)
        else:
            # CSV 파일 로드 시 오류 처리
            try:
                fg_df = pd.read_csv(existing_fg_file)
            except pd.errors.ParserError as e:
                print(f"⚠️ CSV 파싱 오류 발생, 오류 행을 건너뛰고 로드합니다: {e}")
                fg_df = pd.read_csv(existing_fg_file, on_bad_lines='skip')
    except Exception as e:
        raise ValueError(f"FG 파일 로드 실패: {e}")
    
    print(f"✅ FG 정보 파일 로드 완료: {len(fg_df)}개 분자")
    print(f"📊 FG 파일 컬럼: {fg_df.columns.tolist()}")
    
    # SMILES 컬럼명 자동 감지 및 통일
    # FG 파일에서 SMILES 컬럼 찾기
    fg_smiles_col = None
    for col in ['X', 'Drug', 'smiles', 'SMILES', 'Smiles']:
        if col in fg_df.columns:
            fg_smiles_col = col
            break
    
    if fg_smiles_col is None:
        raise ValueError(f"FG 파일에서 SMILES 컬럼을 찾을 수 없습니다. 사용 가능한 컬럼: {fg_df.columns.tolist()}")
    
    # merged_df의 SMILES 컬럼을 'X'로 통일 (기본값이 'X'이므로)
    if smiles_column != 'X' and smiles_column in merged_df.columns:
        merged_df_temp = merged_df.copy()
        merged_df_temp['X'] = merged_df_temp[smiles_column]
    else:
        merged_df_temp = merged_df.copy()
    
    # Drug_ID가 없으면 생성
    if id_column not in merged_df_temp.columns:
        merged_df_temp[id_column] = range(len(merged_df_temp))
    
    # FG 정보와 매핑
    print("🔗 FG 정보 매핑 중...")
    
    # 문자열로 저장된 리스트/딕셔너리를 파싱하는 함수
    def safe_eval(x):
        if pd.isna(x):
            return {}
        
        # 문자열인 경우 처리
        if isinstance(x, str):
            x = x.strip()
            if x == '[]':
                return []
            elif x == '{}':
                return {}
            elif x == '' or x == 'nan':
                return {}
            
            # 연속된 빈 딕셔너리나 리스트 패턴 제거
            if x.startswith('{}{}{}') or x.startswith('[][][]'):
                print(f"⚠️ 손상된 데이터 감지, 기본값으로 대체: {x[:50]}...")
                return {}
            
            try:
                result = ast.literal_eval(x)
                # 결과 검증
                if isinstance(result, (list, dict)):
                    return result
                else:
                    return {}
            except (ValueError, SyntaxError) as e:
                print(f"⚠️ 파싱 오류, 기본값으로 대체: {x[:50]}... (오류: {e})")
                return {}
        
        # 이미 파이썬 객체인 경우
        if isinstance(x, (list, dict)):
            return x
        
        # 기타 경우 기본값 반환
        return {}
    
    # FG 정보 컬럼들을 안전하게 파싱
    fg_df['fg_names'] = fg_df['fg_names'].apply(safe_eval)
    fg_df['fg_counts'] = fg_df['fg_counts'].apply(safe_eval)
    fg_df['fg_full'] = fg_df['fg_full'].apply(safe_eval)
    
    # X 컬럼을 기준으로 매핑 (두 파일 모두 X 컬럼 사용)
    result_df = merged_df_temp.merge(
        fg_df[['X', 'fg_names', 'fg_counts', 'total_fg_count', 'fg_full']], 
        on='X', 
        how='left'
    )
    
    # 매핑되지 않은 분자 수 확인
    unmapped_count = result_df['fg_names'].isna().sum()
    if unmapped_count > 0:
        print(f"⚠️ {unmapped_count}개 분자의 FG 정보를 찾을 수 없습니다")
        # 매핑되지 않은 분자들에 대해 빈 FG 정보 할당
        result_df['fg_names'] = result_df['fg_names'].fillna('[]')
        result_df['fg_counts'] = result_df['fg_counts'].fillna('{}')
        result_df['total_fg_count'] = result_df['total_fg_count'].fillna(0)
        result_df['fg_full'] = result_df['fg_full'].fillna('{}')
    else:
        print("✅ 모든 분자의 FG 정보가 성공적으로 매핑되었습니다")
    
    # 데이터 타입 통일 (Parquet 저장을 위해)
    print("🔧 데이터 타입 통일 중...")
    
    # fg_names를 리스트로 변환
    def ensure_list(x):
        if isinstance(x, list):
            return x
        elif isinstance(x, str):
            try:
                result = ast.literal_eval(x)
                return result if isinstance(result, list) else []
            except:
                return []
        else:
            return []
    
    # fg_counts와 fg_full을 딕셔너리로 변환
    def ensure_dict(x):
        if isinstance(x, dict):
            return x
        elif isinstance(x, str):
            try:
                result = ast.literal_eval(x)
                return result if isinstance(result, dict) else {}
            except:
                return {}
        else:
            return {}
    
    result_df['fg_names'] = result_df['fg_names'].apply(ensure_list)
    result_df['fg_counts'] = result_df['fg_counts'].apply(ensure_dict)
    result_df['fg_full'] = result_df['fg_full'].apply(ensure_dict)
    result_df['total_fg_count'] = pd.to_numeric(result_df['total_fg_count'], errors='coerce').fillna(0).astype(int)
    
    print(f"📊 매핑 결과: {len(result_df)}개 분자")
    print(f"  - FG 정보가 있는 분자: {(result_df['total_fg_count'] > 0).sum()}개")
    print(f"  - FG 정보가 없는 분자: {(result_df['total_fg_count'] == 0).sum()}개")
    
    return result_df


def preprocessing_main(merged_df: pd.DataFrame, 
                      enable_checkpoint: bool = True,
                      max_workers: int = None,
                      checkpoint_path: str = None,
                      save_dir: str = "/Users/jang-wonjun/Desktop/DMISLab/FG_Level_DeToxicity/detoxicity_model/Preprocessed_data/",
                      resume_from_saved: bool = True,
                      existing_fg_file: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, pd.DataFrame]:
    """
    전처리 메인 함수 - 각 단계별로 중간 결과를 저장하며 진행
    
    Args:
        merged_df: 입력 DataFrame
        enable_checkpoint: 체크포인트 사용 여부
        max_workers: 워커 수
        checkpoint_path: 체크포인트 경로
        save_dir: 중간 결과 저장 디렉토리
        resume_from_saved: 저장된 파일에서 재시작할지 여부
        existing_fg_file: 기존 FG 정보가 있는 파일 경로 (CSV/Parquet)
        
    Returns:
        (merged_df_with_FG, merged_df_cross_sim_df, merged_df_pairs_df, merged_df_stats, merged_df_with_pair_info)
    """
    
    # 저장 디렉토리 생성
    os.makedirs(save_dir, exist_ok=True)
    
    # 파일 경로들 정의
    fg_file = os.path.join(save_dir, "merged_df_with_FG.parquet")
    cross_sim_file = os.path.join(save_dir, "merged_df_cross_sim_df.parquet")
    pairs_file = os.path.join(save_dir, "merged_df_pairs_df.parquet")
    stats_file = os.path.join(save_dir, "merged_df_stats.pkl")
    pair_info_file = os.path.join(save_dir, "merged_df_with_pair_info.parquet")
    
    print(f"📁 저장 디렉토리: {save_dir}")
    
    # 1단계: FG Information 추출
    print("\n🔬 1단계: Functional Group 정보 추출")
    if resume_from_saved and os.path.exists(fg_file):
        print(f"📂 기존 FG 데이터 로드: {fg_file}")
        merged_df_with_FG = pd.read_parquet(fg_file)
        print(f"✅ FG 데이터 로드 완료: {len(merged_df_with_FG)}개 분자")
    elif existing_fg_file and os.path.exists(existing_fg_file):
        print(f"📂 기존 FG 정보 파일 사용: {existing_fg_file}")
        merged_df_with_FG = map_existing_fg_data(merged_df, existing_fg_file)
        
        # FG 데이터 저장
        print(f"💾 FG 데이터 저장: {fg_file}")
        merged_df_with_FG.to_parquet(fg_file, index=False)
        print("✅ FG 데이터 저장 완료")
    else:
        print("🚀 FG 정보 추출 시작...")
        from FunctionalGroup_Tox_Preprocessor import process_dataset_with_fg
        merged_df_with_FG = process_dataset_with_fg(
            merged_df, 
            use_multithreading=True,
            max_workers=max_workers,
            enable_checkpoint=enable_checkpoint,
            checkpoint_path=checkpoint_path
        )
        
        # FG 데이터 저장
        print(f"💾 FG 데이터 저장: {fg_file}")
        merged_df_with_FG.to_parquet(fg_file, index=False)
        print("✅ FG 데이터 저장 완료")
    
    # 2단계: 유사도 분석 (Matrix 생성)
    print("\n🔍 2단계: 유사도 분석")
    if resume_from_saved and os.path.exists(cross_sim_file) and os.path.exists(pairs_file) and os.path.exists(stats_file):
        print(f"📂 기존 유사도 데이터 로드...")
        merged_df_cross_sim_df = pd.read_parquet(cross_sim_file)
        merged_df_pairs_df = pd.read_parquet(pairs_file)
        with open(stats_file, 'rb') as f:
            merged_df_stats = pickle.load(f)
        print(f"✅ 유사도 데이터 로드 완료: {merged_df_cross_sim_df.shape}, {len(merged_df_pairs_df)}개 쌍")
    else:
        print("🚀 유사도 분석 시작...")
        from FunctionalGroup_Tox_Preprocessor import analyze_similarity_from_single_df

        merged_df_cross_sim_df, merged_df_pairs_df, merged_df_stats = analyze_similarity_from_single_df(
            merged_df,
            label_column='Y',          # 기본값
            non_toxic_label=0,         # 기본값
            toxic_label=1,             # 기본값
            id_column='Drug_ID',       # 없으면 자동 생성
            smiles_column='X',         # X 컬럼 사용
            min_threshold=0.5,         # 범위 하한 (포함)
            max_threshold=0.8          # 범위 상한 (포함)
        )
        
        # 유사도 데이터 저장
        print(f"💾 유사도 데이터 저장...")
        merged_df_cross_sim_df.to_parquet(cross_sim_file, index=False)
        merged_df_pairs_df.to_parquet(pairs_file, index=False)
        with open(stats_file, 'wb') as f:
            pickle.dump(merged_df_stats, f)
        print("✅ 유사도 데이터 저장 완료")
    
    # 3단계: 유사도 기반 pair 정보 DataFrame 생성
    print("\n🔗 3단계: Pair 메타데이터 생성")
    if resume_from_saved and os.path.exists(pair_info_file):
        print(f"📂 기존 Pair 메타데이터 로드: {pair_info_file}")
        merged_df_with_pair_info = pd.read_parquet(pair_info_file)
        print(f"✅ Pair 메타데이터 로드 완료: {len(merged_df_with_pair_info)}개 쌍")
    else:
        print("🚀 Pair 메타데이터 생성 시작...")
        from FunctionalGroup_Tox_Preprocessor import build_pair_metadata_dataframe
        merged_df_with_pair_info = build_pair_metadata_dataframe(merged_df_with_FG, merged_df_pairs_df)
        
        # Pair 메타데이터 저장
        print(f"💾 Pair 메타데이터 저장: {pair_info_file}")
        merged_df_with_pair_info.to_parquet(pair_info_file, index=False)
        print("✅ Pair 메타데이터 저장 완료")
    
    print("\n🎉 모든 전처리 단계 완료!")
    print(f"📊 최종 결과:")
    print(f"  - FG 데이터: {len(merged_df_with_FG)}개 분자")
    print(f"  - 유사도 매트릭스: {merged_df_cross_sim_df.shape}")
    print(f"  - 유사도 쌍: {len(merged_df_pairs_df)}개")
    print(f"  - Pair 메타데이터: {len(merged_df_with_pair_info)}개 쌍")
    
    return merged_df_with_FG, merged_df_cross_sim_df, merged_df_pairs_df, merged_df_stats, merged_df_with_pair_info


def load_preprocessed_data(save_dir: str = "/Users/jang-wonjun/Desktop/DMISLab/FG_Level_DeToxicity/detoxicity_model/Preprocessed_data/") -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[dict], Optional[pd.DataFrame]]:
    """
    저장된 전처리 데이터를 로드하는 함수
    
    Args:
        save_dir: 저장 디렉토리 경로
        
    Returns:
        (merged_df_with_FG, merged_df_cross_sim_df, merged_df_pairs_df, merged_df_stats, merged_df_with_pair_info)
        파일이 없으면 None 반환
    """
    fg_file = os.path.join(save_dir, "merged_df_with_FG.parquet")
    cross_sim_file = os.path.join(save_dir, "merged_df_cross_sim_df.parquet")
    pairs_file = os.path.join(save_dir, "merged_df_pairs_df.parquet")
    stats_file = os.path.join(save_dir, "merged_df_stats.pkl")
    pair_info_file = os.path.join(save_dir, "merged_df_with_pair_info.parquet")
    
    merged_df_with_FG = None
    merged_df_cross_sim_df = None
    merged_df_pairs_df = None
    merged_df_stats = None
    merged_df_with_pair_info = None
    
    if os.path.exists(fg_file):
        merged_df_with_FG = pd.read_parquet(fg_file)
        print(f"✅ FG 데이터 로드: {len(merged_df_with_FG)}개 분자")
    
    if os.path.exists(cross_sim_file):
        merged_df_cross_sim_df = pd.read_parquet(cross_sim_file)
        print(f"✅ 유사도 매트릭스 로드: {merged_df_cross_sim_df.shape}")
    
    if os.path.exists(pairs_file):
        merged_df_pairs_df = pd.read_parquet(pairs_file)
        print(f"✅ 유사도 쌍 로드: {len(merged_df_pairs_df)}개 쌍")
    
    if os.path.exists(stats_file):
        with open(stats_file, 'rb') as f:
            merged_df_stats = pickle.load(f)
        print(f"✅ 통계 데이터 로드 완료")
    
    if os.path.exists(pair_info_file):
        merged_df_with_pair_info = pd.read_parquet(pair_info_file)
        print(f"✅ Pair 메타데이터 로드: {len(merged_df_with_pair_info)}개 쌍")
    
    return merged_df_with_FG, merged_df_cross_sim_df, merged_df_pairs_df, merged_df_stats, merged_df_with_pair_info


def check_preprocessing_status(save_dir: str = "/Users/jang-wonjun/Desktop/DMISLab/FG_Level_DeToxicity/detoxicity_model/Preprocessed_data/") -> dict:
    """
    전처리 진행 상태를 확인하는 함수
    
    Args:
        save_dir: 저장 디렉토리 경로
        
    Returns:
        각 단계별 완료 상태를 담은 딕셔너리
    """
    status = {
        'fg_extraction': False,
        'similarity_analysis': False,
        'pair_metadata': False,
        'all_complete': False
    }
    
    fg_file = os.path.join(save_dir, "merged_df_with_FG.parquet")
    cross_sim_file = os.path.join(save_dir, "merged_df_cross_sim_df.parquet")
    pairs_file = os.path.join(save_dir, "merged_df_pairs_df.parquet")
    stats_file = os.path.join(save_dir, "merged_df_stats.pkl")
    pair_info_file = os.path.join(save_dir, "merged_df_with_pair_info.parquet")
    
    # 각 단계별 완료 상태 확인
    if os.path.exists(fg_file):
        status['fg_extraction'] = True
    
    if all(os.path.exists(f) for f in [cross_sim_file, pairs_file, stats_file]):
        status['similarity_analysis'] = True
    
    if os.path.exists(pair_info_file):
        status['pair_metadata'] = True
    
    if all([status['fg_extraction'], status['similarity_analysis'], status['pair_metadata']]):
        status['all_complete'] = True
    
    return status


def use_existing_fg_file_example():
    """
    기존 FG 정보 파일을 사용하는 예시 함수
    """
    print("📋 기존 FG 정보 파일 사용 예시:")
    print("=" * 50)
    
    # 예시 1: CSV 파일 사용
    print("1. CSV 파일 사용:")
    print("   existing_fg_file = '/path/to/your/fg_data.csv'")
    print("   merged_df_with_FG, ... = preprocessing_main(")
    print("       merged_df,")
    print("       existing_fg_file=existing_fg_file")
    print("   )")
    print()
    
    # 예시 2: Parquet 파일 사용
    print("2. Parquet 파일 사용:")
    print("   existing_fg_file = '/path/to/your/fg_data.parquet'")
    print("   merged_df_with_FG, ... = preprocessing_main(")
    print("       merged_df,")
    print("       existing_fg_file=existing_fg_file")
    print("   )")
    print()
    
    # 예시 3: 직접 매핑 함수 사용
    print("3. 직접 매핑 함수 사용:")
    print("   merged_df_with_FG = map_existing_fg_data(")
    print("       merged_df,")
    print("       existing_fg_file='/path/to/your/fg_data.csv'")
    print("   )")
    print()
    
    print("📝 필수 컬럼:")
    print("   - X: SMILES 문자열 (기본값)")
    print("   - fg_names: Functional Group 이름 리스트")
    print("   - fg_counts: Functional Group 개수 딕셔너리")
    print("   - total_fg_count: 전체 Functional Group 개수")
    print("   - fg_full: 상세 FG 정보 (atom indices 포함)")
    print()
    print("💡 지원하는 SMILES 컬럼명:")
    print("   - X (기본값), Drug, smiles, SMILES, Smiles (자동 감지)")
    print("   - 매핑 시: merged_df의 'X' 컬럼 ↔ FG 파일의 'X' 컬럼")
    print()
    print("🎯 unique_smiles_fg_extraction.csv 사용 예시:")
    print("   existing_fg_file = 'unique_smiles_fg_extraction.csv'")
    print("   # 이 파일은 X, fg_names, fg_counts, total_fg_count, fg_full 컬럼 포함")
    print("=" * 50)


def clean_corrupted_csv(csv_path: str) -> bool:
    """
    손상된 CSV 파일을 정리하는 함수
    
    Args:
        csv_path: 정리할 CSV 파일 경로
        
    Returns:
        정리 성공 여부
    """
    try:
        print(f"🧹 손상된 CSV 파일 정리 중: {csv_path}")
        
        # 백업 파일 생성
        backup_path = csv_path + ".backup"
        if os.path.exists(csv_path):
            import shutil
            shutil.copy2(csv_path, backup_path)
            print(f"📋 백업 파일 생성: {backup_path}")
        
        # CSV 파일 읽기 (오류 행 건너뛰기)
        try:
            df = pd.read_csv(csv_path, on_bad_lines='skip')
        except Exception as e:
            print(f"❌ CSV 파일 읽기 실패: {e}")
            return False
        
        # 데이터 정리
        if len(df) > 0:
            # 빈 행 제거
            df = df.dropna(subset=['Drug'])
            
            # 중복 제거
            df = df.drop_duplicates(subset=['Drug'])
            
            # 손상된 데이터 정리
            def clean_fg_data(x):
                if pd.isna(x):
                    return {} if 'fg_counts' in str(x) or 'fg_full' in str(x) else []
                
                if isinstance(x, str):
                    x = x.strip()
                    # 연속된 빈 딕셔너리 패턴 제거
                    if x.startswith('{}{}{}') or x.startswith('[][][]'):
                        return {} if 'fg_counts' in str(x) or 'fg_full' in str(x) else []
                    
                    # 빈 문자열 처리
                    if x == '' or x == 'nan':
                        return {} if 'fg_counts' in str(x) or 'fg_full' in str(x) else []
                
                return x
            
            # FG 관련 컬럼 정리
            for col in ['fg_names', 'fg_counts', 'fg_full']:
                if col in df.columns:
                    df[col] = df[col].apply(clean_fg_data)
            
            # total_fg_count 정리
            if 'total_fg_count' in df.columns:
                df['total_fg_count'] = pd.to_numeric(df['total_fg_count'], errors='coerce').fillna(0).astype(int)
        
        # 정리된 파일 저장
        df.to_csv(csv_path, index=False)
        print(f"✅ CSV 파일 정리 완료: {len(df)}개 행")
        return True
        
    except Exception as e:
        print(f"❌ CSV 파일 정리 실패: {e}")
        return False


def clear_preprocessed_data(save_dir: str = "/Users/jang-wonjun/Desktop/DMISLab/FG_Level_DeToxicity/detoxicity_model/Preprocessed_data/") -> bool:
    """
    저장된 전처리 데이터를 삭제하는 함수
    
    Args:
        save_dir: 저장 디렉토리 경로
        
    Returns:
        삭제 성공 여부
    """
    try:
        if os.path.exists(save_dir):
            import shutil
            shutil.rmtree(save_dir)
            print(f"🗑️ 전처리 데이터 삭제 완료: {save_dir}")
            return True
        else:
            print(f"📂 삭제할 디렉토리가 없습니다: {save_dir}")
            return False
    except Exception as e:
        print(f"❌ 삭제 실패: {e}")
        return False


if __name__ == "__main__":

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.warning')
    import pandas as pd

    # 전처리 상태 확인
    print("🔍 전처리 상태 확인 중...")
    status = check_preprocessing_status()
    print(f"📊 현재 상태:")
    print(f"  - FG 추출: {'✅ 완료' if status['fg_extraction'] else '❌ 미완료'}")
    print(f"  - 유사도 분석: {'✅ 완료' if status['similarity_analysis'] else '❌ 미완료'}")
    print(f"  - Pair 메타데이터: {'✅ 완료' if status['pair_metadata'] else '❌ 미완료'}")
    print(f"  - 전체 완료: {'✅ 완료' if status['all_complete'] else '❌ 미완료'}")
    
    if status['all_complete']:
        print("\n🎉 모든 전처리가 완료되었습니다!")
        print("📂 저장된 데이터를 로드합니다...")
        merged_df_with_FG, merged_df_cross_sim_df, merged_df_pairs_df, merged_df_stats, merged_df_with_pair_info = load_preprocessed_data()
    else:
        print("\n🚀 전처리를 시작합니다...")
        merged_df = pd.read_csv("/Users/jang-wonjun/Desktop/DMISLab/FG_Level_DeToxicity/detoxicity_model/Raw_data/merge_df.csv")

        '''
        merged_df.shape: (1942514, 4)
        merged_df.columns: Index(['Drug_ID', 'Y', 'X', 'Task'], dtype='object')
        '''
        
        # ✨ final_FG_dictionary.csv 사용 (완성된 FG 정보)
        final_fg_dict_file = "/Users/jang-wonjun/Desktop/DMISLab/FG_Level_DeToxicity/detoxicity_model/final_FG_dictionary.csv"
        
        # 특정 Task만 필터링 (DataFrame으로 유지)
        task_filter = merged_df['Task'].str.startswith(('toxcast_', 'tox21_', 'clintox_', 'dili_'), na=False)
        merged_df = merged_df[task_filter].copy()
        
        print(f"📊 필터링된 데이터: {len(merged_df)}개 분자")
        print(f"📋 Task 분포:")
        print(merged_df['Task'].value_counts())
        
        merged_df_with_FG, merged_df_cross_sim_df, merged_df_pairs_df, merged_df_stats, merged_df_with_pair_info = preprocessing_main(
            merged_df,
            enable_checkpoint=True,
            max_workers=8,  # CPU 코어 수에 맞게 조정
            checkpoint_path='./large_dataset_fg_extraction.pkl',
            resume_from_saved=True,  # 저장된 파일에서 재시작
            existing_fg_file=final_fg_dict_file  # 완성된 FG dictionary 파일 경로
        )

    # 결과 확인
    print("\n📊 최종 결과 확인:")
    if merged_df_with_FG is not None:
        print(f"  - FG 데이터: {len(merged_df_with_FG)}개 분자")
        print(f"  - FG가 있는 분자: {(merged_df_with_FG['total_fg_count'] > 0).sum()}개")
        print(f"  - 평균 FG 개수: {merged_df_with_FG['total_fg_count'].mean():.2f}")
    if merged_df_cross_sim_df is not None:
        print(f"  - 유사도 매트릭스: {merged_df_cross_sim_df.shape}")
    if merged_df_pairs_df is not None:
        print(f"  - 유사도 쌍: {len(merged_df_pairs_df)}개")
    if merged_df_with_pair_info is not None:
        print(f"  - Pair 메타데이터: {len(merged_df_with_pair_info)}개 쌍")

    # print(merged_df_with_FG.head(2))
    # print(merged_df_cross_sim_df.head(2))
    # print(merged_df_pairs_df.head(2))
    # print(merged_df_stats)
    # print(merged_df_with_pair_info.head(2))