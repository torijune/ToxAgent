#!/usr/bin/env python3
"""
Functional Group Level Property Extraction Module
SMILES 데이터셋에서 Functional Group Level 정보를 추출하는 모듈화된 클래스

Input: SMILES 컬럼이 포함된 pandas DataFrame
Output: FG properties가 추가된 pandas DataFrame
"""

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
try:
    from rdkit.Chem import rdMolStandardize
except ImportError:
    # RDKit 버전에 따라 import 경로가 다를 수 있음
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize
    except ImportError:
        # rdMolStandardize를 사용할 수 없는 경우 None으로 설정
        rdMolStandardize = None
        print("⚠️ Warning: rdMolStandardize not available. Some features may be limited.")
from AccFG.accfg import AccFG
import pandas as pd
import numpy as np
from collections import Counter
from tqdm.auto import tqdm
from typing import Optional, Dict, List, Any, Tuple
import logging
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp
import os
import json
import pickle
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# tqdm.pandas()  # Disabled to avoid multiprocessing pickle issues

# Constants
DEFAULT_MORGAN_RADIUS = 2
DEFAULT_FP_SIZE = 1024
DEFAULT_SIMILARITY_THRESHOLD = 0.5
SIMILARITY_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_MAX_WORKERS = min(32, (os.cpu_count() or 1) + 4)  # CPU 코어 수에 기반한 기본 워커 수
DEFAULT_CHECKPOINT_INTERVAL = 1000  # 체크포인트 저장 간격 (처리된 분자 수)


def _save_checkpoint(checkpoint_path: str, 
                    processed_count: int, 
                    total_count: int,
                    results: List[Dict[str, Any]], 
                    metadata: Dict[str, Any]) -> None:
    """
    체크포인트 저장
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        processed_count: 처리된 분자 수
        total_count: 전체 분자 수
        results: 처리된 결과들
        metadata: 메타데이터 (설정 정보 등)
    """
    checkpoint_data = {
        'processed_count': processed_count,
        'total_count': total_count,
        'results': results,
        'metadata': metadata,
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    # 디렉토리 생성
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    
    # 체크포인트 저장
    with open(checkpoint_path, 'wb') as f:
        pickle.dump(checkpoint_data, f)
    
    logger.info(f"💾 Checkpoint saved: {processed_count}/{total_count} molecules processed")


def _load_checkpoint(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    체크포인트 로드
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        
    Returns:
        체크포인트 데이터 또는 None (파일이 없는 경우)
    """
    if not os.path.exists(checkpoint_path):
        return None
    
    try:
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        logger.info(f"📂 Checkpoint loaded: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']} molecules")
        return checkpoint_data
    except Exception as e:
        logger.warning(f"⚠️ Failed to load checkpoint: {e}")
        return None


def _get_checkpoint_path(base_path: str, task_name: str) -> str:
    """
    체크포인트 파일 경로 생성
    
    Args:
        base_path: 기본 경로
        task_name: 작업 이름
        
    Returns:
        체크포인트 파일 경로
    """
    checkpoint_dir = os.path.join(base_path, '.checkpoints')
    return os.path.join(checkpoint_dir, f'{task_name}_checkpoint.pkl')


def _process_smiles_batch(smiles_batch: List[str], lite_mode: bool = False) -> List[Dict[str, Any]]:
    """
    SMILES 배치를 처리하는 워커 함수 (멀티스레딩용)
    
    Args:
        smiles_batch: 처리할 SMILES 리스트
        lite_mode: AccFG lite mode 사용 여부
        
    Returns:
        FG 속성 리스트
    """
    # 각 워커 스레드에서 독립적인 AccFG 인스턴스 생성
    afg = AccFG(lite=lite_mode, print_load_info=False)
    
    results = []
    for smiles in smiles_batch:
        try:
            # Extract FGs with atom indices
            fgs = afg.run(smiles, show_atoms=True, canonical=True)
            
            if not fgs:
                results.append({
                    'fg_names': [],
                    'fg_counts': {},
                    'total_fg_count': 0,
                    'fg_full': {}
                })
            else:
                # Calculate FG counts
                fg_counts = {name: len(atoms_list) for name, atoms_list in fgs.items()}
                
                results.append({
                    'fg_names': list(fgs.keys()),
                    'fg_counts': fg_counts,
                    'total_fg_count': sum(fg_counts.values()),
                    'fg_full': fgs
                })
        except Exception as e:
            logger.debug(f"Error processing SMILES: {smiles}, Error: {e}")
            results.append({
                'fg_names': [],
                'fg_counts': {},
                'total_fg_count': 0,
                'fg_full': {}
            })
    
    return results


def _canonicalize_smiles_batch(smiles_batch: List[str]) -> List[Optional[str]]:
    """
    SMILES 배치를 canonicalize하는 워커 함수 (멀티스레딩용)
    
    Args:
        smiles_batch: canonicalize할 SMILES 리스트
        
    Returns:
        canonicalized SMILES 리스트
    """
    results = []
    for smiles in smiles_batch:
        try:
            cano = canonicalize_smiles_largest_fragment(smiles, keep_isomeric=False)
            results.append(cano)
        except Exception:
            results.append(None)
    return results


# --- Canonicalization helper for largest organic fragment after cleanup/disconnect ---
def canonicalize_smiles_largest_fragment(smi: str, keep_isomeric: bool = False) -> str | None:
    """
    Canonicalize a SMILES string while:
      1) Performing RDKit Cleanup
      2) Disconnecting metal–organic bonds (e.g., Ca2+)
      3) Selecting the largest fragment
    Optionally preserve isomeric information.

    Returns canonical SMILES of the largest organic fragment, or None on failure.
    """
    try:
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)

        # rdMolStandardize가 사용 가능한 경우에만 사용
        if rdMolStandardize is not None:
            # 1) standard cleanup
            mol = rdMolStandardize.Cleanup(mol)

            # 2) disconnect metal–organic bonds (keeps counter-ions separate)
            mol = rdMolStandardize.MetalDisconnector().Disconnect(mol)

            # 3) (optional) uncharge — comment out if you don't want protonation changes
            # mol = rdMolStandardize.Uncharger().uncharge(mol)

            # 4) keep only the largest fragment
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        else:
            # rdMolStandardize가 없는 경우 기본 처리만 수행
            # 가장 큰 fragment만 선택 (간단한 방법)
            smiles_parts = Chem.MolToSmiles(mol).split('.')
            if len(smiles_parts) > 1:
                # 가장 긴 SMILES 부분 선택
                largest_smiles = max(smiles_parts, key=len)
                mol = Chem.MolFromSmiles(largest_smiles)
                if mol is None:
                    return None

        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=keep_isomeric)
    except Exception:
        return None


class FunctionalGroupExtractor:
    """
    Functional Group 정보를 추출하는 메인 클래스
    
    Input Requirements:
        - DataFrame with SMILES column
        - SMILES column should contain valid chemical structures
    
    Output:
        - Original DataFrame + 4 new columns:
          * fg_names: List of functional group names
          * fg_counts: Dictionary of {fg_name: count}
          * total_fg_count: Total number of functional groups
          * fg_full: Detailed FG info with atom indices
    """
    
    # Common SMILES column names
    COMMON_SMILES_COLUMNS = ['Drug', 'smiles', 'SMILES', 'Smiles', 'canonical_smiles', 'molecule']
    
    def __init__(self, lite_mode: bool = False, verbose: bool = True, max_workers: int = DEFAULT_MAX_WORKERS,
                 enable_checkpoint: bool = True, checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL):
        """
        Initialize the Functional Group Extractor
        
        Args:
            lite_mode: Use AccFG lite mode (faster, less comprehensive)
            verbose: Print progress information
            max_workers: Maximum number of worker threads for parallel processing
            enable_checkpoint: Whether to enable checkpoint saving
            checkpoint_interval: Interval for saving checkpoints (number of processed molecules)
        """
        self.lite_mode = lite_mode
        self.verbose = verbose
        self.max_workers = max_workers
        self.enable_checkpoint = enable_checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.afg = None
        self._initialize_accfg()
    
    def _initialize_accfg(self):
        """Initialize AccFG with specified settings"""
        if self.verbose:
            print(f"🔧 Initializing AccFG (lite_mode={self.lite_mode})...")
        self.afg = AccFG(lite=self.lite_mode, print_load_info=self.verbose)
    
    @classmethod
    def detect_smiles_column(cls, df: pd.DataFrame) -> Optional[str]:
        """
        DataFrame에서 SMILES 컬럼 자동 감지
        
        Args:
            df: Input DataFrame
            
        Returns:
            감지된 SMILES 컬럼명 또는 None
        """
        for col_name in cls.COMMON_SMILES_COLUMNS:
            if col_name in df.columns:
                return col_name
        return None
    
    @staticmethod
    def canonicalize_smiles(smiles: str) -> Optional[str]:
        """
        Convert SMILES to canonical form
        
        Args:
            smiles: SMILES string
            
        Returns:
            Canonical SMILES string or None if invalid
        """
        try:
            return canonicalize_smiles_largest_fragment(smiles, keep_isomeric=False)
        except Exception:
            return None
    

##### Single SMILES에 대해서 Functional Group Properties 추출 #####
    def extract_single_fg_properties(self, smiles: str) -> Dict[str, Any]:
        """
        Extract FG properties from a single SMILES string
        
        Args:
            smiles: SMILES string
            
        Returns:
            Dictionary containing FG properties:
            - fg_names: List of FG names
            - fg_counts: Dict of {fg_name: count}
            - total_fg_count: Total FG count
            - fg_full: Detailed FG info with atom indices
        """
        try:
            # Extract FGs with atom indices
            fgs = self.afg.run(smiles, show_atoms=True, canonical=True)
            
            if not fgs:
                return {
                    'fg_names': [],
                    'fg_counts': {},
                    'total_fg_count': 0,
                    'fg_full': {}
                }
            
            # Calculate FG counts
            fg_counts = {name: len(atoms_list) for name, atoms_list in fgs.items()}
            
            return {
                'fg_names': list(fgs.keys()),
                'fg_counts': fg_counts,
                'total_fg_count': sum(fg_counts.values()),
                'fg_full': fgs
            }
        except Exception as e:
            if self.verbose:
                print(f"Error processing SMILES: {smiles}, Error: {e}")
            return {
                'fg_names': [],
                'fg_counts': {},
                'total_fg_count': 0,
                'fg_full': {}
            }
    
##### DataFrame에 대해서 Functional Group Properties 추출 실행 부분 #####
    def process_dataframe(self, 
                         df: pd.DataFrame, 
                         smiles_column: Optional[str] = None,
                         canonicalize: bool = True,
                         use_multithreading: bool = True,
                         checkpoint_path: Optional[str] = None,
                         resume_from_checkpoint: bool = True) -> pd.DataFrame:
        """
        Process entire DataFrame to extract FG properties
        
        Args:
            df: Input DataFrame with SMILES column
            smiles_column: Name of SMILES column (auto-detect if None)
            canonicalize: Whether to canonicalize SMILES first
            use_multithreading: Whether to use multithreading for faster processing
            checkpoint_path: Path for checkpoint file (auto-generated if None)
            resume_from_checkpoint: Whether to resume from existing checkpoint
            
        Returns:
            DataFrame with original columns + 4 new FG columns
        
        This canonicalization keeps the largest organic fragment after disconnecting metals/salts.
        """
        if self.verbose:
            print(f"📂 Processing DataFrame with {len(df)} molecules")
        
        # Auto-detect SMILES column if not specified
        if smiles_column is None:
            smiles_column = self.detect_smiles_column(df)
            if smiles_column is None:
                raise ValueError(f"Could not auto-detect SMILES column. Available columns: {df.columns.tolist()}")
            if self.verbose:
                print(f"🔍 Auto-detected SMILES column: '{smiles_column}'")
        
        # Validate input
        if smiles_column not in df.columns:
            raise ValueError(f"Column '{smiles_column}' not found. Available columns: {df.columns.tolist()}")
        
        # Create working copy
        result_df = df.copy()
        
        # 체크포인트 설정
        if checkpoint_path is None and self.enable_checkpoint:
            checkpoint_path = _get_checkpoint_path(os.getcwd(), f"fg_extraction_{len(df)}")
        
        # 체크포인트에서 이어서 처리할지 확인
        checkpoint_data = None
        if resume_from_checkpoint and checkpoint_path and self.enable_checkpoint:
            checkpoint_data = _load_checkpoint(checkpoint_path)
            if checkpoint_data and self.verbose:
                print(f"🔄 Resuming from checkpoint: {checkpoint_data['processed_count']}/{checkpoint_data['total_count']} molecules")
        
        # Canonicalize SMILES if requested
        if canonicalize:
            if self.verbose:
                print("🧪 Canonicalizing SMILES...")
            
            smiles_list = result_df[smiles_column].tolist()
            
            if use_multithreading and len(smiles_list) > 100:  # 멀티스레딩은 큰 데이터셋에서만 사용
                if self.verbose:
                    print(f"🚀 Using multithreading with {self.max_workers} workers...")
                
                # SMILES를 배치로 나누기
                batch_size = max(1, len(smiles_list) // (self.max_workers * 4))
                batches = [smiles_list[i:i + batch_size] for i in range(0, len(smiles_list), batch_size)]
                
                canonical_smiles = []
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 진행률 표시를 위한 tqdm과 함께 실행
                    if self.verbose:
                        from tqdm import tqdm
                        futures = [executor.submit(_canonicalize_smiles_batch, batch) for batch in batches]
                        for future in tqdm(concurrent.futures.as_completed(futures), 
                                         total=len(futures), desc="Canonicalizing SMILES"):
                            canonical_smiles.extend(future.result())
                    else:
                        futures = [executor.submit(_canonicalize_smiles_batch, batch) for batch in batches]
                        for future in concurrent.futures.as_completed(futures):
                            canonical_smiles.extend(future.result())
            else:
                # 단일 스레드 처리
                canonical_smiles = []
                if self.verbose:
                    from tqdm import tqdm
                    for smiles in tqdm(smiles_list, desc="Canonicalizing SMILES"):
                        canonical_smiles.append(self.canonicalize_smiles(smiles))
                else:
                    for smiles in smiles_list:
                        canonical_smiles.append(self.canonicalize_smiles(smiles))
            
            result_df['canonical_smiles'] = canonical_smiles
            working_smiles = 'canonical_smiles'
        else:
            working_smiles = smiles_column
        
        # Remove invalid SMILES
        initial_count = len(result_df)
        result_df = result_df.dropna(subset=[working_smiles])
        removed_count = initial_count - len(result_df)
        if removed_count > 0 and self.verbose:
            print(f"⚠️  Removed {removed_count} invalid SMILES")
        
        # Extract FG properties
        if self.verbose:
            print("🔬 Extracting Functional Group properties...")
        
        smiles_list = result_df[working_smiles].tolist()
        
        # 체크포인트에서 이어서 처리
        start_idx = 0
        fg_properties = []
        
        if checkpoint_data and resume_from_checkpoint:
            # 체크포인트에서 결과 로드
            fg_properties = checkpoint_data['results']
            start_idx = checkpoint_data['processed_count']
            
            if self.verbose:
                print(f"📂 Loaded {len(fg_properties)} results from checkpoint")
                
            # SMILES 리스트가 변경되었는지 확인
            if len(fg_properties) != len(smiles_list):
                if self.verbose:
                    print("⚠️ Checkpoint data length doesn't match current data. Starting fresh.")
                fg_properties = []
                start_idx = 0
        
        if use_multithreading and len(smiles_list) > 50:  # 멀티스레딩은 중간 이상 데이터셋에서만 사용
            if self.verbose:
                print(f"🚀 Using multithreading with {self.max_workers} workers...")
            
            # 처리할 SMILES 리스트 (이미 처리된 것은 제외)
            remaining_smiles = smiles_list[start_idx:]
            if not remaining_smiles:
                if self.verbose:
                    print("✅ All molecules already processed from checkpoint")
            else:
                # SMILES를 배치로 나누기
                batch_size = max(1, len(remaining_smiles) // (self.max_workers * 2))
                batches = [remaining_smiles[i:i + batch_size] for i in range(0, len(remaining_smiles), batch_size)]
                
                processed_count = start_idx
                total_count = len(smiles_list)
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # 진행률 표시를 위한 tqdm과 함께 실행
                    if self.verbose:
                        from tqdm import tqdm
                        # 전체 분자 수로 진행률 표시
                        pbar = tqdm(total=total_count, initial=start_idx, desc="Processing SMILES", unit="molecules", 
                                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} molecules [{elapsed}<{remaining}, {rate_fmt}]')
                        futures = [executor.submit(_process_smiles_batch, batch, self.lite_mode) for batch in batches]
                        for future in concurrent.futures.as_completed(futures):
                            batch_results = future.result()
                            fg_properties.extend(batch_results)
                            processed_count += len(batch_results)
                            
                            # 진행률 업데이트 (개별 분자 단위)
                            pbar.update(len(batch_results))
                            
                            # 체크포인트 저장
                            if self.enable_checkpoint and checkpoint_path and processed_count % self.checkpoint_interval == 0:
                                metadata = {
                                    'lite_mode': self.lite_mode,
                                    'smiles_column': smiles_column,
                                    'canonicalize': canonicalize,
                                    'use_multithreading': use_multithreading
                                }
                                _save_checkpoint(checkpoint_path, processed_count, total_count, fg_properties, metadata)
                        pbar.close()
                    else:
                        futures = [executor.submit(_process_smiles_batch, batch, self.lite_mode) for batch in batches]
                        for future in concurrent.futures.as_completed(futures):
                            batch_results = future.result()
                            fg_properties.extend(batch_results)
                            processed_count += len(batch_results)
                            
                            # 체크포인트 저장
                            if self.enable_checkpoint and checkpoint_path and processed_count % self.checkpoint_interval == 0:
                                metadata = {
                                    'lite_mode': self.lite_mode,
                                    'smiles_column': smiles_column,
                                    'canonicalize': canonicalize,
                                    'use_multithreading': use_multithreading
                                }
                                _save_checkpoint(checkpoint_path, processed_count, total_count, fg_properties, metadata)
        else:
            # 단일 스레드 처리
            remaining_smiles = smiles_list[start_idx:]
            if not remaining_smiles:
                if self.verbose:
                    print("✅ All molecules already processed from checkpoint")
            else:
                processed_count = start_idx
                total_count = len(smiles_list)
                
                if self.verbose:
                    from tqdm import tqdm
                    pbar = tqdm(total=total_count, initial=start_idx, desc="Processing SMILES", unit="molecules",
                              bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} molecules [{elapsed}<{remaining}, {rate_fmt}]')
                    for i, smiles in enumerate(remaining_smiles):
                        fg_properties.append(self.extract_single_fg_properties(smiles))
                        processed_count += 1
                        pbar.update(1)
                        
                        # 체크포인트 저장
                        if self.enable_checkpoint and checkpoint_path and processed_count % self.checkpoint_interval == 0:
                            metadata = {
                                'lite_mode': self.lite_mode,
                                'smiles_column': smiles_column,
                                'canonicalize': canonicalize,
                                'use_multithreading': use_multithreading
                            }
                            _save_checkpoint(checkpoint_path, processed_count, total_count, fg_properties, metadata)
                    pbar.close()
                else:
                    for smiles in remaining_smiles:
                        fg_properties.append(self.extract_single_fg_properties(smiles))
                        processed_count += 1
                        
                        # 체크포인트 저장
                        if self.enable_checkpoint and checkpoint_path and processed_count % self.checkpoint_interval == 0:
                            metadata = {
                                'lite_mode': self.lite_mode,
                                'smiles_column': smiles_column,
                                'canonicalize': canonicalize,
                                'use_multithreading': use_multithreading
                            }
                            _save_checkpoint(checkpoint_path, processed_count, total_count, fg_properties, metadata)
        
        # Add FG columns
        result_df['fg_names'] = [x['fg_names'] for x in fg_properties]
        result_df['fg_counts'] = [x['fg_counts'] for x in fg_properties]
        result_df['total_fg_count'] = [x['total_fg_count'] for x in fg_properties]
        result_df['fg_full'] = [x['fg_full'] for x in fg_properties]
        
        # 최종 체크포인트 저장 (완료 표시)
        if self.enable_checkpoint and checkpoint_path:
            metadata = {
                'lite_mode': self.lite_mode,
                'smiles_column': smiles_column,
                'canonicalize': canonicalize,
                'use_multithreading': use_multithreading,
                'completed': True
            }
            _save_checkpoint(checkpoint_path, len(fg_properties), len(smiles_list), fg_properties, metadata)
            
            if self.verbose:
                print("✅ Final checkpoint saved - processing completed")
        
        # Print statistics
        if self.verbose:
            self._print_statistics(result_df)
        
        ##### reuslt_df = Single SMILES에 대한 FG 정보들이 있는 DataFrame #####
        return result_df
    
    def _print_statistics(self, df: pd.DataFrame):
        """Print processing statistics"""
        print("\n📊 Dataset Statistics:")
        print(f"  - Total molecules processed: {len(df)}")
        print(f"  - Molecules with FGs: {(df['total_fg_count'] > 0).sum()}")
        print(f"  - Molecules without FGs: {(df['total_fg_count'] == 0).sum()}")
        print(f"  - Average FGs per molecule: {df['total_fg_count'].mean():.2f}")
        
        # Most common FGs
        all_fg_counts = Counter()
        for fg_count_dict in df['fg_counts']:
            all_fg_counts.update(fg_count_dict)
        
        print("\n🏆 Top 10 Most Common Functional Groups:")
        for fg_name, count in all_fg_counts.most_common(10):
            print(f"  {fg_name}: {count}")
    
    ##### DataFrame에 대해서 Functional Group Feature Matrix 생성 부분 #####
    def create_fg_feature_matrix(self, 
                                df: pd.DataFrame, 
                                smiles_column: str = 'smiles') -> pd.DataFrame:
        """
        Create FG feature matrix for machine learning
        
        Args:
            df: DataFrame with FG properties
            smiles_column: SMILES column name
            
        Returns:
            DataFrame with SMILES + FG count vectors
        """
        if self.verbose:
            print("🔄 Creating FG feature matrix...")
        
        # Collect all unique FG names
        all_fg_names = set()
        for fg_names in df['fg_names']:
            all_fg_names.update(fg_names)
        
        all_fg_names = sorted(list(all_fg_names))
        if self.verbose:
            print(f"  - Total unique FGs: {len(all_fg_names)}")
        
        # Create FG count vectors
        fg_matrix = []
        for fg_counts in tqdm(df['fg_counts'], desc="Creating feature vectors"):
            fg_vector = [fg_counts.get(fg_name, 0) for fg_name in all_fg_names]
            fg_matrix.append(fg_vector)
        
        # Create DataFrame
        fg_feature_df = pd.DataFrame(fg_matrix, columns=all_fg_names)
        
        # Combine with SMILES if available
        if smiles_column in df.columns:
            result_df = pd.concat([
                df[[smiles_column]].reset_index(drop=True), 
                fg_feature_df
            ], axis=1)
        else:
            result_df = fg_feature_df
        
        ##### result_df = Functional Group Feature Matrix #####
        return result_df
    
    ##### DataFrame을 CSV 또는 Parquet 파일로 저장 부분 #####
    def save_results(self, 
                    df: pd.DataFrame, 
                    output_path: str,
                    save_parquet: bool = True):
        """
        Save results to file
        
        Args:
            df: DataFrame to save
            output_path: Output file path
            save_parquet: Also save as Parquet format
        """
        if self.verbose:
            print(f"💾 Saving results to: {output_path}")
        
        # Save CSV
        df.to_csv(output_path, index=False)
        
        # Save Parquet if requested
        if save_parquet:
            parquet_path = output_path.replace('.csv', '.parquet')
            try:
                df.to_parquet(parquet_path, index=False)
                if self.verbose:
                    print(f"💾 Also saved as: {parquet_path}")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️ Parquet save failed: {e}")

##### DataFrame에 대해서 Molecular Similarity 분석 부분 #####
class MolecularSimilarityAnalyzer:
    """
    분자 유사도 분석을 위한 클래스
    Morgan Fingerprint 기반 Tanimoto 유사도 계산
    """
    
    def __init__(self, radius: int = DEFAULT_MORGAN_RADIUS, fp_size: int = DEFAULT_FP_SIZE):
        """
        Initialize Molecular Similarity Analyzer
        
        Args:
            radius: Morgan Fingerprint radius
            fp_size: Fingerprint size
        """
        self.radius = radius
        self.fp_size = fp_size
        self.fpgen = AllChem.GetMorganGenerator(radius=radius, fpSize=fp_size)
    
    ##### Tanimoto Similarity 계산을 위한 Morgan Fingerprint 생성 부분 #####
    def _generate_fingerprints(self, 
                              df: pd.DataFrame, 
                              id_column: str, 
                              smiles_column: str,
                              desc: str = "Generating fingerprints") -> Tuple[List, List, List]:
        """
        DataFrame에서 fingerprint 생성
        
        Args:
            df: Input DataFrame
            id_column: ID 컬럼 이름
            smiles_column: SMILES 컬럼 이름
            desc: Progress bar description
            
        Returns:
            (fingerprints, valid_indices, ids) 튜플
        """
        ids = df[id_column].tolist()
        smiles_list = df[smiles_column].tolist()
        
        fingerprints = []
        valid_indices = []
        
        for i, smi in enumerate(tqdm(smiles_list, desc=desc)):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    fp = self.fpgen.GetFingerprint(mol)
                    fingerprints.append(fp)
                    valid_indices.append(i)
            except Exception as e:
                logger.debug(f"Failed to generate fingerprint for SMILES '{smi}': {e}")
                continue
        
        valid_ids = [ids[i] for i in valid_indices]
        return fingerprints, valid_indices, valid_ids
    
    @staticmethod
    def _handle_duplicate_ids(ids: List) -> List:
        """
        중복 ID 처리 - 중복된 경우 suffix 추가
        
        Args:
            ids: ID 리스트
            
        Returns:
            중복 처리된 ID 리스트
        """
        if len(ids) != len(set(ids)):
            duplicate_count = len(ids) - len(set(ids))
            logger.warning(f"Found {duplicate_count} duplicate IDs, adding suffixes")
            
            id_counts = {}
            unique_ids = []
            for id_val in ids:
                if id_val in id_counts:
                    id_counts[id_val] += 1
                    unique_ids.append(f"{id_val}_{id_counts[id_val]}")
                else:
                    id_counts[id_val] = 0
                    unique_ids.append(id_val)
            return unique_ids
        return ids
    
    ##### 두 DataFrame 간의 교차 유사도 매트릭스 생성 부분 #####
    ##### 주로 Non-toxic과 Toxic 분자들 간의 유사도 분석에 사용 #####
    def build_cross_similarity_matrix(self, 
                                     df1: pd.DataFrame, 
                                     id_column1: str, 
                                     smiles_column1: str,
                                     df2: pd.DataFrame, 
                                     id_column2: str, 
                                     smiles_column2: str) -> pd.DataFrame:
        """
        두 DataFrame 간의 교차 유사도 매트릭스 생성
        
        Args:
            df1: 첫 번째 DataFrame
            id_column1: 첫 번째 DataFrame의 ID 컬럼
            smiles_column1: 첫 번째 DataFrame의 SMILES 컬럼
            df2: 두 번째 DataFrame
            id_column2: 두 번째 DataFrame의 ID 컬럼
            smiles_column2: 두 번째 DataFrame의 SMILES 컬럼
            
        Returns:
            교차 유사도 매트릭스 DataFrame
        """
        logger.info(f"Processing {len(df1)} molecules from df1 and {len(df2)} molecules from df2")
        
        # Generate fingerprints for both DataFrames
        fps1, _, ids1 = self._generate_fingerprints(df1, id_column1, smiles_column1, "df1 fingerprints")
        fps2, _, ids2 = self._generate_fingerprints(df2, id_column2, smiles_column2, "df2 fingerprints")
        
        if not fps1 or not fps2:
            logger.warning("No valid fingerprints generated for one or both DataFrames")
            return pd.DataFrame()
        
        # Handle duplicate IDs
        ids1 = self._handle_duplicate_ids(ids1)
        ids2 = self._handle_duplicate_ids(ids2)
        
        # Compute cross-similarity matrix
        n1, n2 = len(fps1), len(fps2)
        cross_sim_matrix = np.zeros((n1, n2), dtype=np.float32)
        
        logger.info(f"Computing cross-similarity matrix ({n1} x {n2})")
        for i in tqdm(range(n1), desc="Computing cross-similarities"):
            sims = DataStructs.BulkTanimotoSimilarity(fps1[i], fps2)
            cross_sim_matrix[i, :] = np.array(sims, dtype=np.float32)
        
        # Convert to DataFrame
        cross_sim_df = pd.DataFrame(
            cross_sim_matrix,
            index=ids1,
            columns=ids2
        )
        
        logger.info(f"✅ Cross-similarity matrix created: {cross_sim_df.shape}")
        ##### 일단은 유사도 matrix만 생성 및 반환 #####
        return cross_sim_df
    
    @staticmethod
    ##### 높은 유사도를 가진 분자 쌍 추출 부분 #####
    ##### 주로 Non-toxic과 Toxic 분자들 간의 유사도에서 Functional Group Level에서 분석할 정도로 유사한 쌍들을 추출 #####
    def extract_high_similarity_pairs(cross_sim_df: pd.DataFrame,
                                     min_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                                     max_threshold: float = 0.8,
                                     threshold: Optional[float] = None) -> pd.DataFrame:
        """
        높은 유사도를 가진 분자 쌍 추출 (범위 필터링 지원)
        
        Args:
            cross_sim_df: 교차 유사도 매트릭스
            min_threshold: 유사도 최소 임계값 (포함)
            max_threshold: 유사도 최대 임계값 (포함)
            threshold: 하위 호환용 단일 임계값 (> threshold). 제공 시 min_threshold로 사용되고 max_threshold는 1.0으로 설정
            
        Returns:
            높은 유사도 쌍들의 DataFrame (similarity 내림차순 정렬)
        """
        # Backward compatibility: threshold가 지정되면 범위 대신 단일 임계값 모드로 동작
        if threshold is not None:
            min_threshold = threshold
            max_threshold = 0.8
        
        # 유효 범위로 clamp 및 검증
        min_threshold = max(0.0, float(min_threshold))
        max_threshold = min(0.8, float(max_threshold))
        if min_threshold > max_threshold:
            raise ValueError(f"min_threshold ({min_threshold}) cannot be greater than max_threshold ({max_threshold})")
        
        high_sim_pairs = []
        
        for i in range(len(cross_sim_df)):
            for j in range(len(cross_sim_df.columns)):
                sim = cross_sim_df.iloc[i, j]
                if (sim >= min_threshold) and (sim <= max_threshold):
                    high_sim_pairs.append({
                        'molecule1_id': cross_sim_df.index[i],
                        'molecule2_id': cross_sim_df.columns[j],
                        'similarity': sim
                    })
        
        high_sim_pairs_df = pd.DataFrame(high_sim_pairs)
        
        if not high_sim_pairs_df.empty:
            high_sim_pairs_df = high_sim_pairs_df.sort_values('similarity', ascending=False)
        
        ##### high_sim_pairs_df = 높은 유사도를 가진 분자 쌍들의 DataFrame #####
        return high_sim_pairs_df
    
    @staticmethod
    def get_similarity_statistics(cross_sim_df: pd.DataFrame, 
                                 thresholds: List[float] = None) -> Dict[str, Any]:
        """
        유사도 통계 계산
        
        Args:
            cross_sim_df: 교차 유사도 매트릭스
            thresholds: 분석할 임계값 리스트
            
        Returns:
            통계 정보 딕셔너리
        """
        if thresholds is None:
            thresholds = SIMILARITY_THRESHOLDS
        
        all_similarities = cross_sim_df.values.flatten()
        
        stats = {
            'total_comparisons': len(all_similarities),
            'mean_similarity': float(np.mean(all_similarities)),
            'median_similarity': float(np.median(all_similarities)),
            'std_similarity': float(np.std(all_similarities)),
            'max_similarity': float(np.max(all_similarities)),
            'min_similarity': float(np.min(all_similarities)),
            'threshold_counts': {}
        }
        
        # Count pairs above each threshold
        for threshold in thresholds:
            count = (cross_sim_df > threshold).sum().sum()
            stats['threshold_counts'][threshold] = int(count)
        
        # Find max similarity pair
        max_idx = cross_sim_df.stack().idxmax()
        stats['max_similarity_pair'] = {
            'molecule1': max_idx[0],
            'molecule2': max_idx[1],
            'similarity': float(cross_sim_df.loc[max_idx])
        }
        
        return stats

######################### 실행 함수 부분 #########################

##### FG 얻는 실행 함수 #####
def process_dataset_with_fg(df: pd.DataFrame, 
                           smiles_column: Optional[str] = None,
                           lite_mode: bool = False,
                           canonicalize: bool = True,
                           verbose: bool = True,
                           use_multithreading: bool = True,
                           max_workers: int = DEFAULT_MAX_WORKERS,
                           enable_checkpoint: bool = True,
                           checkpoint_path: Optional[str] = None,
                           resume_from_checkpoint: bool = True) -> pd.DataFrame:
    """
    Convenience function to process a DataFrame with FG extraction
    
    Args:
        df: Input DataFrame with SMILES column
        smiles_column: Name of SMILES column (auto-detect if None)
        lite_mode: Use AccFG lite mode
        canonicalize: Canonicalize SMILES first
        verbose: Print progress information
        use_multithreading: Whether to use multithreading for faster processing
        max_workers: Maximum number of worker threads
        enable_checkpoint: Whether to enable checkpoint saving
        checkpoint_path: Path for checkpoint file (auto-generated if None)
        resume_from_checkpoint: Whether to resume from existing checkpoint
        
    Returns:
        DataFrame with FG properties added
    """
    extractor = FunctionalGroupExtractor(
        lite_mode=lite_mode, 
        verbose=verbose, 
        max_workers=max_workers,
        enable_checkpoint=enable_checkpoint
    )
    return extractor.process_dataframe(
        df, smiles_column, canonicalize, use_multithreading, 
        checkpoint_path, resume_from_checkpoint
    )


##### DataFrame에 대해서 Molecular Similarity 분석 실행 부분 #####
def build_cross_similarity_matrix_from_dfs(df1: pd.DataFrame, 
                                           id_column1: str, 
                                           smiles_column1: str, 
                                           df2: pd.DataFrame, 
                                           id_column2: str, 
                                           smiles_column2: str, 
                                           radius: int = DEFAULT_MORGAN_RADIUS, 
                                           fpSize: int = DEFAULT_FP_SIZE) -> pd.DataFrame:
    """
    두 개의 DataFrame에 있는 분자들 간의 교차 유사도 매트릭스를 생성합니다.
    (Deprecated: MolecularSimilarityAnalyzer 클래스 사용을 권장합니다)
    
    Args:
        df1: 첫 번째 DataFrame
        id_column1: 첫 번째 DataFrame의 ID 컬럼 이름
        smiles_column1: 첫 번째 DataFrame의 SMILES 컬럼 이름
        df2: 두 번째 DataFrame
        id_column2: 두 번째 DataFrame의 ID 컬럼 이름
        smiles_column2: 두 번째 DataFrame의 SMILES 컬럼 이름
        radius: Morgan Fingerprint radius (기본값: 2)
        fpSize: Fingerprint 크기 (기본값: 2048)
    
    Returns:
        교차 유사도 DataFrame
    """
    logger.warning("build_cross_similarity_matrix_from_dfs is deprecated. Use MolecularSimilarityAnalyzer instead.")
    
    analyzer = MolecularSimilarityAnalyzer(radius=radius, fp_size=fpSize)
    return analyzer.build_cross_similarity_matrix(
        df1, id_column1, smiles_column1,
        df2, id_column2, smiles_column2
    )

##### Non-toxic과 Toxic 분자들 간의 교차 유사도 분석 실행 부분 #####
def analyze_cross_similarity(non_toxic_df: pd.DataFrame, 
                           toxic_df: pd.DataFrame,
                           threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                           id_column: str = 'Drug_ID',
                           smiles_column: str = 'Drug') -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Non-toxic과 Toxic 분자들 간의 교차 유사도 분석
    
    Args:
        non_toxic_df: Non-toxic 분자들의 DataFrame
        toxic_df: Toxic 분자들의 DataFrame
        threshold: 높은 유사도로 판단할 임계값 (기본값: 0.7)
        id_column: ID 컬럼 이름 (기본값: 'Drug_ID')
        smiles_column: SMILES 컬럼 이름 (기본값: 'Drug')
    
    Returns:
        (cross_sim_df, high_sim_pairs_df, statistics) 튜플
        - cross_sim_df: 교차 유사도 매트릭스
        - high_sim_pairs_df: 높은 유사도를 가진 쌍들의 DataFrame
        - statistics: 통계 정보 딕셔너리
    """
    logger.info("Analyzing cross-similarity between non-toxic and toxic molecules")
    
    # 유사도 분석기 초기화
    analyzer = MolecularSimilarityAnalyzer()
    
    # 교차 유사도 매트릭스 생성
    cross_sim_df = analyzer.build_cross_similarity_matrix(
        df1=non_toxic_df, 
        id_column1=id_column, 
        smiles_column1=smiles_column,
        df2=toxic_df, 
        id_column2=id_column, 
        smiles_column2=smiles_column
    )
    
    if cross_sim_df.empty:
        logger.warning("Empty similarity matrix returned")
        return cross_sim_df, pd.DataFrame(), {}
    
    # 통계 계산
    statistics = analyzer.get_similarity_statistics(cross_sim_df)
    
    # 높은 유사도 쌍 추출
    high_sim_pairs_df = analyzer.extract_high_similarity_pairs(cross_sim_df, threshold)
    
    # 컬럼명 변경 (특정 도메인에 맞게)
    if not high_sim_pairs_df.empty:
        high_sim_pairs_df = high_sim_pairs_df.rename(columns={
            'molecule1_id': 'non_toxic_drug',
            'molecule2_id': 'toxic_drug'
        })
    
    # 결과 요약 출력
    logger.info(f"\n{'='*60}")
    logger.info("Cross-Similarity Analysis Results")
    logger.info(f"{'='*60}")
    logger.info(f"Total comparisons: {statistics.get('total_comparisons', 0):,}")
    logger.info(f"Mean similarity: {statistics.get('mean_similarity', 0):.4f}")
    logger.info(f"Max similarity: {statistics.get('max_similarity', 0):.4f}")
    logger.info(f"\nPairs above threshold ({threshold}):")
    logger.info(f"  Count: {len(high_sim_pairs_df)}")
    
    if 'threshold_counts' in statistics:
        logger.info(f"\nPairs above various thresholds:")
        for thresh, count in sorted(statistics['threshold_counts'].items()):
            logger.info(f"  > {thresh}: {count:,}")
    
    logger.info(f"{'='*60}\n")
    
    return cross_sim_df, high_sim_pairs_df, statistics


# ##### 단일 DataFrame에서 Y로 분리하여 자동 분석하는 간단 헬퍼 #####
def analyze_similarity_with_checkpoint(df: pd.DataFrame,
                                       label_column: str = 'Y',
                                       non_toxic_label: int = 0,
                                       toxic_label: int = 1,
                                       id_column: str = 'Drug_ID',
                                       smiles_column: str = 'X',
                                       min_threshold: float = 0.5,
                                       max_threshold: float = 0.8,
                                       checkpoint_dir: str = "./checkpoints",
                                       batch_size: int = 100) -> pd.DataFrame:
    """
    유사도 계산을 batch 단위로 나누고, 주기적으로 CSV에 저장합니다.
    중간에 중단돼도 이어서 재시작할 수 있습니다.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    # 라벨별 분리
    non_toxic_df = df[df[label_column] == non_toxic_label].copy()
    toxic_df = df[df[label_column] == toxic_label].copy()

    analyzer = MolecularSimilarityAnalyzer()

    results = []
    total_batches = (len(non_toxic_df) + batch_size - 1) // batch_size

    for i in tqdm(range(total_batches), desc="Cross Similarity Batches"):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(non_toxic_df))

        non_toxic_batch = non_toxic_df.iloc[start_idx:end_idx]

        # --- 실제 유사도 계산 ---
        cross_sim_df = analyzer.build_cross_similarity_matrix(
            df1=non_toxic_batch,
            id_column1=id_column,
            smiles_column1=smiles_column,
            df2=toxic_df,
            id_column2=id_column,
            smiles_column2=smiles_column
        )

        # --- 체크포인트 저장 ---
        checkpoint_path = os.path.join(checkpoint_dir, f"cross_sim_batch_{i}.csv")
        cross_sim_df.to_csv(checkpoint_path, index=False)

        results.append(cross_sim_df)

    # --- 전체 병합 ---
    full_cross_sim_df = pd.concat(results, ignore_index=True)

    # --- 통계 및 필터링 ---
    stats = analyzer.get_similarity_statistics(full_cross_sim_df)
    high_sim_pairs_df = analyzer.extract_high_similarity_pairs(
        full_cross_sim_df,
        min_threshold=min_threshold,
        max_threshold=max_threshold
    )

    return full_cross_sim_df, high_sim_pairs_df, stats


##### 유사도 pair와 FG 메타 정보를 병합하여 pair 메타데이터 DataFrame 생성 #####
def build_pair_metadata_dataframe(
    df_with_fg: pd.DataFrame,
    pairs_df: pd.DataFrame,
    id_column: str = 'Drug_ID',
    smiles_column: str = 'Drug',
    label_column: str = 'Y'
) -> pd.DataFrame:
    """
    df_with_fg (각 Drug의 FG 메타정보 포함)와 pairs_df (non_toxic_drug, toxic_drug, similarity)를 합쳐
    요청한 컬럼 구조의 pair 메타데이터 DataFrame을 생성합니다.

    Args:
        df_with_fg: 각 분자의 FG 정보가 포함된 DataFrame
                    (필수 컬럼: id_column, smiles_column, label_column, fg_names, fg_counts, total_fg_count, fg_full)
        pairs_df: 유사도 기반 pair DataFrame (필수 컬럼: non_toxic_drug, toxic_drug, similarity)
        id_column: 분자 ID 컬럼명 (기본 'Drug_ID')
        smiles_column: SMILES 컬럼명 (기본 'Drug')
        label_column: 라벨 컬럼명 (기본 'Y')

    Returns:
        요청된 컬럼을 갖는 pair 메타데이터 DataFrame
    """
    required_fg_cols = [id_column, smiles_column, label_column, 'fg_names', 'fg_counts', 'total_fg_count', 'fg_full']
    missing_fg_cols = [c for c in required_fg_cols if c not in df_with_fg.columns]
    if missing_fg_cols:
        raise ValueError(f"df_with_fg is missing columns: {missing_fg_cols}. Available: {df_with_fg.columns.tolist()}")

    required_pair_cols = ['non_toxic_drug', 'toxic_drug', 'similarity']
    missing_pair_cols = [c for c in required_pair_cols if c not in pairs_df.columns]
    if missing_pair_cols:
        raise ValueError(f"pairs_df is missing columns: {missing_pair_cols}. Available: {pairs_df.columns.tolist()}")

    # Toxic side metadata
    toxic_meta = df_with_fg[required_fg_cols].copy()
    toxic_meta = toxic_meta.rename(columns={
        id_column: 'toxic_drug',
        smiles_column: 'Toxic_Drug_SMILES',
        label_column: 'Toxic_Drug_Y',
        'fg_names': 'Toxic_Drug_fg_names',
        'fg_counts': 'Toxic_Drug_fg_counts',
        'total_fg_count': 'Toxic_Drug_total_fg_count',
        'fg_full': 'Toxic_Drug_fg_full'
    })

    # Non-toxic side metadata
    non_toxic_meta = df_with_fg[required_fg_cols].copy()
    non_toxic_meta = non_toxic_meta.rename(columns={
        id_column: 'non_toxic_drug',
        smiles_column: 'Non_Toxic_Drug_SMILES',
        label_column: 'Non_Toxic_Drug_Y',
        'fg_names': 'Non_Toxic_Drug_fg_names',
        'fg_counts': 'Non_Toxic_Drug_fg_counts',
        'total_fg_count': 'Non_Toxic_Drug_total_fg_count',
        'fg_full': 'Non_Toxic_Drug_fg_full'
    })

    # Merge pairs with toxic metadata
    merged = pairs_df.merge(toxic_meta, on='toxic_drug', how='left')
    # Merge with non-toxic metadata
    merged = merged.merge(non_toxic_meta, on='non_toxic_drug', how='left')

    # Final column order per request (plus similarity at the end for reference)
    desired_columns = [
        'Toxic_Drug_SMILES', 'Non_Toxic_Drug_SMILES',
        'Toxic_Drug_Y', 'Non_Toxic_Drug_Y',
        'Toxic_Drug_fg_names', 'Toxic_Drug_fg_counts', 'Toxic_Drug_total_fg_count', 'Toxic_Drug_fg_full',
        'Non_Toxic_Drug_fg_names', 'Non_Toxic_Drug_fg_counts', 'Non_Toxic_Drug_total_fg_count', 'Non_Toxic_Drug_fg_full',
        'toxic_drug', 'non_toxic_drug', 'similarity'
    ]

    # Keep only available columns to avoid KeyError if some are missing for any reason
    available_columns = [c for c in desired_columns if c in merged.columns]
    merged = merged[available_columns]

    return merged


##### SMILES 리스트를 입력받아 FG 정보를 추출하는 함수 (새로 작성) #####
def extract_fg_from_smiles_list_new(
    smiles_list: list[str],
    lite_mode: bool = False,
    canonicalize: bool = True,
    verbose: bool = True,
    output_csv_path: str | None = None,
    keep_all: bool = False  # ✅ 추가: 실패도 행으로 보존
) -> pd.DataFrame:
    if verbose:
        print(f"🧪 Processing {len(smiles_list)} SMILES strings")
        print(f"🔧 Settings: lite_mode={lite_mode}, canonicalize={canonicalize}, keep_all={keep_all}")

    extractor = FunctionalGroupExtractor(
        lite_mode=lite_mode, verbose=False, max_workers=1, enable_checkpoint=False
    )

    rows = []
    failed_count = 0

    iterator = smiles_list
    if verbose:
        from tqdm import tqdm
        iterator = tqdm(smiles_list, desc="Extracting FG", unit="molecules")

    for smi in iterator:
        rec = {
            "input_smiles": smi,
            "working_smiles": None,
            "status": "failed",     # ok / failed
            "error": None,
            "fg_names": None,
            "fg_counts": None,
            "total_fg_count": None,
            "fg_full": None,
        }
        try:
            # 1) 정규화
            if canonicalize:
                cano = extractor.canonicalize_smiles(smi)
                if cano is None:
                    rec["error"] = "canonicalization failed"
                    failed_count += 1
                    if keep_all:
                        rows.append(rec)   # 실패도 남김
                    continue
                rec["working_smiles"] = cano
            else:
                rec["working_smiles"] = smi

            # 2) FG 추출
            fg = extractor.extract_single_fg_properties(rec["working_smiles"])
            if fg is None:
                rec["error"] = "fg_extraction returned None"
                failed_count += 1
                if keep_all:
                    rows.append(rec)
                continue

            # 3) 성공 기록
            rec["status"] = "ok"
            rec["fg_names"] = fg.get("fg_names")
            rec["fg_counts"] = fg.get("fg_counts")
            rec["total_fg_count"] = fg.get("total_fg_count")
            rec["fg_full"] = fg.get("fg_full")
            rows.append(rec)

        except Exception as e:
            rec["error"] = str(e)
            failed_count += 1
            if keep_all:
                rows.append(rec)
            if verbose:
                try:
                    from tqdm import tqdm
                    tqdm.write(f"❌ Error: {smi[:60]} … | {e}")
                except Exception:
                    print(f"❌ Error: {smi[:60]} … | {e}")

    # DataFrame 생성
    cols = ["input_smiles","working_smiles","status","error",
            "fg_names","fg_counts","total_fg_count","fg_full"]
    df = pd.DataFrame(rows, columns=cols)

    # keep_all=False 이면 성공건만 반환
    if not keep_all:
        df = df[df["status"] == "ok"].copy()

    # CSV 저장
    if output_csv_path:
        try:
            df.to_csv(output_csv_path, index=False)
            if verbose:
                print(f"💾 Results saved to: {output_csv_path}")
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to save CSV: {e}")

    if verbose:
        total = len(smiles_list)
        ok = (df["status"] == "ok").sum() if keep_all else len(df)
        print(f"\n✅ Processing completed:")
        print(f"  - Successfully processed: {ok} molecules")
        print(f"  - Failed: {failed_count} molecules")
        if ok > 0:
            print(f"  - Average FGs per molecule: {df.loc[df['status']=='ok','total_fg_count'].mean() if keep_all else df['total_fg_count'].mean():.2f}")

    return df


##### SMILES 리스트를 입력받아 FG 정보를 추출하는 함수 (기존 버전 - Deprecated) #####
def extract_fg_from_smiles_list(smiles_list: List[str], 
                               lite_mode: bool = False,
                               canonicalize: bool = True,
                               verbose: bool = True,
                               use_multithreading: bool = True,
                               max_workers: int = DEFAULT_MAX_WORKERS,
                               output_csv_path: str = None,
                               resume_from_csv: bool = True) -> pd.DataFrame:
    """
    SMILES 리스트를 입력받아 각각의 Functional Group 정보를 추출하고 DataFrame으로 반환
    각 SMILES 처리 시마다 CSV 파일에 저장하여 중간에 끊겨도 이어서 처리 가능
    
    Args:
        smiles_list: SMILES 문자열 리스트
        lite_mode: AccFG lite mode 사용 여부
        canonicalize: SMILES canonicalization 여부
        verbose: 진행상황 출력 여부
        use_multithreading: 멀티스레딩 사용 여부 (단일 SMILES 처리이므로 무시됨)
        max_workers: 최대 워커 수 (단일 SMILES 처리이므로 무시됨)
        output_csv_path: 결과 저장할 CSV 파일 경로 (None이면 자동 생성)
        resume_from_csv: 기존 CSV에서 이어서 처리할지 여부
        
    Returns:
        SMILES와 FG 정보가 포함된 DataFrame
        - Drug: SMILES 문자열
        - Drug_ID: 자동 생성된 ID
        - fg_names: Functional Group 이름 리스트
        - fg_counts: Functional Group 개수 딕셔너리
        - total_fg_count: 전체 Functional Group 개수
        - fg_full: 상세 FG 정보 (atom indices 포함)
        
    Example:
        >>> smiles_list = ['CCO', 'CCN', 'CC(=O)O']
        >>> df = extract_fg_from_smiles_list(smiles_list, output_csv_path='results.csv')
        >>> print(df[['Drug', 'fg_names', 'total_fg_count']])
    """
    if verbose:
        print(f"🧪 Processing {len(smiles_list)} SMILES strings")
        print(f"🔧 Settings: lite_mode={lite_mode}, canonicalize={canonicalize}")
    
    # 출력 CSV 파일 경로 설정
    if output_csv_path is None:
        output_csv_path = f"fg_extraction_results_{len(smiles_list)}_molecules.csv"
    
    # Functional Group 추출기 초기화
    extractor = FunctionalGroupExtractor(
        lite_mode=lite_mode,
        verbose=False,  # 개별 처리에서는 verbose 비활성화
        max_workers=1,
        enable_checkpoint=False
    )
    
    # 기존 CSV 파일에서 처리된 결과 로드
    processed_smiles = set()
    if resume_from_csv and os.path.exists(output_csv_path):
        try:
            existing_df = pd.read_csv(output_csv_path)
            # Drug_ID 대신 SMILES로 중복 체크 (Drug_ID는 리스트 순서이므로 중복될 수 있음)
            processed_smiles = set(existing_df['Drug'].tolist())
            if verbose:
                print(f"📂 Found existing CSV with {len(processed_smiles)} processed molecules")
        except Exception as e:
            if verbose:
                print(f"⚠️ Could not load existing CSV: {e}")
            processed_smiles = set()
    
    # CSV 파일 헤더 작성 (처음 실행 시)
    if not os.path.exists(output_csv_path):
        # 헤더만 있는 빈 CSV 파일 생성
        header_df = pd.DataFrame(columns=['Drug_ID', 'Drug', 'fg_names', 'fg_counts', 'total_fg_count', 'fg_full'])
        header_df.to_csv(output_csv_path, index=False)
        if verbose:
            print(f"📄 Created new CSV file: {output_csv_path}")
    
    # tqdm 진행률 표시 설정
    if verbose:
        from tqdm import tqdm
        pbar = tqdm(total=len(smiles_list), desc="Processing SMILES", unit="molecules",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} molecules [{elapsed}<{remaining}, {rate_fmt}]')
    
    # 각 SMILES를 순차적으로 처리
    for i, smiles in enumerate(smiles_list):
        # SMILES canonicalization (중복 체크 전에 수행)
        if canonicalize:
            canonical_smiles = extractor.canonicalize_smiles(smiles)
            if canonical_smiles is None:
                if verbose:
                    tqdm.write(f"⚠️ Invalid SMILES, skipping: {smiles}")
                    pbar.update(1)
                continue
            working_smiles = canonical_smiles
        else:
            working_smiles = smiles
        
        # SMILES 기준으로 중복 체크
        if working_smiles in processed_smiles:
            if verbose:
                pbar.update(1)
                tqdm.write(f"⏭️ Skipping already processed SMILES {i+1}/{len(smiles_list)}: {working_smiles[:50]}...")
            continue
        
        if verbose:
            tqdm.write(f"🔬 Processing SMILES {i+1}/{len(smiles_list)}: {working_smiles[:50]}...")
        
        try:
            
            # FG 정보 추출
            fg_properties = extractor.extract_single_fg_properties(working_smiles)
            
            # 결과를 DataFrame으로 변환
            result_row = pd.DataFrame({
                'Drug_ID': [i],
                'Drug': [working_smiles],
                'fg_names': [fg_properties['fg_names']],
                'fg_counts': [fg_properties['fg_counts']],
                'total_fg_count': [fg_properties['total_fg_count']],
                'fg_full': [fg_properties['fg_full']]
            })
            
            # CSV 파일에 추가 (append mode)
            result_row.to_csv(output_csv_path, mode='a', header=False, index=False)
            
            # 처리된 SMILES를 set에 추가
            processed_smiles.add(working_smiles)
            
            if verbose:
                tqdm.write(f"✅ Saved SMILES {i+1}: {fg_properties['total_fg_count']} FGs found")
                pbar.update(1)
                
        except Exception as e:
            if verbose:
                tqdm.write(f"❌ Error processing SMILES {i+1}: {e}")
            pbar.update(1)
            continue
    
    if verbose:
        pbar.close()
    
    # 최종 결과 로드
    try:
        final_df = pd.read_csv(output_csv_path)
        if verbose:
            print(f"✅ FG extraction completed for {len(final_df)} molecules")
            print(f"📊 Average FGs per molecule: {final_df['total_fg_count'].mean():.2f}")
            print(f"💾 Results saved to: {output_csv_path}")
        
        return final_df
    except Exception as e:
        if verbose:
            print(f"❌ Error loading final results: {e}")
        return pd.DataFrame()


def extract_fg_from_smiles_list_simple(smiles_list: List[str], 
                                      lite_mode: bool = False,
                                      canonicalize: bool = True,
                                      output_csv_path: str = None) -> pd.DataFrame:
    """
    SMILES 리스트를 입력받아 FG 정보를 추출하는 간단한 함수 (진행률 표시 없음)
    
    Args:
        smiles_list: SMILES 문자열 리스트
        lite_mode: AccFG lite mode 사용 여부
        canonicalize: SMILES canonicalization 여부
        output_csv_path: 결과 저장할 CSV 파일 경로
        
    Returns:
        SMILES와 FG 정보가 포함된 DataFrame
    """
    return extract_fg_from_smiles_list(
        smiles_list=smiles_list,
        lite_mode=lite_mode,
        canonicalize=canonicalize,
        verbose=False,
        use_multithreading=False,
        max_workers=1,
        output_csv_path=output_csv_path,
        resume_from_csv=True
    )


def test_smiles_list_extraction():
    """
    SMILES 리스트 추출 함수 테스트
    """
    # 테스트용 SMILES 리스트
    test_smiles = [
        'O=[N+]([O-])c1ccc(Cl)cc1',
        'Nc1ccc([N+](=O)[O-])cc1',
        'O=[N+]([O-])c1ccc(O)cc1',
        'O=C(O)c1ccc(C(=O)O)cc1',
        'Cl[Dy](Cl)Cl',
        'CCN(CC)CCO',
        'CCCCCC(CO)CCC',
        'OB(O)O',
        'OCc1ccccc1'
    ]
    
    print("🧪 Testing SMILES list extraction...")
    df = extract_fg_from_smiles_list(
        test_smiles, 
        verbose=True,
        output_csv_path="test_fg_extraction.csv"
    )
    
    print("\n📊 Results:")
    print(df[['Drug', 'fg_names', 'total_fg_count']].head())
    
    return df


# ========================================
# Checkpoint Management Utilities
# ========================================

def list_checkpoints(checkpoint_dir: str = None) -> List[str]:
    """
    사용 가능한 체크포인트 파일들을 나열
    
    Args:
        checkpoint_dir: 체크포인트 디렉토리 경로 (기본값: 현재 디렉토리의 .checkpoints)
        
    Returns:
        체크포인트 파일 경로 리스트
    """
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(os.getcwd(), '.checkpoints')
    
    if not os.path.exists(checkpoint_dir):
        return []
    
    checkpoint_files = []
    for file in os.listdir(checkpoint_dir):
        if file.endswith('_checkpoint.pkl'):
            checkpoint_files.append(os.path.join(checkpoint_dir, file))
    
    return sorted(checkpoint_files)


def get_checkpoint_info(checkpoint_path: str) -> Dict[str, Any]:
    """
    체크포인트 파일의 정보를 반환
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        
    Returns:
        체크포인트 정보 딕셔너리
    """
    checkpoint_data = _load_checkpoint(checkpoint_path)
    if checkpoint_data is None:
        return {}
    
    return {
        'file_path': checkpoint_path,
        'processed_count': checkpoint_data['processed_count'],
        'total_count': checkpoint_data['total_count'],
        'progress_percent': (checkpoint_data['processed_count'] / checkpoint_data['total_count']) * 100,
        'timestamp': checkpoint_data.get('timestamp', 'Unknown'),
        'completed': checkpoint_data.get('metadata', {}).get('completed', False),
        'metadata': checkpoint_data.get('metadata', {})
    }


def delete_checkpoint(checkpoint_path: str) -> bool:
    """
    체크포인트 파일 삭제
    
    Args:
        checkpoint_path: 체크포인트 파일 경로
        
    Returns:
        삭제 성공 여부
    """
    try:
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            logger.info(f"🗑️ Checkpoint deleted: {checkpoint_path}")
            return True
        return False
    except Exception as e:
        logger.error(f"❌ Failed to delete checkpoint: {e}")
        return False


def cleanup_checkpoints(checkpoint_dir: str = None, keep_latest: bool = True) -> int:
    """
    오래된 체크포인트 파일들 정리
    
    Args:
        checkpoint_dir: 체크포인트 디렉토리 경로
        keep_latest: 최신 체크포인트만 유지할지 여부
        
    Returns:
        삭제된 파일 수
    """
    checkpoint_files = list_checkpoints(checkpoint_dir)
    if not checkpoint_files:
        return 0
    
    if keep_latest:
        # 최신 파일만 유지하고 나머지 삭제
        files_to_delete = checkpoint_files[:-1]
    else:
        # 모든 파일 삭제
        files_to_delete = checkpoint_files
    
    deleted_count = 0
    for file_path in files_to_delete:
        if delete_checkpoint(file_path):
            deleted_count += 1
    
    return deleted_count


# ========================================
# TDC/ClinTox Dataset Specific Helpers
# ========================================

def process_tdc_dataset(df: pd.DataFrame,
                       lite_mode: bool = False,
                       canonicalize: bool = True,
                       verbose: bool = True) -> pd.DataFrame:
    """
    TDC 데이터셋 (Drug, Y 컬럼 구조) 전용 FG 추출 함수
    
    TDC 데이터셋 구조:
    - Drug: SMILES 문자열
    - Y: Label (0 or 1)
    - Drug_ID: (optional) 분자 ID
    
    Args:
        df: TDC DataFrame (Drug, Y 컬럼 포함)
        lite_mode: AccFG lite mode 사용 여부
        canonicalize: SMILES canonicalization 여부
        verbose: 진행상황 출력 여부
        
    Returns:
        FG properties가 추가된 DataFrame
        
    Example:
        >>> import pandas as pd
        >>> df = pd.read_csv('clintox.csv')  # Drug, Y 컬럼 포함
        >>> result = process_tdc_dataset(df)
        >>> print(result.columns)
        # ['Drug', 'Y', 'Drug_ID', ..., 'fg_names', 'fg_counts', 'total_fg_count', 'fg_full']
    """
    if verbose:
        print("🧪 Processing TDC dataset (Drug/Y format)")
    
    # Validate TDC format
    if 'Drug' not in df.columns:
        raise ValueError(f"'Drug' column not found. This function expects TDC format. Available columns: {df.columns.tolist()}")
    
    if 'Y' not in df.columns and verbose:
        logger.warning("'Y' column not found. This is unusual for TDC datasets.")
    
    extractor = FunctionalGroupExtractor(lite_mode=lite_mode, verbose=verbose)
    return extractor.process_dataframe(df, smiles_column='Drug', canonicalize=canonicalize)


def analyze_tdc_toxicity_similarity(df: pd.DataFrame,
                                   threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
                                   label_column: str = 'Y',
                                   non_toxic_label: int = 0,
                                   toxic_label: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    TDC toxicity 데이터셋의 non-toxic과 toxic 분자 간 유사도 분석
    
    Args:
        df: TDC DataFrame (Drug, Y 컬럼 포함)
        threshold: 높은 유사도 판단 임계값
        label_column: Label 컬럼 이름 (기본값: 'Y')
        non_toxic_label: Non-toxic을 나타내는 값 (기본값: 0)
        toxic_label: Toxic을 나타내는 값 (기본값: 1)
    
    Returns:
        (cross_sim_df, high_sim_pairs_df, statistics) 튜플
        
    Example:
        >>> df = pd.read_csv('clintox.csv')
        >>> cross_sim, high_pairs, stats = analyze_tdc_toxicity_similarity(df, threshold=0.7)
        >>> print(f"Found {len(high_pairs)} high similarity pairs")
    """
    # Validate columns
    required_cols = ['Drug', 'Drug_ID', label_column]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available: {df.columns.tolist()}")
    
    # Split by toxicity
    non_toxic_df = df[df[label_column] == non_toxic_label].copy()
    toxic_df = df[df[label_column] == toxic_label].copy()
    
    logger.info(f"Split dataset: {len(non_toxic_df)} non-toxic, {len(toxic_df)} toxic molecules")
    
    if len(non_toxic_df) == 0 or len(toxic_df) == 0:
        raise ValueError("Dataset must contain both toxic and non-toxic molecules")
    
    # Analyze cross-similarity
    return analyze_cross_similarity(
        non_toxic_df=non_toxic_df,
        toxic_df=toxic_df,
        threshold=threshold,
        id_column='Drug_ID',
        smiles_column='Drug'
    )


def load_and_process_tdc_dataset(file_path: str,
                                file_format: str = 'auto',
                                process_fg: bool = True,
                                lite_mode: bool = False,
                                verbose: bool = True) -> pd.DataFrame:
    """
    TDC 데이터셋 파일을 로드하고 FG 추출까지 한번에 수행
    
    Args:
        file_path: 데이터셋 파일 경로 (.csv, .parquet, .tsv 지원)
        file_format: 파일 형식 ('csv', 'parquet', 'tsv', 'auto')
        process_fg: FG 추출 수행 여부
        lite_mode: AccFG lite mode 사용 여부
        verbose: 진행상황 출력 여부
        
    Returns:
        처리된 DataFrame
        
    Example:
        >>> df = load_and_process_tdc_dataset('clintox.csv', process_fg=True)
        >>> print(df[['Drug', 'Y', 'fg_names']].head())
    """
    if verbose:
        print(f"📂 Loading dataset from: {file_path}")
    
    # Auto-detect format
    if file_format == 'auto':
        if file_path.endswith('.csv'):
            file_format = 'csv'
        elif file_path.endswith('.parquet'):
            file_format = 'parquet'
        elif file_path.endswith(('.tsv', '.tab')):
            file_format = 'tsv'
        else:
            raise ValueError(f"Cannot auto-detect format for: {file_path}")
    
    # Load file
    if file_format == 'csv':
        df = pd.read_csv(file_path)
    elif file_format == 'parquet':
        df = pd.read_parquet(file_path)
    elif file_format == 'tsv':
        df = pd.read_csv(file_path, sep='\t')
    else:
        raise ValueError(f"Unsupported format: {file_format}")
    
    if verbose:
        print(f"✅ Loaded {len(df)} rows")
        print(f"📊 Columns: {df.columns.tolist()}")
    
    # Process FG if requested
    if process_fg:
        df = process_tdc_dataset(df, lite_mode=lite_mode, verbose=verbose)
    
    return df