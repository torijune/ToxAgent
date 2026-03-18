"""Main module for building QA sets from pairs_one_diff_only.csv.

This module provides a unified interface to generate QA sets for all tasks.
All task-specific generators can be imported and used from here.

Structure:
- recognition_task/
  - only_fg/one_step/
  - only_fg/multi_step/
  - only_stereo/one_step/
  - only_stereo/multi_step/
  - both/one_step/
  - both/multi_step/
- generation_task/
  - only_fg/one_step/
  - only_fg/multi_step/
  - only_stereo/one_step/
  - only_stereo/multi_step/
  - both/one_step/
  - both/multi_step/
"""
from pathlib import Path
import sys
import importlib
from typing import Dict, Callable, Optional, List
import inspect

# Add src directory to path
src_dir = Path(__file__).parent
sys.path.insert(0, str(src_dir))


# ============================================================================
# Task Module Registry
# ============================================================================
# This registry maps task categories to their module paths and function names.
# 
# To add a new task:
#   1. Create the task module file (e.g., recognition_task/only_fg/multi_step/new_task.py)
#   2. Implement a function with signature: generate_*_qa(csv_path, output_path, smiles_format, limit)
#   3. Add an entry to TASK_REGISTRY below:
#      (category, 'module.path.to.task', 'function_name', 'task_key')
#
# Format: (task_category, module_path, function_name, task_key)
#   - task_category: 'recognition' or 'generation'
#   - module_path: Dot-separated module path from src/ directory
#   - function_name: Name of the generator function in the module
#   - task_key: Unique identifier for the task (used in CLI and API)
#
TASK_REGISTRY: List[tuple] = [
    # Recognition tasks - only_fg/one_step
    ('recognition', 'recognition_task.only_fg.one_step.fg_identification', 'generate_fg_identification_qa', 'fg_identification'),
    ('recognition', 'recognition_task.only_fg.one_step.fg_localization', 'generate_toxic_site_localization_qa', 'toxic_site_localization'),
    ('recognition', 'recognition_task.only_fg.one_step.fg_remove_add_planning', 'generate_fg_repair_planning_qa', 'fg_repair_planning'),
    
    # Recognition tasks - only_fg/multi_step
    ('recognition', 'recognition_task.only_fg.multi_step.fg_identification', 'generate_fg_identification_qa', 'fg_identification_multi'),
    ('recognition', 'recognition_task.only_fg.multi_step.fg_localization', 'generate_toxic_site_localization_qa', 'toxic_site_localization_multi'),
    ('recognition', 'recognition_task.only_fg.multi_step.fg_remove_add_planning', 'generate_fg_repair_planning_qa', 'fg_repair_planning_multi'),
    
    # Recognition tasks - only_stereo/one_step
    ('recognition', 'recognition_task.only_stereo.one_step.stereo_identification', 'generate_stereo_identification_qa', 'stereo_identification'),
    ('recognition', 'recognition_task.only_stereo.one_step.stereo_localization', 'generate_stereo_localization_qa', 'stereo_localization'),
    ('recognition', 'recognition_task.only_stereo.one_step.stereo_remove_add_planning', 'generate_stereo_repair_planning_qa', 'stereo_repair_planning'),
    
    # Recognition tasks - only_stereo/multi_step
    ('recognition', 'recognition_task.only_stereo.multi_step.stereo_identification', 'generate_stereo_identification_qa', 'stereo_identification_multi'),
    ('recognition', 'recognition_task.only_stereo.multi_step.stereo_localization', 'generate_stereo_localization_qa', 'stereo_localization_multi'),
    ('recognition', 'recognition_task.only_stereo.multi_step.stereo_remove_add_planning', 'generate_stereo_repair_planning_qa', 'stereo_repair_planning_multi'),
    
    # Generation tasks - only_fg/one_step
    ('generation', 'generation_task.only_fg.one_step.fg_remove_add_repair_gen', 'generate_fg_remove_add_repair_multi_qa', 'fg_remove_add_repair_multi'),
    ('generation', 'generation_task.only_fg.one_step.fg_repair_gen', 'generate_toxicity_repair_generation_multi_qa', 'toxicity_repair_generation_multi'),
    
    # Generation tasks - only_fg/multi_step
    ('generation', 'generation_task.only_fg.multi_step.fg_remove_add_repair_gen', 'generate_fg_remove_add_repair_multi_qa', 'fg_remove_add_repair_multi_step'),
    ('generation', 'generation_task.only_fg.multi_step.fg_repair_gen', 'generate_toxicity_repair_generation_multi_qa', 'toxicity_repair_generation_multi_step'),
    
    # Generation tasks - only_stereo/one_step
    ('generation', 'generation_task.only_stereo.one_step.stereo_remove_add_repair_gen', 'generate_stereo_remove_add_repair_gen_qa', 'stereo_remove_add_repair_gen'),
    ('generation', 'generation_task.only_stereo.one_step.stereo_repair_gen', 'generate_stereo_repair_gen_qa', 'stereo_repair_gen'),
    
    # Generation tasks - only_stereo/multi_step
    ('generation', 'generation_task.only_stereo.multi_step.stereo_remove_add_repair_gen', 'generate_stereo_remove_add_repair_gen_qa', 'stereo_remove_add_repair_gen_multi'),
    ('generation', 'generation_task.only_stereo.multi_step.stereo_repair_gen', 'generate_stereo_repair_gen_qa', 'stereo_repair_gen_multi'),
    
    # Recognition/Generation tasks - both/one_step and both/multi_step
    # (Add similar entries for 'both' category tasks when implemented)
]


def _load_task_function(module_path: str, function_name: str, verbose: bool = False) -> Optional[Callable]:
    """Dynamically load a task function from a module.
    
    Args:
        module_path: Dot-separated module path (e.g., 'recognition_task.only_fg.one_step.fg_identification')
        function_name: Name of the function to load
        verbose: Whether to print warnings (default: False)
        
    Returns:
        The function if found, None otherwise
    """
    try:
        # Add project root to path for Mol_FG imports
        project_root = src_dir.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        module = importlib.import_module(module_path)
        if hasattr(module, function_name):
            return getattr(module, function_name)
        else:
            if verbose:
                print(f"Warning: Function '{function_name}' not found in module '{module_path}'")
            return None
    except ImportError as e:
        # Suppress import warnings during module loading - they're expected if dependencies aren't available
        # The modules handle their own import paths when actually executed
        if verbose:
            print(f"Warning: Could not import module '{module_path}': {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"Warning: Error loading function '{function_name}' from '{module_path}': {e}")
        return None


def _build_task_generators() -> Dict[str, Callable]:
    """Build the TASK_GENERATORS dictionary by loading all registered tasks.
    
    Returns:
        Dictionary mapping task keys to their generator functions
    """
    generators = {}
    
    for category, module_path, function_name, task_key in TASK_REGISTRY:
        func = _load_task_function(module_path, function_name)
        if func is not None:
            generators[task_key] = func
        else:
            print(f"Warning: Failed to load task '{task_key}' from {module_path}.{function_name}")
    
    return generators


# Build the task generators dictionary
TASK_GENERATORS: Dict[str, Callable] = _build_task_generators()


# ============================================================================
# Public API
# ============================================================================

def get_available_tasks() -> List[str]:
    """Get list of all available task names.
    
    Returns:
        List of task names that can be used with generate_qa_for_task()
    """
    return sorted(TASK_GENERATORS.keys())


def get_tasks_by_category(category: Optional[str] = None) -> Dict[str, List[str]]:
    """Get tasks grouped by category.
    
    Args:
        category: Optional category filter ('recognition' or 'generation')
        
    Returns:
        Dictionary mapping categories to lists of task names
    """
    tasks_by_category = {
        'recognition': [],
        'generation': []
    }
    
    for cat, _, _, task_key in TASK_REGISTRY:
        if task_key in TASK_GENERATORS:
            if category is None or cat == category:
                tasks_by_category[cat].append(task_key)
    
    return tasks_by_category


def _get_task_metadata(task_name: str) -> Optional[Dict[str, str]]:
    """Get metadata for a task (style, step_type, etc.) from TASK_REGISTRY.
    
    Args:
        task_name: Task key name
        
    Returns:
        Dictionary with metadata: {'style': 'only_fg', 'step_type': 'one_step', 'task_type': 'recognition', 'task_file': 'fg_identification'}
        Returns None if task not found
    """
    for cat, module_path, _, task_key in TASK_REGISTRY:
        if task_key == task_name:
            # Parse module_path: e.g., 'recognition_task.only_fg.one_step.fg_identification'
            parts = module_path.split('.')
            
            # Extract style (only_fg, only_stereo, both)
            style = None
            step_type = None
            task_file = None
            
            if 'only_fg' in parts:
                style = 'only_fg'
            elif 'only_stereo' in parts or 'stereo' in parts:
                style = 'stereo_fg'  # QA 디렉토리 구조에 맞춤
            elif 'both' in parts:
                style = 'both'
            
            # Extract step_type (one_step, multi_step)
            if 'one_step' in parts:
                step_type = 'one_step'
            elif 'multi_step' in parts:
                step_type = 'multi_step'
            
            # Extract task file name (last part before function)
            if len(parts) > 0:
                task_file = parts[-1]
            
            return {
                'style': style,
                'step_type': step_type,
                'task_type': cat,  # 'recognition' or 'generation'
                'task_file': task_file,
                'module_path': module_path
            }
    
    return None


def _ensure_output_dir(file_path: str) -> None:
    """Create parent directories for the given file path if they do not exist.
    
    Ensures QA/VQA, style (only_fg/stereo/both), and step_type (one_step/multi_step)
    directories exist so that output files can be written without error.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _get_default_output_path(task_name: str, smiles_format: str = "smiles", base_dir: Optional[str] = None) -> Path:
    """Get default output path for a task based on its metadata.
    
    Args:
        task_name: Task key name
        smiles_format: Molecular format ('smiles' or 'selfies')
        base_dir: Base directory for QA output (default: MolDeTox_bench/QA)
        
    Returns:
        Path object for the output file
    """
    if base_dir is None:
        # Default to MolDeTox_bench/QA relative to src directory
        base_dir = src_dir.parent / "QA"
    
    base_path = Path(base_dir)
    
    # Get task metadata
    metadata = _get_task_metadata(task_name)
    if metadata is None:
        # Fallback: use task_name directly
        return base_path / f"{task_name}_{smiles_format}.jsonl"
    
    style = metadata.get('style')
    step_type = metadata.get('step_type')
    
    # Use task_name for filename (more consistent)
    # Build path: QA/{style}/{step_type}/{task_name}_{smiles_format}.jsonl
    if style and step_type:
        output_path = base_path / style / step_type / f"{task_name}_{smiles_format}.jsonl"
    elif style:
        output_path = base_path / style / f"{task_name}_{smiles_format}.jsonl"
    else:
        output_path = base_path / f"{task_name}_{smiles_format}.jsonl"
    
    return output_path


def generate_qa_for_task(
    task_name: str,
    csv_path: str,
    output_path: Optional[str] = None,
    smiles_format: str = "smiles",
    limit: Optional[int] = None,
    auto_save: bool = True,
    qa_base_dir: Optional[str] = None
) -> List[dict]:
    """Generate QA set for a specific task.
    
    Args:
        task_name: Name of the task (e.g., 'fg_identification', 'fg_repair_planning')
        csv_path: Path to pairs_one_diff_only.csv
        output_path: Optional path to save JSONL output (if None and auto_save=True, uses default path)
        smiles_format: Molecular format ('smiles' or 'selfies')
        limit: Optional limit on number of items to generate
        auto_save: If True and output_path is None, automatically save to QA/{style}/{step_type}/ directory
        qa_base_dir: Base directory for QA output (default: MolDeTox_bench/QA, only used if auto_save=True)
        
    Returns:
        List of QA item dictionaries
        
    Raises:
        ValueError: If task_name is not recognized
    """
    if task_name not in TASK_GENERATORS:
        available_tasks = ', '.join(get_available_tasks())
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {available_tasks}"
        )
    
    # Determine output path
    final_output_path = output_path
    if final_output_path is None and auto_save:
        final_output_path = str(_get_default_output_path(task_name, smiles_format, qa_base_dir))
        print(f"Auto-saving to: {final_output_path}")
    
    # Ensure QA/VQA and style/step_type directories exist before writing
    if final_output_path:
        _ensure_output_dir(final_output_path)
    
    generator = TASK_GENERATORS[task_name]
    return generator(
        csv_path=csv_path,
        output_path=final_output_path,
        smiles_format=smiles_format,
        limit=limit
    )


def generate_qa_for_all_tasks(
    csv_path: str,
    output_dir: Optional[str] = None,
    smiles_format: str = "smiles",
    limit: Optional[int] = None,
    category: Optional[str] = None,
    auto_save: bool = True,
    qa_base_dir: Optional[str] = None
) -> Dict[str, List[dict]]:
    """Generate QA sets for all available tasks.
    
    Args:
        csv_path: Path to pairs_one_diff_only.csv
        output_dir: Optional directory to save JSONL outputs (if None and auto_save=True, uses QA/{style}/{step_type}/)
        smiles_format: Molecular format ('smiles' or 'selfies')
        limit: Optional limit on number of items to generate per task
        category: Optional category filter ('recognition' or 'generation')
        auto_save: If True, automatically save to appropriate QA/{style}/{step_type}/ directories
        qa_base_dir: Base directory for QA output (default: MolDeTox_bench/QA, only used if auto_save=True)
        
    Returns:
        Dictionary mapping task names to their QA items
    """
    results = {}
    tasks_to_process = get_available_tasks()
    
    if category:
        tasks_by_cat = get_tasks_by_category(category)
        tasks_to_process = tasks_by_cat.get(category, [])
    
    for task_name in tasks_to_process:
        print(f"\n{'='*80}")
        print(f"Processing task: {task_name}")
        print(f"{'='*80}")
        
        # Determine output path
        output_path = None
        if output_dir:
            output_path = str(Path(output_dir) / f"{task_name}.jsonl")
        elif auto_save:
            # Use auto-generated path based on task metadata
            output_path = str(_get_default_output_path(task_name, smiles_format, qa_base_dir))
            print(f"Auto-saving to: {output_path}")
        
        try:
            qa_items = generate_qa_for_task(
                task_name=task_name,
                csv_path=csv_path,
                output_path=output_path,
                smiles_format=smiles_format,
                limit=limit,
                auto_save=False  # Already handled above
            )
            results[task_name] = qa_items
            print(f"✓ Generated {len(qa_items)} QA items for {task_name}")
        except Exception as e:
            print(f"✗ Error generating QA for {task_name}: {e}")
            import traceback
            traceback.print_exc()
            results[task_name] = []
    
    return results


# ============================================================================
# Direct function exports for backward compatibility
# ============================================================================
# Export individual functions for direct import
# These will be populated dynamically from TASK_GENERATORS

def _export_functions():
    """Export all task generator functions to module namespace."""
    for task_name, generator in TASK_GENERATORS.items():
        # Create a clean function name from task name
        func_name = generator.__name__
        globals()[func_name] = generator


_export_functions()

# Build __all__ list dynamically
__all__ = [
    'TASK_GENERATORS',
    'get_available_tasks',
    'get_tasks_by_category',
    'generate_qa_for_task',
    'generate_qa_for_all_tasks',
] + [gen.__name__ for gen in TASK_GENERATORS.values()]


# ============================================================================
# Command-line interface
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate QA sets from pairs_one_diff_only.csv",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available tasks
  python buildqa.py --list-tasks
  
  # Generate QA for a specific task (auto-saves to QA/only_fg/one_step/fg_identification_smiles.jsonl)
  python buildqa.py --task fg_identification --csv-path data/pairs.csv
  
  # Generate QA for a specific task with custom output path
  python buildqa.py --task fg_identification --csv-path data/pairs.csv --output-path custom.jsonl
  
  # Generate QA for all tasks (auto-saves to QA/{style}/{step_type}/ directories)
  python buildqa.py --all --csv-path data/pairs.csv
  
  # Generate QA for all recognition tasks
  python buildqa.py --all --csv-path data/pairs.csv --category recognition
  
  # Generate QA without auto-saving (use custom output paths)
  python buildqa.py --all --csv-path data/pairs.csv --no-auto-save --output-dir custom_output/
        """
    )
    
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        choices=get_available_tasks(),
        help="Task name to generate QA for (use --all to generate for all tasks)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Generate QA for all available tasks"
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        required=False,
        help="Path to pairs_one_diff_only.csv (required unless --list-tasks is used)"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Path to save JSONL output (for single task mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save JSONL outputs (for --all mode, files will be named {task_name}.jsonl). If not specified, uses auto-generated paths in QA/{style}/{step_type}/"
    )
    parser.add_argument(
        "--no-auto-save",
        action="store_true",
        help="Disable automatic saving to QA/{style}/{step_type}/ directories"
    )
    parser.add_argument(
        "--qa-base-dir",
        type=str,
        default=None,
        help="Base directory for QA output (default: MolDeTox_bench/QA)"
    )
    parser.add_argument(
        "--smiles-format",
        type=str,
        default="smiles",
        choices=["smiles", "selfies"],
        help="Molecular format (smiles or selfies)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of items to generate per task"
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        choices=["recognition", "generation"],
        help="Category filter (only used with --all)"
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit"
    )
    
    args = parser.parse_args()
    
    # List tasks if requested (do this before other validations)
    if args.list_tasks:
        print("Available tasks:")
        print("=" * 80)
        tasks_by_cat = get_tasks_by_category()
        print(f"\nRecognition tasks ({len(tasks_by_cat['recognition'])}):")
        for task in tasks_by_cat['recognition']:
            print(f"  - {task}")
        print(f"\nGeneration tasks ({len(tasks_by_cat['generation'])}):")
        for task in tasks_by_cat['generation']:
            print(f"  - {task}")
        print(f"\nTotal: {len(get_available_tasks())} tasks")
        sys.exit(0)
    
    # Validate arguments (skip if just listing tasks)
    if not args.list_tasks:
        if not args.csv_path:
            parser.error("--csv-path is required (unless --list-tasks is used)")
        
        if not args.all and not args.task:
            parser.error("Either --task or --all must be specified")
        
        if args.all and args.output_path:
            parser.error("--output-path cannot be used with --all, use --output-dir instead")
        
        if args.task and args.output_dir:
            parser.error("--output-dir cannot be used with --task, use --output-path instead")
    
    # Generate QA
    auto_save = not args.no_auto_save
    
    if args.all:
        results = generate_qa_for_all_tasks(
            csv_path=args.csv_path,
            output_dir=args.output_dir,
            smiles_format=args.smiles_format,
            limit=args.limit,
            category=args.category,
            auto_save=auto_save,
            qa_base_dir=args.qa_base_dir
        )
        
        print(f"\n{'='*80}")
        print("Summary:")
        print(f"{'='*80}")
        total_items = sum(len(items) for items in results.values())
        print(f"Total tasks processed: {len(results)}")
        print(f"Total QA items generated: {total_items}")
        for task_name, items in results.items():
            if auto_save and not args.output_dir:
                output_path = _get_default_output_path(task_name, args.smiles_format, args.qa_base_dir)
                print(f"  {task_name}: {len(items)} items -> {output_path}")
            else:
                print(f"  {task_name}: {len(items)} items")
    else:
        qa_items = generate_qa_for_task(
            task_name=args.task,
            csv_path=args.csv_path,
            output_path=args.output_path,
            smiles_format=args.smiles_format,
            limit=args.limit,
            auto_save=auto_save,
            qa_base_dir=args.qa_base_dir
        )
        if auto_save and not args.output_path:
            output_path = _get_default_output_path(args.task, args.smiles_format, args.qa_base_dir)
            print(f"\nTotal QA items generated: {len(qa_items)}")
            print(f"Saved to: {output_path}")
        else:
            print(f"\nTotal QA items generated: {len(qa_items)}")
