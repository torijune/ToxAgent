"""QA Templates for only_fg type pairs.

Tasks:
  Recognition:
    1. FG Identification      - which FG causes toxicity?
    2. Toxic Site Localization - where is the toxic FG in the molecule?
    3. FG Repair Planning      - how to repair (remove/add plan)?

  Generation:
    4. Toxicity Repair Generation Multi  - open-ended SMILES generation
    5. FG Remove Add Repair Multi        - guided SMILES generation with FG list

step_type parameter controls question wording:
  'one_step'   : exactly 1 FG difference (n_fg_diff = 1)
  'multi_step' : multiple FG differences (n_fg_diff > 1)
"""
import json
import os
from typing import Dict, List, Optional, Any, Tuple

import pandas as pd

# Import dataset/endpoint context from centralized endpoint_desc module
from endpoint_desc import get_dataset_context


# ---------------------------------------------------------------------------
# FG SMARTS helper
# ---------------------------------------------------------------------------

_FG_SMARTS_CACHE: Optional[Dict[str, str]] = None


def _load_fg_smarts() -> Dict[str, str]:
    global _FG_SMARTS_CACHE
    if _FG_SMARTS_CACHE is not None:
        return _FG_SMARTS_CACHE
    mapping: Dict[str, str] = {}
    search_dirs = [
        "detoxicity_model/AccFG/accfg",
        "AccFG/accfg",
        "../AccFG/accfg",
        "../../AccFG/accfg",
    ]
    for d in search_dirs:
        common = os.path.join(d, "fgs_common.csv")
        hetero = os.path.join(d, "fgs_heterocycle.csv")
        if os.path.exists(common):
            try:
                df = pd.read_csv(common)
                for _, row in df.iterrows():
                    name = str(row.get("Functional Group", "")).strip()
                    smarts = str(row.get("SMARTS Pattern", "")).strip()
                    if name and smarts and not name.startswith("#"):
                        mapping[name.lower()] = smarts
            except Exception:
                pass
        if os.path.exists(hetero):
            try:
                df = pd.read_csv(hetero)
                for _, row in df.iterrows():
                    name = str(row.get("Functional Group", "")).strip()
                    smarts = str(row.get("SMARTS Pattern", "")).strip()
                    if name and smarts:
                        mapping[name.lower()] = smarts
            except Exception:
                pass
        if mapping:
            break
    _FG_SMARTS_CACHE = mapping
    return mapping


def get_fg_smarts(fg_name: str) -> Optional[str]:
    """Return SMARTS pattern for a functional group name (case-insensitive)."""
    if not fg_name:
        return None
    mapping = _load_fg_smarts()
    key = fg_name.lower().strip()
    if key in mapping:
        return mapping[key]
    for k, v in mapping.items():
        if key in k or k in key:
            return v
    return None


# ---------------------------------------------------------------------------
# Step-type helper text
# ---------------------------------------------------------------------------

def _step_context(step_type: str) -> str:
    """Short sentence inserted into questions to describe the complexity."""
    if step_type == "one_step":
        return ("Note: The two molecules differ in exactly ONE functional group. "
                "This is a single, targeted structural change.\n\n")
    else:
        return ("Note: The two molecules differ in MULTIPLE functional groups, "
                "requiring a multi-step repair strategy.\n\n")


# ===========================================================================
# OnlyFGQATemplate
# ===========================================================================

class OnlyFGQATemplate:
    """QA templates for only_fg type pairs.
    
    All question-creation methods accept a `step_type` parameter:
      'one_step'   -> one FG difference, simpler phrasing
      'multi_step' -> multiple FG differences, emphasises multi-step nature
    """

    # ----- dataset context (class-level convenience) -----
    @staticmethod
    def get_dataset_context(dataset_name: Optional[str] = None,
                            endpoint: Optional[str] = None) -> str:
        return get_dataset_context(dataset_name, endpoint)

    # ===================================================================
    # Task 1 – FG Identification
    # ===================================================================
    @staticmethod
    def create_fg_identification_question(
        tx_fgs: List[str],
        nt_fgs: List[str],
        tx_mol: str,
        nt_mol: str,
        mol_format: str,
        tx_unique: Optional[List[str]] = None,
        nt_unique: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
    ) -> str:
        """Ask which FG(s) cause the toxicity difference."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        all_fgs = list(dict.fromkeys((tx_fgs or []) + (nt_fgs or [])))
        fgs_str = ", ".join(all_fgs) if all_fgs else "none"

        if step_type == "one_step":
            # "ONE functional group" 이라는 명시를 통해서 one step이라는 언급만 해줌 -> Answer의 포멧을 바꿔야할 듯
            task_desc = (
                "The following two molecules are structurally similar, but the first one is toxic "
                "while the second one is non-toxic. They differ in exactly ONE functional group. "
                "Identify the single functional group most likely responsible for the toxicity."
            )
        else: 
            task_desc = (
                "The following two molecules are structurally similar, but the first one is toxic "
                "while the second one is non-toxic. They differ in MULTIPLE functional groups. "
                "Identify all functional group(s) responsible for the toxicity difference."
            )

        body = (
            f"{task_desc}\n\n"
            f"Toxic_{mol_format}: {tx_mol}\n"
            f"NonToxic_{mol_format}: {nt_mol}\n\n"
            f"{step_note}"
            f"Functional Groups present in the pair: {fgs_str}\n\n"
            "Output STRICTLY in JSON (no prose outside JSON):\n"
            "{\n"
            '  "Correct_Answer": ["<FG name 1>", ...]  // all toxic FGs responsible\n'
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_fg_identification_answer(
        tx_unique: List[str],
        nt_unique: Optional[List[str]] = None,
    ) -> str:
        return json.dumps({"Correct_Answer": tx_unique or []}, ensure_ascii=False)

    # ===================================================================
    # Task 2 – Toxic Site Localization
    # ===================================================================
    @staticmethod
    def create_toxic_site_localization_question(
        tx_mol: str,
        mol_format: str,
        tx_fgs: List[str],
        nt_fgs: List[str],
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
    ) -> str:
        """Ask for atom-level locations of the toxic FG(s)."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        all_fgs = list(dict.fromkeys((tx_fgs or []) + (nt_fgs or [])))
        fgs_str = ", ".join(all_fgs) if all_fgs else "none"

        if step_type == "one_step":
            task_desc = (
                "Identify the functional group and its atom positions within the toxic molecule "
                "that are most likely responsible for its toxicity. "
                "There is exactly ONE key functional group difference."
            )
        else:
            task_desc = (
                "Identify all functional groups and their atom positions within the toxic molecule "
                "that are most likely responsible for its toxicity. "
                "There are MULTIPLE functional group differences to localize."
            )

        body = (
            f"{task_desc}\n\n"
            f"Toxic_{mol_format}: {tx_mol}\n\n"
            f"{step_note}"
            f"Functional Group Information: {fgs_str}\n\n"
            "IMPORTANT: Reference ONLY the functional groups listed above.\n"
            "Note on Atom Indices: Atom numbers appear as [C:1], [N:2], etc. "
            "Use these indices in your response.\n\n"
            "Output STRICTLY in JSON:\n"
            "{\n"
            '  "Toxic_Sites": [\n'
            '    {"FG_Name": "<name>", "Atom_Indices": [[...]]},\n'
            '    ...\n'
            "  ]\n"
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_toxic_site_localization_answer(
        tx_unique: List[str],
        tx_fg_full: Dict,
    ) -> str:
        if not tx_unique or not tx_fg_full:
            return json.dumps({"Toxic_Sites": []}, ensure_ascii=False)
        sites = []
        for fg in tx_unique:
            if fg in tx_fg_full:
                for indices in tx_fg_full[fg]:
                    sites.append({"FG_Name": fg, "Atom_Indices": [list(indices)]})
        return json.dumps({"Toxic_Sites": sites}, ensure_ascii=False)

    # ===================================================================
    # Task 3 – FG Repair Planning
    # ===================================================================
    @staticmethod
    def create_fg_repair_planning_question(
        tx_mol: str,
        mol_format: str,
        tx_fgs: List[str],
        nt_fgs: List[str],
        tx_unique: Optional[List[str]] = None,
        nt_unique: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
    ) -> str:
        """Ask for a remove/add plan to reduce toxicity."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        all_fgs = list(dict.fromkeys((tx_fgs or []) + (nt_fgs or [])))
        fgs_str = ", ".join(all_fgs) if all_fgs else "none"

        atom_note = (
            "Note on Atom Indices: Atom numbers appear as [C:1], [N:2], etc. "
            "Use these when identifying atoms to modify. "
            "Output should NOT contain atom numbers.\n\n"
        ) if mol_format.upper() == "SMILES" else ""

        if step_type == "one_step":
            task_desc = (
                "The following molecule is toxic. Design a repair plan targeting "
                "the single functional group responsible for toxicity."
            )
        else:
            task_desc = (
                "The following molecule is toxic. Design a multi-step repair plan "
                "targeting all functional groups responsible for toxicity."
            )

        body = (
            f"{task_desc}\n\n"
            f'Toxic_{mol_format.upper()}: "{tx_mol}"\n\n'
            f"{atom_note}"
            f"{step_note}"
            f"Functional Groups present in the pair: {fgs_str}\n\n"
            "Create a plan by:\n"
            "1. Identifying which FGs should be REMOVED (drive toxicity).\n"
            "2. Identifying which FGs should be ADDED (safer replacements).\n"
            "3. Briefly explaining how the plan reduces toxicity.\n\n"
            "Output STRICTLY in JSON:\n"
            "{\n"
            '  "Removed_Functional_Groups": ["fg_1", ...],\n'
            '  "Added_Functional_Groups": ["fg_1", ...],\n'
            '  "Reasoning": "<2-3 sentences>"\n'
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_fg_repair_planning_answer(
        tx_unique: List[str],
        nt_unique: List[str],
        tx_fg_full: Optional[Dict] = None,
        nt_fg_full: Optional[Dict] = None,
    ) -> str:
        def _entries(unique: List[str], fg_full: Optional[Dict]) -> List[Dict]:
            out = []
            for fg in (unique or []):
                details = fg
                if fg_full and fg in fg_full:
                    parts = [f"{fg} (atoms: {', '.join(map(str, idx))})"
                             for idx in fg_full[fg]]
                    details = "; ".join(parts)
                out.append({"FG_Name": fg, "Details": details})
            return out

        parts = []
        if tx_unique:
            parts.append(f"Remove toxic FGs ({', '.join(tx_unique)})")
        if nt_unique:
            parts.append(f"replace with safer FGs ({', '.join(nt_unique)})")
        reasoning = " and ".join(parts) if parts else "No FG changes required."

        return json.dumps({
            "Removed_Functional_Groups": _entries(tx_unique, tx_fg_full),
            "Added_Functional_Groups": _entries(nt_unique, nt_fg_full),
            "Reasoning": reasoning,
        }, ensure_ascii=False)

    # ===================================================================
    # Task 4 – Toxicity Repair Generation Multi (open-ended)
    # ===================================================================
    @staticmethod
    def create_toxicity_repair_generation_multi_question(
        tx_mol: str,
        mol_format: str,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
    ) -> str:
        """Ask LLM to generate 3 repaired SMILES (open-ended, FG-level guidance)."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        if step_type == "one_step":
            focus = (
                "Focus on the single functional group modification responsible for toxicity. "
                "Each strategy should explore a different way to address this one change."
            )
        else:
            focus = (
                "Focus on ALL functional group modifications responsible for toxicity. "
                "Each strategy may address the changes in different order or combination."
            )

        body = (
            "The following molecule is toxic. Modify or replace the functional groups "
            "responsible for toxicity to design a non-toxic analog. "
            "Generate THREE different repaired SMILES strings.\n\n"
            f"Toxic_{mol_format}: {tx_mol}\n\n"
            f"{step_note}"
            f"{focus}\n\n"
            "Instructions:\n"
            "- Generate THREE chemically valid, parseable SMILES (no atom numbers).\n"
            "- Each SMILES represents a different repair strategy.\n"
            "- Describe modifications and expected toxicity outcome.\n\n"
            "Output STRICTLY in JSON:\n"
            "{\n"
            '  "Repaired_SMILES": ["<SMILES1>", "<SMILES2>", "<SMILES3>"],\n'
            '  "Modification": "<description of strategies, 2-3 sentences>",\n'
            '  "Expected_Toxicity": "non-toxic"\n'
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_toxicity_repair_generation_multi_answer(
        tx_unique: List[str],
        nt_unique: List[str],
        nt_mol: str,
    ) -> str:
        if not tx_unique:
            desc = "No specific toxic FGs detected; general structural modification applied."
        else:
            desc = f"Removed toxic FGs ({tx_unique})"
            if nt_unique:
                desc += f" and introduced safer FGs ({nt_unique})"
            desc += " to reduce toxicity."
        return json.dumps({
            "Repaired_SMILES": nt_mol,
            "Modification": desc,
            "Expected_Toxicity": "non-toxic",
        }, ensure_ascii=False)

    # ===================================================================
    # Task 5 – FG Remove Add Repair Multi (guided generation)
    # ===================================================================
    @staticmethod
    def create_fg_remove_add_repair_multi_question(
        tx_mol: str,
        mol_format: str,
        tx_unique: List[str],
        nt_unique: List[str],
        tx_fg_full: Optional[Dict] = None,
        nt_fg_full: Optional[Dict] = None,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
    ) -> str:
        """Ask LLM to generate 3 repaired SMILES with explicit FG lists."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        def _fg_lines(fg_list: List[str], fg_full: Optional[Dict]) -> str:
            lines = []
            for fg in (fg_list or []):
                smarts = get_fg_smarts(fg)
                if fg_full and fg in fg_full:
                    for idx in fg_full[fg]:
                        atoms = ", ".join(map(str, idx))
                        s = f"  - {fg} (atoms: {atoms}"
                        if smarts:
                            s += f", SMARTS: {smarts}"
                        lines.append(s + ")")
                else:
                    s = f"  - {fg}"
                    if smarts:
                        s += f" (SMARTS: {smarts})"
                    lines.append(s)
            return "\n".join(lines) if lines else "  - none"

        remove_str = _fg_lines(tx_unique, tx_fg_full)
        add_str = _fg_lines(nt_unique, nt_fg_full)

        if step_type == "one_step":
            intro = (
                "The following molecule is toxic. Apply the ONE functional group change "
                "specified below to design a non-toxic analog."
            )
        else:
            intro = (
                "The following molecule is toxic. Apply ALL the functional group changes "
                "specified below. This is a multi-step repair — all changes are required."
            )

        body = (
            f"{intro}\n\n"
            f'Toxic_{mol_format}: "{tx_mol}"\n\n'
            "Atom Indices note: numbers appear as [C:1], [N:2], etc. "
            "Your output SMILES must NOT contain atom numbers.\n\n"
            f"{step_note}"
            f"Functional groups to REMOVE (toxic):\n{remove_str}\n\n"
            f"Functional groups to ADD (non-toxic replacements):\n{add_str}\n\n"
            "Generate THREE different repaired SMILES, each reflecting a different strategy "
            "for how to perform these changes.\n\n"
            "Output STRICTLY in JSON:\n"
            "{\n"
            '  "Repaired_SMILES": ["<SMILES1>", "<SMILES2>", "<SMILES3>"]\n'
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_fg_remove_add_repair_multi_answer(nt_mol: str) -> str:
        return json.dumps({"Repaired_SMILES": nt_mol}, ensure_ascii=False)

    # ===================================================================
    # Unified create_question / create_answer interface
    # (keeps backward-compat with task files that use question_type=)
    # ===================================================================
    @staticmethod
    def create_question(
        tx_fgs: List[str],
        nt_fgs: List[str],
        tx_mol: str,
        nt_mol: str,
        mol_format: str,
        question_type: str,
        tx_unique: Optional[List[str]] = None,
        nt_unique: Optional[List[str]] = None,
        tx_fg_full: Optional[Dict] = None,
        nt_fg_full: Optional[Dict] = None,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
        # legacy params
        common_fgs: Optional[List[str]] = None,
        correct_fg: Optional[str] = None,
        subset: Optional[str] = None,
    ) -> str:
        """Unified question dispatcher."""
        ep = endpoint or subset  # accept both names
        if question_type == "fg_identification":
            return OnlyFGQATemplate.create_fg_identification_question(
                tx_fgs, nt_fgs, tx_mol, nt_mol, mol_format,
                tx_unique, nt_unique, dataset_name, ep, step_type)
        if question_type == "toxic_site_localization":
            return OnlyFGQATemplate.create_toxic_site_localization_question(
                tx_mol, mol_format, tx_fgs, nt_fgs, dataset_name, ep, step_type)
        if question_type == "fg_repair_planning":
            return OnlyFGQATemplate.create_fg_repair_planning_question(
                tx_mol, mol_format, tx_fgs, nt_fgs,
                tx_unique, nt_unique, dataset_name, ep, step_type)
        if question_type == "toxicity_repair_generation_multi":
            return OnlyFGQATemplate.create_toxicity_repair_generation_multi_question(
                tx_mol, mol_format, dataset_name, ep, step_type)
        if question_type in ("fg_remove_add_repair_multi", "fg_remove_add_repair"):
            # compute tx_unique / nt_unique on-the-fly if not provided
            tx_u = tx_unique if tx_unique is not None else sorted(set(tx_fgs) - set(nt_fgs))
            nt_u = nt_unique if nt_unique is not None else sorted(set(nt_fgs) - set(tx_fgs))
            return OnlyFGQATemplate.create_fg_remove_add_repair_multi_question(
                tx_mol, mol_format, tx_u, nt_u,
                tx_fg_full, nt_fg_full, dataset_name, ep, step_type)
        raise ValueError(f"Unknown question_type: {question_type}")

    @staticmethod
    def create_answer(
        tx_unique: List[str],
        nt_unique: List[str],
        nt_mol: str,
        question_type: str,
        tx_fg_full: Optional[Dict] = None,
        nt_fg_full: Optional[Dict] = None,
        step_type: str = "one_step",
        # legacy params
        common_fgs: Optional[List[str]] = None,
        subset: Optional[str] = None,
    ) -> str:
        """Unified answer dispatcher."""
        if question_type == "fg_identification":
            return OnlyFGQATemplate.create_fg_identification_answer(tx_unique, nt_unique)
        if question_type == "toxic_site_localization":
            return OnlyFGQATemplate.create_toxic_site_localization_answer(
                tx_unique, tx_fg_full or {})
        if question_type == "fg_repair_planning":
            return OnlyFGQATemplate.create_fg_repair_planning_answer(
                tx_unique, nt_unique, tx_fg_full, nt_fg_full)
        if question_type == "toxicity_repair_generation_multi":
            return OnlyFGQATemplate.create_toxicity_repair_generation_multi_answer(
                tx_unique, nt_unique, nt_mol)
        if question_type in ("fg_remove_add_repair_multi", "fg_remove_add_repair"):
            return OnlyFGQATemplate.create_fg_remove_add_repair_multi_answer(nt_mol)
        raise ValueError(f"Unknown question_type: {question_type}")


# ---------------------------------------------------------------------------
# Backward-compat alias (task files currently use QATemplate)
# ---------------------------------------------------------------------------
QATemplate = OnlyFGQATemplate
