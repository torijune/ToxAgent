"""QA Templates for only_stereo type pairs.

Tasks:
  Recognition:
    1. Stereo Identification         - which stereo feature causes toxicity?
    2. Stereo Toxic Site Localization - where is the toxic stereo feature?
    3. Stereo Repair Planning         - how to repair (plan with stereo details)?

  Generation:
    4. Stereo Repair Gen             - generate single repaired SMILES
    5. Stereo Remove Add Repair Gen  - generate 3 repaired SMILES (guided)

step_type parameter:
  'one_step'   : exactly 1 stereo difference (n_stereo_diff = 1)
  'multi_step' : multiple stereo differences (n_stereo_diff > 1)
"""
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Import dataset/endpoint context from centralized endpoint_desc module
from endpoint_desc import get_dataset_context

# Import add_atom_numbers_to_smiles from utils (self-contained)
def _get_add_atom_numbers():
    try:
        from utils import add_atom_numbers_to_smiles
        return add_atom_numbers_to_smiles
    except ImportError:
        return lambda s: s  # no-op fallback


# ---------------------------------------------------------------------------
# Step-type helper text
# ---------------------------------------------------------------------------

def _step_context(step_type: str) -> str:
    if step_type == "one_step":
        return ("Note: The two molecules differ in exactly ONE stereochemical feature. "
                "This is a single, targeted change.\n\n")
    else:
        return ("Note: The two molecules differ in MULTIPLE stereochemical features, "
                "requiring a multi-step repair strategy.\n\n")


# All possible stereo types shown in questions
_STEREO_TYPES_LIST = (
    "Possible Stereochemistry Types:\n"
    "- Chiral Centers: R configuration, S configuration\n"
    "- E/Z Double Bonds: E geometry, Z geometry\n\n"
    "IMPORTANT: Refer ONLY to the stereochemistry types listed above. "
    "Do NOT invent types outside this list.\n\n"
)


# ===========================================================================
# OnlyStereoQATemplate
# ===========================================================================

class OnlyStereoQATemplate:
    """QA templates for only_stereo type pairs.

    All question methods accept a `step_type` parameter:
      'one_step'   -> one stereo difference, simpler phrasing
      'multi_step' -> multiple stereo differences, multi-step emphasis
    """

    @staticmethod
    def get_dataset_context(dataset_name: Optional[str] = None,
                            endpoint: Optional[str] = None) -> str:
        return get_dataset_context(dataset_name, endpoint)

    # ===================================================================
    # Task 1 – Stereo Identification
    # ===================================================================
    @staticmethod
    def create_stereo_identification_question(
        tx_mol: str,
        nt_mol: str,
        mol_format: str,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
    ) -> str:
        """Ask which stereochemical feature(s) cause the toxicity difference."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        if step_type == "one_step":
            task_desc = (
                "Identify the single stereochemical feature most likely responsible "
                "for the toxicity difference between these two molecules."
            )
        else:
            task_desc = (
                "Identify all stereochemical features responsible for the toxicity "
                "difference between these two molecules. "
                "Multiple stereo changes are involved."
            )

        body = (
            f"{task_desc}\n\n"
            f"Toxic_{mol_format.upper()}: {tx_mol}\n"
            f"NonToxic_{mol_format.upper()}: {nt_mol}\n\n"
            f"{step_note}"
            f"{_STEREO_TYPES_LIST}"
            "Output STRICTLY in JSON:\n"
            "{\n"
            '  "Toxic_Stereochemistry_Features": ["<feature_type>", ...]\n'
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_stereo_identification_answer(
        toxic_unique_stereo: List[str],
    ) -> str:
        return json.dumps(
            {"Toxic_Stereochemistry_Features": toxic_unique_stereo or []},
            ensure_ascii=False,
        )

    # ===================================================================
    # Task 2 – Stereo Toxic Site Localization
    # ===================================================================
    @staticmethod
    def create_stereo_localization_question(
        tx_mol: str,
        mol_format: str,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
    ) -> str:
        """Ask for atom-level locations of toxic stereo features."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        # Add atom numbers to SMILES for localization task
        add_atoms = _get_add_atom_numbers()
        display_mol = tx_mol
        if mol_format.lower() == "smiles":
            numbered = add_atoms(tx_mol)
            if numbered:
                display_mol = numbered

        if step_type == "one_step":
            task_desc = (
                "Locate the single stereochemical site in the toxic molecule "
                "responsible for its toxicity. Provide atom index(es)."
            )
        else:
            task_desc = (
                "Locate ALL stereochemical sites in the toxic molecule responsible "
                "for its toxicity. Multiple sites must be identified."
            )

        atom_note = (
            "Atom indices appear as [C:1], [N:2], etc. in the SMILES above. "
            "Use these indices in your response.\n\n"
        ) if mol_format.lower() == "smiles" else ""

        body = (
            f"{task_desc}\n\n"
            f"Toxic_{mol_format.upper()}: {display_mol}\n\n"
            f"{atom_note}"
            f"{step_note}"
            f"{_STEREO_TYPES_LIST}"
            "Output STRICTLY in JSON:\n"
            "{\n"
            '  "Toxic_Stereo_Sites": [\n'
            '    {"Feature_Type": "R|S|E|Z", "Atom_Indices": [...]},\n'
            '    ...\n'
            "  ]\n"
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_stereo_localization_answer(
        toxic_stereochemistry: Dict[str, Any],
        nontoxic_stereochemistry: Dict[str, Any],
    ) -> str:
        sites = []
        tx_chiral = (toxic_stereochemistry.get("chiral_centers", {})
                     .get("chiral_centers", []) or [])
        nt_chiral = (nontoxic_stereochemistry.get("chiral_centers", {})
                     .get("chiral_centers", []) or [])
        nt_map = {c.get("atom_idx"): c.get("chirality", "")
                  for c in nt_chiral if isinstance(c, dict)}

        for c in tx_chiral:
            if not isinstance(c, dict):
                continue
            atom_idx = c.get("atom_idx")
            chir = c.get("chirality", "") or c.get("config", "")
            nt_chir = nt_map.get(atom_idx, "")
            if atom_idx not in nt_map or chir != nt_chir:
                config = "R" if "R" in str(chir) else ("S" if "S" in str(chir) else chir)
                sites.append({"Feature_Type": config, "Atom_Indices": [atom_idx]})

        tx_ez = toxic_stereochemistry.get("ez_bonds", {}).get("ez_bonds", []) or []
        nt_ez = nontoxic_stereochemistry.get("ez_bonds", {}).get("ez_bonds", []) or []
        nt_ez_map = {}
        for b in nt_ez:
            if isinstance(b, dict):
                key = tuple(b.get("bond", ())) if isinstance(b.get("bond"), list) else b.get("bond", "")
                nt_ez_map[key] = b.get("geometry", "")

        for b in tx_ez:
            if not isinstance(b, dict):
                continue
            bond = b.get("bond", "")
            key = tuple(bond) if isinstance(bond, list) else bond
            geom = b.get("geometry", "")
            if key not in nt_ez_map or nt_ez_map[key] != geom:
                indices = list(bond) if isinstance(bond, (list, tuple)) else [bond]
                sites.append({"Feature_Type": geom, "Atom_Indices": indices})

        return json.dumps({"Toxic_Stereo_Sites": sites}, ensure_ascii=False)

    # ===================================================================
    # Task 3 – Stereo Repair Planning
    # ===================================================================
    @staticmethod
    def create_stereo_repair_planning_question(
        tx_mol: str,
        mol_format: str,
        stereo_summary: str,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
    ) -> str:
        """Ask LLM to design a stereochemical repair plan."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        atom_note = (
            "Atom indices appear as [C:1], [N:2], etc. "
            "Reference them in your plan but output SMILES WITHOUT atom numbers.\n\n"
        ) if mol_format.lower() == "smiles" else ""

        if step_type == "one_step":
            task_desc = (
                "The following toxic molecule has a single stereochemical feature causing toxicity. "
                "Design a repair plan to modify this one stereochemical feature."
            )
        else:
            task_desc = (
                "The following toxic molecule has multiple stereochemical features causing toxicity. "
                "Design a multi-step repair plan to modify all relevant stereo features."
            )

        body = (
            f"{task_desc}\n\n"
            f'Toxic_{mol_format.upper()}: "{tx_mol}"\n\n'
            f"{atom_note}"
            f"{step_note}"
            f"{_STEREO_TYPES_LIST}"
            "Observed stereochemical profile (toxic molecule):\n"
            f"{stereo_summary}\n\n"
            "Create a plan by:\n"
            "- Identifying which chiral centers or E/Z bonds should be modified.\n"
            "- Proposing what the target configuration should be.\n"
            "- Briefly explaining why the change reduces toxicity (2-3 sentences).\n\n"
            "Output STRICTLY in JSON:\n"
            "{\n"
            '  "Removed_Stereochemical_Features": [\n'
            '    {"Type": "Chiral_Center|E_Z_Bond", "Atom_Index": <n>, '
            '"Current_Config": "R|S|E|Z", "Rationale": "..."},\n'
            "    ...\n"
            "  ],\n"
            '  "Added_Stereochemical_Features": [\n'
            '    {"Type": "Chiral_Center|E_Z_Bond", "Atom_Index": <n>, '
            '"Desired_Config": "R|S|E|Z", "Rationale": "..."},\n'
            "    ...\n"
            "  ],\n"
            '  "Repair_Strategy": "<brief explanation>"\n'
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_stereo_repair_planning_answer(
        comparison: Dict[str, Any],
    ) -> str:
        toxic_stereo = comparison.get("toxic_stereochemistry") or {}
        nontoxic_stereo = comparison.get("nontoxic_stereochemistry") or {}

        def _normalize(v: str) -> str:
            if not v:
                return "unspecified"
            return v.replace("@", "").strip().split("/")[0].upper() or "unspecified"

        removed = []
        added = []

        tx_chiral = toxic_stereo.get("chiral_centers", {}).get("chiral_centers", []) or []
        nt_chiral = nontoxic_stereo.get("chiral_centers", {}).get("chiral_centers", []) or []
        nt_map = {c.get("atom_idx"): c for c in nt_chiral if isinstance(c, dict)}

        for c in tx_chiral:
            if not isinstance(c, dict):
                continue
            atom_idx = c.get("atom_idx")
            chir = _normalize(c.get("chirality", "") or c.get("config", ""))
            nt = nt_map.get(atom_idx)
            nt_chir = _normalize(nt.get("chirality", "") or nt.get("config", "")) if nt else None
            if nt is None or chir != nt_chir:
                removed.append({
                    "Type": "Chiral_Center",
                    "Atom_Index": atom_idx,
                    "Current_Config": chir,
                    "Rationale": f"Chiral center at atom {atom_idx} ({chir}) differs from non-toxic"
                })
                if nt:
                    added.append({
                        "Type": "Chiral_Center",
                        "Atom_Index": atom_idx,
                        "Desired_Config": nt_chir,
                        "Rationale": f"Change to {nt_chir} as in non-toxic molecule"
                    })

        tx_ez = toxic_stereo.get("ez_bonds", {}).get("ez_bonds", []) or []
        nt_ez = nontoxic_stereo.get("ez_bonds", {}).get("ez_bonds", []) or []
        nt_ez_map = {}
        for b in nt_ez:
            if isinstance(b, dict):
                key = tuple(b.get("bond", ())) if isinstance(b.get("bond"), list) else b.get("bond", "")
                nt_ez_map[key] = b.get("geometry", "")

        for b in tx_ez:
            if not isinstance(b, dict):
                continue
            bond = b.get("bond", "")
            key = tuple(bond) if isinstance(bond, list) else bond
            geom = b.get("geometry", "")
            nt_geom = nt_ez_map.get(key)
            if nt_geom is None or geom != nt_geom:
                removed.append({
                    "Type": "E_Z_Bond",
                    "Atom_Index": list(bond) if isinstance(bond, (list, tuple)) else bond,
                    "Current_Config": geom,
                    "Rationale": f"E/Z bond {bond} ({geom}) differs"
                })
                if nt_geom:
                    added.append({
                        "Type": "E_Z_Bond",
                        "Atom_Index": list(bond) if isinstance(bond, (list, tuple)) else bond,
                        "Desired_Config": nt_geom,
                        "Rationale": f"Change to {nt_geom}"
                    })

        strategy = (
            f"Modify {len(removed)} stereochemical feature(s) to match non-toxic molecule configuration."
        )
        return json.dumps({
            "Removed_Stereochemical_Features": removed,
            "Added_Stereochemical_Features": added,
            "Repair_Strategy": strategy,
        }, ensure_ascii=False)

    # ===================================================================
    # Task 4 – Stereo Repair Gen (single SMILES)
    # ===================================================================
    @staticmethod
    def create_stereo_repair_gen_question(
        tx_mol: str,
        mol_format: str,
        stereo_explanation: str,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
        use_atom_numbering: bool = True,
    ) -> str:
        """Ask LLM to produce one repaired SMILES by modifying stereo features."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        display_mol = tx_mol
        atom_note = ""
        if use_atom_numbering and mol_format.lower() == "smiles":
            add_atoms = _get_add_atom_numbers()
            numbered = add_atoms(tx_mol)
            if numbered:
                display_mol = numbered
            atom_note = (
                "Atom indices appear as [C:1], [N:2], etc. Use them to identify "
                "which stereo features to modify. Your output SMILES must NOT contain atom numbers.\n\n"
            )

        if step_type == "one_step":
            intro = "Repair the following toxic molecule by modifying the single stereochemical feature responsible for toxicity."
        else:
            intro = "Repair the following toxic molecule by modifying ALL stereochemical features responsible for toxicity."

        body = (
            f"{intro}\n\n"
            f'Toxic_{mol_format.upper()}: "{display_mol}"\n\n'
            f"{atom_note}"
            f"{step_note}"
            "Stereochemical Information:\n"
            f"{stereo_explanation}\n\n"
            "Output a single valid SMILES (no atom numbers).\n\n"
            "Output STRICTLY in JSON:\n"
            "{\n"
            '  "Repaired_SMILES": "<SMILES>"\n'
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_stereo_repair_gen_answer(nt_mol: str) -> str:
        return json.dumps({"Repaired_SMILES": nt_mol}, ensure_ascii=False)

    # ===================================================================
    # Task 5 – Stereo Remove Add Repair Gen (3 SMILES)
    # ===================================================================
    @staticmethod
    def create_stereo_remove_add_repair_gen_question(
        tx_mol: str,
        mol_format: str,
        stereo_explanation: str,
        dataset_name: Optional[str] = None,
        endpoint: Optional[str] = None,
        step_type: str = "one_step",
        use_atom_numbering: bool = True,
    ) -> str:
        """Ask LLM to produce 3 repaired SMILES (multi-SMILES, stereo-guided)."""
        ctx = get_dataset_context(dataset_name, endpoint)
        step_note = _step_context(step_type)

        display_mol = tx_mol
        atom_note = ""
        if use_atom_numbering and mol_format.lower() == "smiles":
            add_atoms = _get_add_atom_numbers()
            numbered = add_atoms(tx_mol)
            if numbered:
                display_mol = numbered
            atom_note = (
                "Atom indices appear as [C:1], [N:2], etc. "
                "Output SMILES must NOT contain atom numbers.\n\n"
            )

        if step_type == "one_step":
            intro = (
                "Repair the toxic molecule by modifying the single stereochemical feature. "
                "Generate THREE different SMILES, each using a different modification strategy."
            )
        else:
            intro = (
                "Repair the toxic molecule by modifying ALL stereochemical features. "
                "Generate THREE different SMILES, each exploring different combinations "
                "of the required stereo changes."
            )

        body = (
            f"{intro}\n\n"
            f'Toxic_{mol_format.upper()}: "{display_mol}"\n\n'
            f"{atom_note}"
            f"{step_note}"
            "Stereochemical Information:\n"
            f"{stereo_explanation}\n\n"
            "Instructions:\n"
            "- Generate THREE chemically valid, parseable SMILES (no atom numbers).\n"
            "- Focus on R/S chiral center and E/Z bond geometry modifications.\n"
            "- Each SMILES represents a distinct repair strategy.\n\n"
            "Output STRICTLY in JSON:\n"
            "{\n"
            '  "Repaired_SMILES": ["<SMILES1>", "<SMILES2>", "<SMILES3>"]\n'
            "}\n"
        )
        return (ctx + "\n\n" + body) if ctx else body

    @staticmethod
    def create_stereo_remove_add_repair_gen_answer(nt_mol: str) -> str:
        return json.dumps({"Repaired_SMILES": nt_mol}, ensure_ascii=False)

    # ===================================================================
    # Convenience wrappers matching existing task-file call patterns
    # ===================================================================
    @staticmethod
    def _create_stereo_identification_question(tx_mol, nt_mol, mol_format,
                                               dataset_name=None, subset=None,
                                               step_type="one_step"):
        return OnlyStereoQATemplate.create_stereo_identification_question(
            tx_mol, nt_mol, mol_format, dataset_name, subset, step_type)

    @staticmethod
    def _create_stereo_identification_answer(toxic_unique_stereo):
        return OnlyStereoQATemplate.create_stereo_identification_answer(toxic_unique_stereo)

    @staticmethod
    def _create_stereo_toxic_site_localization_question(tx_mol, mol_format,
                                                        dataset_name=None, subset=None,
                                                        step_type="one_step"):
        return OnlyStereoQATemplate.create_stereo_localization_question(
            tx_mol, mol_format, dataset_name, subset, step_type)

    @staticmethod
    def _create_stereo_toxic_site_localization_answer(toxic_stereochemistry,
                                                       nontoxic_stereochemistry):
        return OnlyStereoQATemplate.create_stereo_localization_answer(
            toxic_stereochemistry, nontoxic_stereochemistry)

    @staticmethod
    def _create_stereo_repair_planning_question(tx_mol, mol_format, stereo_summary,
                                                dataset_name=None, subset=None,
                                                step_type="one_step"):
        return OnlyStereoQATemplate.create_stereo_repair_planning_question(
            tx_mol, mol_format, stereo_summary, dataset_name, subset, step_type)

    @staticmethod
    def _create_stereo_repair_planning_answer(comparison):
        return OnlyStereoQATemplate.create_stereo_repair_planning_answer(comparison)

    @staticmethod
    def _create_stereo_repair_question(tx_mol, nt_mol, mol_format, stereo_explanation,
                                       dataset_name=None, subset=None,
                                       use_atom_numbering=True, step_type="one_step"):
        return OnlyStereoQATemplate.create_stereo_repair_gen_question(
            tx_mol, mol_format, stereo_explanation, dataset_name, subset,
            step_type, use_atom_numbering)

    @staticmethod
    def _create_stereo_repair_answer(nt_mol):
        return OnlyStereoQATemplate.create_stereo_repair_gen_answer(nt_mol)

    @staticmethod
    def _create_stereo_remove_add_repair_question(tx_mol, nt_mol, mol_format,
                                                   stereo_explanation,
                                                   dataset_name=None, subset=None,
                                                   use_atom_numbering=True,
                                                   step_type="one_step"):
        return OnlyStereoQATemplate.create_stereo_remove_add_repair_gen_question(
            tx_mol, mol_format, stereo_explanation, dataset_name, subset,
            step_type, use_atom_numbering)

    @staticmethod
    def _create_stereo_remove_add_repair_answer(nt_mol):
        return OnlyStereoQATemplate.create_stereo_remove_add_repair_gen_answer(nt_mol)


# ---------------------------------------------------------------------------
# Backward-compat alias
# ---------------------------------------------------------------------------
StereoQATemplate = OnlyStereoQATemplate
