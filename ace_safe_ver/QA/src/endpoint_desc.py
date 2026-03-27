"""Endpoint descriptions for QA generation.
This module provides endpoint-specific descriptions for various datasets.
Based on Mol_FG/qa_builder.py _get_dataset_context function.
"""
from typing import Optional, Dict


def get_dataset_context(dataset_name: Optional[str] = None, endpoint: Optional[str] = None) -> str:
    """Get dataset-specific context description.
    
    Args:
        dataset_name: Name of dataset ('dilist', 'dictrank', 'diril', 'tox21', 'sider_train', 'sider_test', etc.)
        endpoint: Endpoint name (for Tox21: 'NR-AR', 'NR-ER-LBD', etc. For SIDER: 'Blood and lymphatic system disorders', etc.)
                  Can also be in format 'tox21_NR-AR' or 'herg', 'ames', etc.
        
    Returns:
        Context string to prepend to questions, or empty string if not needed
    """
    if not dataset_name and not endpoint:
        return ""
    
    # Handle cases where endpoint contains dataset name (e.g., 'tox21_NR-AR', 'herg', 'ames')
    if endpoint:
        endpoint_lower = endpoint.lower()
        
        # Check if endpoint starts with dataset prefix
        if endpoint_lower.startswith('tox21_'):
            # Extract actual endpoint name
            actual_endpoint = endpoint.replace('tox21_', '')
            return get_tox21_endpoint_description(actual_endpoint)
        
        # Check if endpoint is a SIDER endpoint
        sider_endpoints = [
            "blood and lymphatic system disorders", "cardiac disorders",
            "congenital, familial and genetic disorders", "ear and labyrinth disorders",
            "eye disorders", "general disorders and administration site conditions",
            "hepatobiliary disorders", "immune system disorders",
            "infections and infestations", "injury, poisoning and procedural complications",
            "investigations", "metabolism and nutrition disorders",
            "musculoskeletal and connective tissue disorders",
            "neoplasms benign, malignant and unspecified (incl cysts and polyps)",
            "nervous system disorders", "pregnancy, puerperium and perinatal conditions",
            "product issues", "psychiatric disorders",
            "renal and urinary disorders", "reproductive system and breast disorders",
            "respiratory, thoracic and mediastinal disorders",
            "skin and subcutaneous tissue disorders", "social circumstances",
            "surgical and medical procedures", "vascular disorders"
        ]
        if endpoint_lower in sider_endpoints:
            return get_sider_endpoint_description(endpoint)
        
        # elif endpoint_lower == 'herg':
        #     return (
        #         "The following molecule blocks the hERG (human Ether-à-go-go-Related Gene) channel, "
        #         "which is crucial for the coordination of the heart's beating. "
        #         "Blocking the hERG channel can lead to severe adverse effects, including cardiac arrhythmias and sudden cardiac death."
        #     )
        # elif endpoint_lower == 'herg_inhib':
        #     return (
        #         "The following molecule blocks the hERG (human Ether-à-go-go-Related Gene) channel, "
        #         "which is crucial for the coordination of the heart's beating. "
        #         "This dataset evaluates hERG inhibition at multiple concentrations (1uM, 10uM) and can lead to cardiac arrhythmias and sudden cardiac death."
        #     )
        elif endpoint_lower == 'herg_unified':
            return (
                "The following molecule has been evaluated for hERG (human Ether-à-go-go-Related Gene) channel blocking activity "
                "under a unified hERG toxicity endpoint that combines data from the hERG, hERG inhibition, and hERG Karim sources. "
                "Blockade of the hERG channel can disrupt cardiac repolarization and lead to serious adverse effects, "
                "including cardiac arrhythmias and sudden cardiac death."
            )
        # elif endpoint_lower == 'herg_karim':
        #     return (
        #         "The following molecule has been evaluated for hERG (human Ether-à-go-go-Related Gene) channel blocking activity. "
        #         "This dataset consists of hERG blockers (<10uM) and non-hERG blockers (>=10uM) from integrated sources. "
        #         "Blocking the hERG channel can lead to cardiac arrhythmias and sudden cardiac death."
        #     )
        elif endpoint_lower == 'ames':
            return (
                "The following molecule is mutagenic, meaning it can cause genetic alterations and DNA damage that may lead to cell death or severe adverse effects."
            )
        elif endpoint_lower == 'clintox':
            return (
                "The following molecule has been associated with clinical toxicity, including drugs that have failed clinical trials due to toxicity reasons."
            )
        elif endpoint_lower == 'dilist':
            return (
                "The following molecule is highly likely to cause human liver injury, or actual cases of liver injury have been reported and confirmed."
            )
        elif endpoint_lower == 'cyp1a2_veith':
            return (
                "The following molecule inhibits CYP P450 1A2 (Veith et al.). "
                "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                "Specifically, CYP1A2 localizes to the endoplasmic reticulum and its expression can be induced by polycyclic aromatic hydrocarbons (PAHs), "
                "some of which are found in cigarette smoke. It can metabolize PAHs to carcinogenic intermediates and also processes xenobiotics such as "
                "caffeine, aflatoxin B1, and acetaminophen. Inhibition can reduce drug metabolism and increase drug-drug interaction risk."
            )
        elif endpoint_lower == 'cyp2c19_veith':
            return (
                "The following molecule inhibits CYP P450 2C19 (Veith et al.). "
                "The CYP P450 genes are essential for the breakdown (metabolism) of various molecules and chemicals within cells. "
                "Inhibiting these enzymes can lead to poor metabolism of this drug and co-administered drugs, increasing the risk of "
                "drug-drug interactions and adverse effects. CYP2C19 is associated with endoplasmic reticulum functions related to protein processing and transport."
            )
        elif endpoint_lower == 'cyp2c9_veith':
            return (
                "The following molecule inhibits CYP P450 2C9 (Veith et al.). "
                "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                "Specifically, CYP2C9 plays a major role in oxidation of both xenobiotic and endogenous compounds. "
                "Inhibition can impair metabolic clearance and increase adverse event risk."
            )
        elif endpoint_lower == 'cyp2d6_veith':
            return (
                "The following molecule inhibits CYP P450 2D6 (Veith et al.). "
                "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                "CYP2D6 is primarily expressed in the liver and is also highly expressed in regions of the central nervous system, including the substantia nigra. "
                "Inhibition can alter metabolic clearance and increase potential toxicity or interaction risk."
            )
        elif endpoint_lower == 'cyp3a4_veith':
            return (
                "The following molecule inhibits CYP P450 3A4 (Veith et al.). "
                "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                "CYP3A4 is an important enzyme mainly found in the liver and intestine, and oxidizes many foreign organic molecules (xenobiotics), "
                "including toxins and drugs, to support elimination. Inhibition can reduce clearance and increase drug-drug interaction risk."
            )
    
    if not dataset_name:
        return ""
    
    dataset_name_lower = dataset_name.lower()
    
    # Dataset-level descriptions (no endpoint needed)
    if dataset_name_lower == "dilist":
        return (
            "The following molecule is highly likely to cause human liver injury, or actual cases of liver injury have been reported and confirmed."
        )
    elif dataset_name_lower == "dictrank":
        return (
            "The following molecule is highly likely to cause human cardiotoxicity, or actual cases of cardiotoxicity have been reported and confirmed."
        )
    elif dataset_name_lower == "diril":
        return (
            "The following molecule is highly likely to cause human renal toxicity, or actual cases of renal toxicity have been reported and confirmed."
        )
    elif dataset_name_lower == "ames":
        return (
            "The following molecule is mutagenic, meaning it can cause genetic alterations and DNA damage that may lead to cell death or severe adverse effects."
        )
    elif dataset_name_lower == "herg_karim":
        return (
            "The following molecule has been evaluated for hERG (human Ether-à-go-go-Related Gene) channel blocking activity. "
            "This dataset consists of hERG blockers (<10uM) and non-hERG blockers (>=10uM) from integrated sources. "
            "Blocking the hERG channel can lead to cardiac arrhythmias and sudden cardiac death."
        )
    elif dataset_name_lower == "herg" or dataset_name_lower == "herg_inhib":
        return (
            "The following molecule blocks the hERG (human Ether-à-go-go-Related Gene) channel, "
            "which is crucial for the coordination of the heart's beating. "
            "Blocking the hERG channel can lead to severe adverse effects, including cardiac arrhythmias and sudden cardiac death."
        )
    elif dataset_name_lower == "herg_unified":
        return (
            "The following molecule has been evaluated for hERG (human Ether-à-go-go-Related Gene) channel blocking activity "
            "using a unified benchmark that integrates hERG, hERG inhibition, and hERG Karim sources. "
            "Blocking the hERG channel can lead to severe adverse effects, including cardiac arrhythmias and sudden cardiac death."
        )
    elif dataset_name_lower == "skin_reaction":
        return (
            "The following molecule can cause skin sensitization, an immune reaction that leads to allergic contact dermatitis upon repeated exposure."
        )
    elif dataset_name_lower == "carcinogens_lagunin":
        return (
            "The following molecule is carcinogenic, meaning it promotes cancer formation through DNA damage or disruption of cellular metabolic processes."
        )
    elif dataset_name_lower == "clintox":
        return (
            "The following molecule has been associated with clinical toxicity, including drugs that have failed clinical trials due to toxicity reasons."
        )
    
    # Tox21 endpoint-specific contexts
    elif dataset_name_lower == "tox21":
        if endpoint:
            return get_tox21_endpoint_description(endpoint)
        else:
            return (
                "Tox21 Context: Tox21 is a dataset that evaluates chemical compounds across 12 different toxicity endpoints "
                "related to nuclear receptor pathways and stress response pathways.\n"
                "- Toxic: The compound activates or disrupts the specific toxicity pathway being evaluated.\n"
                "- Non-Toxic: The compound does not show significant activity in the evaluated toxicity pathway.\n\n"
            )
    
    # SIDER dataset endpoint-specific contexts
    elif dataset_name_lower in ["sider_train", "sider_test", "sider"]:
        if endpoint:
            return get_sider_endpoint_description(endpoint)
        else:
            return (
                "SIDER Context: SIDER (Side Effect Resource) is a dataset that contains information about "
                "recorded side effects of drugs. The following molecule has been associated with adverse drug reactions.\n\n"
            )
    elif dataset_name_lower == "metabolism":
        if endpoint:
            ep = endpoint.lower()
            if ep == "cyp1a2_veith":
                return (
                    "The following molecule inhibits CYP P450 1A2 (Veith et al.). "
                    "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                    "Specifically, CYP1A2 localizes to the endoplasmic reticulum and its expression can be induced by polycyclic aromatic hydrocarbons (PAHs), "
                    "some of which are found in cigarette smoke. It can metabolize PAHs to carcinogenic intermediates and also processes xenobiotics such as "
                    "caffeine, aflatoxin B1, and acetaminophen. Inhibition can reduce drug metabolism and increase drug-drug interaction risk."
                )
            if ep == "cyp2c19_veith":
                return (
                    "The following molecule inhibits CYP P450 2C19 (Veith et al.). "
                    "The CYP P450 genes are essential for the breakdown (metabolism) of various molecules and chemicals within cells. "
                    "Inhibiting these enzymes can lead to poor metabolism of this drug and co-administered drugs, increasing the risk of "
                    "drug-drug interactions and adverse effects. CYP2C19 is associated with endoplasmic reticulum functions related to protein processing and transport."
                )
            if ep == "cyp2c9_veith":
                return (
                    "The following molecule inhibits CYP P450 2C9 (Veith et al.). "
                    "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                    "Specifically, CYP2C9 plays a major role in oxidation of both xenobiotic and endogenous compounds. "
                    "Inhibition can impair metabolic clearance and increase adverse event risk."
                )
            if ep == "cyp2d6_veith":
                return (
                    "The following molecule inhibits CYP P450 2D6 (Veith et al.). "
                    "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                    "CYP2D6 is primarily expressed in the liver and is also highly expressed in regions of the central nervous system, including the substantia nigra. "
                    "Inhibition can alter metabolic clearance and increase potential toxicity or interaction risk."
                )
            if ep == "cyp3a4_veith":
                return (
                    "The following molecule inhibits CYP P450 3A4 (Veith et al.). "
                    "The CYP P450 genes are involved in the formation and breakdown (metabolism) of various molecules and chemicals within cells. "
                    "CYP3A4 is an important enzyme mainly found in the liver and intestine, and oxidizes many foreign organic molecules (xenobiotics), "
                    "including toxins and drugs, to support elimination. Inhibition can reduce clearance and increase drug-drug interaction risk."
                )
        return (
            "Metabolism Context: This endpoint evaluates whether the molecule inhibits key CYP450 enzymes involved in drug metabolism. "
            "Inhibition can reduce metabolic clearance, increase exposure, and cause drug-drug interactions or adverse effects."
        )
    
    return ""


def get_tox21_endpoint_description(endpoint: str) -> str:
    """Get Tox21 endpoint-specific description.
    
    Args:
        endpoint: Tox21 endpoint name (e.g., 'NR-AR', 'NR-ER-LBD')
        
    Returns:
        Endpoint-specific description string
    """
    endpoint_contexts: Dict[str, str] = {
        "NR-AR": (
            "The following molecule activates or disrupts the Androgen Receptor (AR) pathway, "
            "which regulates male sexual development and function. "
            "Disruption of this pathway can affect reproductive development and function."
        ),
        "NR-AR-LBD": (
            "The following molecule binds to the Androgen Receptor Ligand Binding Domain (AR-LBD), "
            "affecting androgen signaling pathways. "
            "This assay evaluates more direct binding mechanisms compared to the full receptor activity assay."
        ),
        "NR-AhR": (
            "The following molecule activates the Aryl Hydrocarbon Receptor (AhR) pathway, "
            "which is involved in xenobiotic metabolism and immune responses. "
            "Activation of this receptor can lead to toxic effects such as liver toxicity, carcinogenicity, and immunotoxicity."
        ),
        "NR-Aromatase": (
            "The following molecule inhibits or activates Aromatase, an enzyme essential for estrogen (female hormone) biosynthesis. "
            "This assay evaluates whether the chemical can affect aromatase enzyme activity, thereby influencing estrogen levels. "
            "Disruption of estrogen balance is important for reproductive health."
        ),
        "NR-ER": (
            "The following molecule activates or disrupts the Estrogen Receptor (ER) pathway, "
            "which regulates female sexual development and function. "
            "Disruption of this pathway can affect female reproductive development and function, and is associated with conditions such as breast cancer."
        ),
        "NR-ER-LBD": (
            "The following molecule binds to the Estrogen Receptor Ligand Binding Domain (ER-LBD), "
            "affecting estrogen signaling pathways. "
            "This assay evaluates more direct binding mechanisms compared to the full receptor activity assay."
        ),
        "NR-PPAR-gamma": (
            "The following molecule activates or disrupts the Peroxisome Proliferator-Activated Receptor gamma (PPAR-gamma) pathway, "
            "which regulates glucose and lipid metabolism, cell differentiation, and inflammatory responses. "
            "This assay evaluates whether the chemical can activate PPAR-gamma, potentially affecting metabolic diseases such as diabetes and obesity."
        ),
        "SR-ARE": (
            "The following molecule activates the Antioxidant Response Element (ARE) pathway, "
            "which regulates cellular antioxidant defense mechanisms. "
            "This assay evaluates whether the chemical can activate or inhibit the cell's antioxidant defense system in response to oxidative stress."
        ),
        "SR-ATAD5": (
            "The following molecule affects ATAD5 (ATPase family AAA domain-containing protein 5), "
            "which plays an important role in DNA damage response and repair. "
            "This assay evaluates whether the chemical can affect the ATAD5 pathway, potentially causing DNA damage and genomic instability issues."
        ),
        "SR-HSE": (
            "The following molecule activates the Heat Shock Response Element (HSE) pathway, "
            "which responds to cellular stress and protein misfolding. "
            "Cells respond to protein denaturation stress (heat, toxic substances, etc.) by inducing the production of heat shock proteins (HSP) to repair damaged proteins. "
            "This assay evaluates whether the chemical disrupts the cell's protein quality control system."
        ),
        "SR-MMP": (
            "The following molecule affects Mitochondrial Membrane Potential (MMP), "
            "which is an important indicator of mitochondrial functional status. "
            "Mitochondria are the cell's energy production factories. "
            "This assay evaluates whether the chemical can damage mitochondrial function, potentially causing problems with cellular energy production, which is one of the important mechanisms of cell toxicity."
        ),
        "SR-p53": (
            "The following molecule activates or disrupts the p53 pathway, a critical tumor suppressor pathway involved in cell cycle control and apoptosis. "
            "p53 is known as the 'guardian of the genome' and responds to DNA damage and cellular stress by inducing cell cycle arrest, DNA repair, and apoptosis (cell death). "
            "This assay evaluates whether the chemical can affect the p53 pathway, potentially causing DNA damage, cell death, or cancer development."
        ),
    }
    
    return endpoint_contexts.get(
        endpoint,
        "The following molecule activates or disrupts a specific toxicity pathway being evaluated in the Tox21 dataset."
    )


def get_sider_endpoint_description(endpoint: str) -> str:
    """Get SIDER endpoint-specific description.
    
    Args:
        endpoint: SIDER endpoint name (e.g., 'Blood and lymphatic system disorders', 'Cardiac disorders')
        
    Returns:
        Endpoint-specific description string
    """
    endpoint_descriptions: Dict[str, str] = {
        "Blood and lymphatic system disorders": (
            "The following molecule has been associated with blood and lymphatic system disorders, "
            "which can include conditions affecting blood cells, clotting mechanisms, or lymphatic circulation. "
            "These disorders may manifest as anemia, bleeding disorders, or immune system complications."
        ),
        "Cardiac disorders": (
            "The following molecule has been associated with cardiac disorders, "
            "which can include arrhythmias, heart failure, myocardial infarction, or other cardiovascular complications. "
            "These conditions can significantly impact heart function and overall cardiovascular health."
        ),
        "Congenital, familial and genetic disorders": (
            "The following molecule has been associated with congenital, familial, and genetic disorders, "
            "which may involve birth defects, inherited conditions, or genetic mutations. "
            "These disorders can affect development, growth, or long-term health outcomes."
        ),
        "Ear and labyrinth disorders": (
            "The following molecule has been associated with ear and labyrinth disorders, "
            "which can include hearing loss, tinnitus, vertigo, or balance problems. "
            "These conditions can affect auditory function and spatial orientation."
        ),
        "Eye disorders": (
            "The following molecule has been associated with eye disorders, "
            "which can include vision impairment, retinal damage, cataracts, or other ocular complications. "
            "These conditions can significantly impact visual function and quality of life."
        ),
        "General disorders and administration site conditions": (
            "The following molecule has been associated with general disorders and administration site conditions, "
            "which can include injection site reactions, systemic reactions, or general malaise. "
            "These conditions may occur at the site of drug administration or manifest as systemic effects."
        ),
        "Hepatobiliary disorders": (
            "The following molecule has been associated with hepatobiliary disorders, "
            "which can include liver damage, hepatitis, cholestasis, or other liver and bile duct complications. "
            "These conditions can significantly impact liver function and metabolic processes."
        ),
        "Immune system disorders": (
            "The following molecule has been associated with immune system disorders, "
            "which can include autoimmune reactions, hypersensitivity, immunosuppression, or other immune-related complications. "
            "These conditions can affect the body's ability to fight infections or maintain immune homeostasis."
        ),
        "Infections and infestations": (
            "The following molecule has been associated with infections and infestations, "
            "which may indicate increased susceptibility to infections or direct infectious complications. "
            "These conditions can result from immunosuppression or other mechanisms that compromise immune defenses."
        ),
        "Injury, poisoning and procedural complications": (
            "The following molecule has been associated with injury, poisoning, and procedural complications, "
            "which can include accidental overdoses, drug interactions, or complications from medical procedures. "
            "These conditions may result from improper use, dosage errors, or adverse interactions."
        ),
        "Investigations": (
            "The following molecule has been associated with abnormal laboratory findings or investigations, "
            "which can include changes in blood chemistry, liver enzymes, kidney function markers, or other diagnostic parameters. "
            "These findings may indicate underlying organ dysfunction or metabolic disturbances."
        ),
        "Metabolism and nutrition disorders": (
            "The following molecule has been associated with metabolism and nutrition disorders, "
            "which can include diabetes, electrolyte imbalances, metabolic syndrome, or nutritional deficiencies. "
            "These conditions can affect energy metabolism, glucose regulation, or nutrient absorption."
        ),
        "Musculoskeletal and connective tissue disorders": (
            "The following molecule has been associated with musculoskeletal and connective tissue disorders, "
            "which can include muscle weakness, joint pain, bone disorders, or connective tissue damage. "
            "These conditions can affect mobility, strength, and structural integrity of the musculoskeletal system."
        ),
        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)": (
            "The following molecule has been associated with neoplasms (tumors), including benign, malignant, and unspecified growths, "
            "as well as cysts and polyps. These conditions involve abnormal cell growth and may indicate carcinogenic potential or tumor-promoting effects."
        ),
        "Nervous system disorders": (
            "The following molecule has been associated with nervous system disorders, "
            "which can include neurotoxicity, seizures, cognitive impairment, or other neurological complications. "
            "These conditions can affect brain function, peripheral nerves, or overall neurological health."
        ),
        "Pregnancy, puerperium and perinatal conditions": (
            "The following molecule has been associated with pregnancy, puerperium, and perinatal conditions, "
            "which can include complications during pregnancy, childbirth, or the postpartum period. "
            "These conditions can affect maternal health, fetal development, or neonatal outcomes."
        ),
        "Product issues": (
            "The following molecule has been associated with product issues, "
            "which can include quality problems, contamination, or manufacturing defects. "
            "These issues may affect drug safety, efficacy, or stability."
        ),
        "Psychiatric disorders": (
            "The following molecule has been associated with psychiatric disorders, "
            "which can include depression, anxiety, psychosis, mood changes, or other mental health complications. "
            "These conditions can significantly impact cognitive function, emotional well-being, and behavioral patterns."
        ),
        "Renal and urinary disorders": (
            "The following molecule has been associated with renal and urinary disorders, "
            "which can include kidney damage, renal failure, urinary tract complications, or other nephrotoxic effects. "
            "These conditions can significantly impact kidney function and fluid-electrolyte balance."
        ),
        "Reproductive system and breast disorders": (
            "The following molecule has been associated with reproductive system and breast disorders, "
            "which can include hormonal imbalances, fertility issues, reproductive organ complications, or breast-related conditions. "
            "These conditions can affect reproductive health, fertility, or hormonal regulation."
        ),
        "Respiratory, thoracic and mediastinal disorders": (
            "The following molecule has been associated with respiratory, thoracic, and mediastinal disorders, "
            "which can include breathing difficulties, lung damage, respiratory infections, or other pulmonary complications. "
            "These conditions can significantly impact respiratory function and oxygen exchange."
        ),
        "Skin and subcutaneous tissue disorders": (
            "The following molecule has been associated with skin and subcutaneous tissue disorders, "
            "which can include rashes, dermatitis, skin irritation, or other dermatological complications. "
            "These conditions can affect skin integrity, appearance, or protective function."
        ),
        "Social circumstances": (
            "The following molecule has been associated with social circumstances, "
            "which may indicate impacts on social functioning, relationships, or daily activities. "
            "These effects may result from physical or psychological side effects that affect quality of life."
        ),
        "Surgical and medical procedures": (
            "The following molecule has been associated with complications from surgical and medical procedures, "
            "which can include adverse reactions during or after medical interventions. "
            "These complications may result from drug interactions, procedural risks, or patient-specific factors."
        ),
        "Vascular disorders": (
            "The following molecule has been associated with vascular disorders, "
            "which can include blood vessel damage, thrombosis, hypertension, or other circulatory complications. "
            "These conditions can affect blood flow, vascular integrity, or cardiovascular function."
        ),
    }
    
    # Return specific description if available, otherwise return generic SIDER description
    return endpoint_descriptions.get(
        endpoint,
        f"The following molecule has been associated with {endpoint}, which indicates potential adverse effects or toxicity related to this condition."
    )


def get_endpoint_description(dataset_name: Optional[str] = None, endpoint: Optional[str] = None) -> str:
    """Get endpoint description for a given dataset and endpoint.
    
    This is a convenience function that calls get_dataset_context.
    
    Args:
        dataset_name: Name of dataset
        endpoint: Endpoint name
        
    Returns:
        Endpoint-specific description string
    """
    return get_dataset_context(dataset_name=dataset_name, endpoint=endpoint)


if __name__ == "__main__":
    # Test the functions
    print("Testing endpoint descriptions...\n")
    
    # Test Tox21 endpoints
    print("Tox21 Endpoints:")
    print(get_tox21_endpoint_description("NR-AR"))
    print("\n" + "="*80 + "\n")
    
    # Test SIDER endpoints
    print("SIDER Endpoints:")
    print(get_sider_endpoint_description("Cardiac disorders"))
    print("\n" + "="*80 + "\n")
    
    # Test dataset context
    print("Dataset Context:")
    print(get_dataset_context("sider_train", "Cardiac disorders"))
    print("\n" + "="*80 + "\n")
    
    print("All tests completed!")
