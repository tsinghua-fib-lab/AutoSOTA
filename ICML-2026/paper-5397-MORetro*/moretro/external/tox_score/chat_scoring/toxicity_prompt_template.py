"""
Toxicity Assessment Prompt Template
This file contains the main prompt template for ChatGPT toxicity scoring.
"""

TOXICITY_PROMPT_TEMPLATE = """You are an expert toxicologist and chemist tasked with evaluating the toxicity of chemical compounds based on their SMILES notation.

TASK: Analyze the given SMILES and provide a toxicity score between 0 and 1, where:
- 0.0 = Non-toxic (safe for human exposure, minimal environmental impact)
- 0.1-0.3 = Low toxicity (minor health concerns, limited environmental impact)
- 0.4-0.6 = Moderate toxicity (significant health concerns, moderate environmental impact)
- 0.7-0.9 = High toxicity (serious health hazards, significant environmental impact)
- 1.0 = Extremely toxic (lethal or severely hazardous, major environmental damage)

Note that the SMILES can either be catalyst (with ligands), solvents, acids, bases, salts or other agents
The catalysts are usually those elements with a transition metal in the center.

IMPORTANT: ALWAYS check the REFERENCE STUDIES TO CONSIDER section first. If you find an exact match or similar structure in my custom research papers, you MUST use that data for your assessment and explicitly mention it in your explanation. If no match is found, then use general toxicological knowledge.

SPECIAL INSTRUCTION FOR TRANSITION METAL CATALYSTS: If the SMILES contains a transition metal that is listed in the custom research papers (catalyst greenness scores), use the provided score for that metal as a baseline and analyze it together with the ligands to propose a final toxicity score. Consider how the ligands might modify the metal's toxicity (e.g., organic ligands may increase bioavailability, while some ligands may reduce toxicity through chelation effects).

EVALUATION CRITERIA (in order of priority):
1. **FIRST**: Check my provided custom papers in REFERENCE STUDIES TO CONSIDER for exact matches or similar compounds
2. Acute toxicity (LD50, LC50 values)
3. Environmental impact (bioaccumulation, persistence, ecotoxicity)
4. Known hazard classifications (GHS categories, OSHA standards)
5. Structural alerts for toxicity (reactive groups, metabolic activation pathways)

ADDITIONAL RESOURCES TO CONSIDER:
- EPA ToxCast database
- OECD QSAR Toolbox guidelines
- PubChem bioassay data
- Structure-activity relationships for similar compounds
- Known mechanisms of toxicity for chemical classes

REFERENCE STUDIES TO CONSIDER:
{custom_papers_section}

EXAMPLES OF HOW TO USE CUSTOM RESEARCH DATA:
- If analyzing CCO (ethanol): "Found in custom research data as 'Recommended' by Prat et al. (2016)..."
- If analyzing CN(C)C=O (DMF): "According to the custom research data, DMF is classified as 'Problematic' in the Prat et al. solvent guide..."
- If analyzing Pd compounds: "Based on the catalyst greenness scores provided, Pd has a score of 0.75..."

RESPONSE FORMAT:
Score: [0.0-1.0]
Explanation: [Start by stating if you found this compound or similar compounds in the custom research papers. Then provide detailed 2-3 sentence explanation covering the main toxicological concerns, structural features contributing to toxicity, specific considerations for solvents/catalysts if applicable, and confidence level in the assessment. If you used custom research data, explicitly cite it (e.g., "Based on Prat et al. data..." or "According to the catalyst greenness scores...")]

SMILES to analyze: {smiles}"""
