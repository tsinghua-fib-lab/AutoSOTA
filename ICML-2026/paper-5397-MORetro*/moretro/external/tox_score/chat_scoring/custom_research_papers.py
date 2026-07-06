# Custom Research Papers for Toxicity Assessment
# Add your specific papers and key findings here

CUSTOM_RESEARCH_PAPERS = """
ADDITIONAL RESEARCH PAPERS AND FINDINGS:

1. Solvent Selection and Recommendation (Prat et al., 2016):
    - Source: Chem21 selection guide of solvents 
    - Table 7 - Common Solvent Rankings (Health/Safety/Environment scores 1-10, 10=safest):
        smiles,name,recommendation_after_discussion
        O,Water,Recommended
        CO,MeOH,Recommended
        CCO,EtOH,Recommended
        CC(C)O,i-PrOH,Recommended
        CCCCO,n-BuOH,Recommended
        CCC(O)C,s-BuOH,Recommended
        OCc1ccccc1,Benzyl alcohol,Problematic
        OCCO,Ethylene glycol,Recommended
        CC(=O)C,Acetone,Recommended
        CC(=O)CC,MEK,Recommended
        CC(=O)CC(C)C,MIBK,Recommended
        O=C1CCCCC1,Cyclohexanone,Problematic
        CC(=O)OC,Methyl acetate,Problematic
        CC(=O)OCC,Ethyl acetate,Recommended
        CC(=O)OC(C)C,i-PrOAc,Recommended
        CC(=O)OCCCC,n-BuOAc,Problematic
        CCOCC,Diethyl ether,HH
        CC(C)OC(C)C,Diisopropyl ether,HH
        COC(C)(C)C,MTBE,Hazardous
        C1CCOC1,THF,Hazardous
        CC1CCOC1,Me-THF,Problematic
        O1CCOCC1,1,4-Dioxane,Hazardous
        COc1ccccc1,Anisole,Recommended
        COCCOC,DME,Hazardous
        CCCCC,Pentane,Hazardous
        CCCCCC,Hexane,Hazardous
        CCCCCCC,Heptane,Problematic
        C1CCCCC1,Cyclohexane,Problematic
        CC1CCCCC1,Methylcyclohexane,Problematic
        c1ccccc1,Benzene,HH
        Cc1ccccc1,Toluene,Problematic
        Cc1ccc(C)cc1,Xylenes,Problematic
        C(Cl)Cl,DCM,Hazardous
        C(Cl)(Cl)Cl,Chloroform,HH
        C(Cl)(Cl)(Cl)Cl,CCl4,HH
        ClCCCl,DCE,Hazardous
        Clc1ccccc1,Chlorobenzene,Problematic
        CC#N,Acetonitrile,Problematic
        CN(C)C=O,DMF,Problematic
        CC(=O)N(C)C,DMAc,Hazardous
        O=C1N(C)CCCC1,NMP,Hazardous
        CN1C(=O)N(C)CCC1,DMPU,Problematic
        CS(=O)C,DMSO,Recommended
        O=S1(=O)CCCC1,Sulfolane,Hazardous
        O=P(N(C)C)(N(C)C)N(C)C,HMPA,Hazardous
        COCCO,Methoxy-ethanol,Hazardous
        CS2,Carbon disulfide,Hazardous
        O=[N+]([O-])c1ccccc1,Nitrobenzene,Hazardous
        O=CO,Formic acid,Problematic
        CC(=O)O,Acetic acid,Problematic
        CC(=O)OC(=O)C,Ac2O,Problematic
        c1ccncc1,Pyridine,Problematic
        CCN(CC)CC,TEA,Hazardous

        Table 8: 
        smiles,name,recommendation_after_discussion
        CCCO,i-Butanol,Recommended
        CC(C)CO,i-Amyl alcohol,Recommended
        OCCCO,1,3-Propanediol,Problematic
        OCC(O)CO,Glycerol,Problematic
        CC(=O)OCCC,i-Butyl acetate,Recommended
        CC(=O)OCC(C)C,i-Amyl acetate,Recommended
        CC(=O)OCH2OC(=O)C,Glycol diacetate,Recommended
        O=C1CCC(=O)O1,Î³-Valerolactone,Problematic
        CCOC(=O)CCOC(=O)CC,Diethyl succinate,Problematic
        CCOC(C)(C)C,TAME,Recommended
        CC1COC(C)O1,CPME,Problematic
        CCOC(C)(C)OC,ETBE,Problematic
        CC1=CC(=C(C=C1)C)C=C,D-Limonene,Problematic
        CC1=CC(=C(C=C1)C)C,Turpentine,Problematic
        CC1=CC=CC=C1,p-Cymene,Problematic
        COC(=O)OC,Dimethyl carbonate,Recommended
        O=C1OCCO1,Ethylene carbonate,Problematic
        O=C(OCC1)OC1,Propylene carbonate,Problematic
        O=C1CCOC1C2CC2,Cyrene,Problematic
        CC(=O)OCC,Ethyl lactate,Problematic
        CC(O)C(=O)O,Lactic acid,Problematic
        OCC1=CC=CO1,TH-Furfuryl alcohol,Hazardous


2. Catalyst Greeness Studies (Brystrzanowska et. al. 2019):
    Source: Ranking of Heterogeneous Catalysts Metals by Their Greeness
    - The study analysed some common transition catalysts and assigned a greeness score based on toxicity. 
    - The score is as follows:

            element,greeness_score
            V,0.5
            Zr,0.5
            Mo,0.25
            Mn,0.25
            Fe,0.25
            Ru,0
            Co,1
            Rh,0.5
            Ir,0
            Ni,1
            Pd,0.75
            Pt,0.25
            Cu,0.5
            Ag,0.5
            Au,0.75
            Zn,0.5
            Cd,0.75
            Sn,0.5

    You will have to score other transition metals based on YOUR KNOWLEDGE.

"""