import json
import math
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from quarc.preprocessing.exceptions import *
from quarc.utils.smiles_utils import get_common_solvents_canonical, canonicalize_smiles

COMMON_SOLVENTS_CANONICAL = get_common_solvents_canonical()


RDLogger.logger().setLevel(RDLogger.ERROR)

_reagent_conv_rules = None


def _get_reagent_conv_rules():
    """Lazy loading of reagent conversion rules. Returns empty dict if unavailable."""
    global _reagent_conv_rules
    if _reagent_conv_rules is None:
        try:
            # Use docker path
            with open("/app/quarc/data/processed/agent_encoder/agent_rules_v1.json", "r") as f:
                _reagent_conv_rules = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _reagent_conv_rules = {}
    return _reagent_conv_rules


def canonicalize_smiles_reagent_conv_rules(s):
    reagent_conv_rules = _get_reagent_conv_rules()
    r = reagent_conv_rules.get(s, None)
    if r is not None:
        return r
    else:
        return s


INVALID_SMILES = [
    "null",
    "S=[NH4+]",
    "C[ClH]",
    "[OH2-]",
    "[ClH-]",
    "[P+3]",
    "[B]",
]


def is_smiles_valid(s):
    return s not in INVALID_SMILES


# re-canonicalized
SMILES2CHARGE = {
    "Br[Br-]Br": -1,
    "[BH]1234[H][BH]561[BH]173[BH]34([H]2)[BH]247[BH]761[BH]165[H][BH]586[BH]32([BH]4715)[H]8": 0,
    "[BH3]S(C)C": 0,
    "[BH3-][S+](C)C": 0,
    "[BH3]O1CCCC1": 0,
    "[BH3]N1CCCCC1C": 0,
}


def get_smiles_charge(s):
    q = 0
    for i in s.split("."):
        q += get_smiles_charge_single_mol(i)
    return q


def get_smiles_charge_single_mol(s):
    if not is_smiles_valid(s):
        raise ProcessingError("get_smiles_charge(): s=" + s)
    q = SMILES2CHARGE.get(s, None)
    if q is None:
        try:
            mol = Chem.MolFromSmiles(canonicalize_smiles_reagent_conv_rules(s), sanitize=False)
        except:
            raise ProcessingError("get_smiles_charge(): MolFromSmiles() fails, s=" + s)
        if mol is None:
            raise ProcessingError("get_smiles_charge(): MolFromSmiles() fails, s=" + s)
        q = Chem.rdmolops.GetFormalCharge(mol)
    return q


# re-canonicalized
SMILES2MW = {
    "Br[Br-]Br": 239.71,
    "[H]1[BH]234[H][BH]256[BH]278[H][BH]29%10[H][BH]92%11[BH]139[BH]451[BH]923[BH]%10%117[BH]6813": 112.142,
    "[BH3]S(C)C": 75.97,
    "[BH3-][S+](C)C": 75.97,
    "[BH3]O1CCCC1": 85.94,
    "[BH3]N1CCCCC1C": 113.009,
    "[BH4][Na]": 37.84,
    "[Li][BH4]": 21.8,
    "[BH4][K]": 53.94,
    "Cl1[Rh]Cl[Rh]1": 276.71,
    "CC1=CC(C)=O[Fe]23(O1)(OC(C)=CC(C)=O2)OC(C)=CC(C)=O3": 356.19,
    "[Li]N([Si](C)(C)C)[Si](C)(C)C": 167.4,
    "C[Si](C)(C)N([Na])[Si](C)(C)C": 183.37,
}


def get_molecular_weight(smiles):
    res = 0
    for s in smiles.split("."):
        mw = SMILES2MW.get(s, None)
        if mw is None:
            try:
                mol = Chem.MolFromSmiles(s, sanitize=True)
                mw = Descriptors.MolWt(mol)
            except:
                pass
        if mw is None:
            raise InvalidQuantError(f"get_molecular_weight(): no MW data, smiles={smiles}, s={s}")
            # sys.stderr.write('get_molecular_weight(): no MW data, smiles='+smiles+', s='+s+'\n')
            return None
        res += mw
    return res


def __test_get_molecular_weight():
    smiles_kcl = "[K+].[Cl-]"
    kcl = get_molecular_weight(smiles_kcl)
    assert abs(kcl - 74.551) < 0.01


SOLUTES = [
    "[Li]CCCC",
    "[Li]C(C)(C)C",
    "[Li]C(C)CC",
    "[Li]C",
    "CC(C)[Mg]Cl",
    "CC(C)[Mg]Br",
    "CC[Zn]CC",
    "B",
    "CC(C)C[AlH]CC(C)C",
    "C[Si](C)(C)C=[N+]=[N-]",
    "[AlH4-].[Li+]",
    "B1C2CCCC1CCC2",
    "C[Si](C)(C)[N-][Si](C)(C)C.[Na+]",
    "Cl",
    "Br",
    "[K+].[OH-]",
    "[Na+].[OH-]",
    "N",
    "O=C([O-])O.[Na+]",
    "O=C([O-])[O-].[K+].[K+]",
    "O=C([O-])[O-].[Na+].[Na+]",
]


def split_neutral_fragment(reagents):
    reagents_neutral = set()
    reagents_charged = []
    for r in reagents:
        r_split = r.split(".")
        r_remaining = []
        for s in r_split:
            q = get_smiles_charge(s)
            if int(q) == 0:
                reagents_neutral.add(s)
            else:
                r_remaining.append(s)
        if len(r_remaining) > 0:
            r_remaining = ".".join(r_remaining)
            q = get_smiles_charge(r_remaining)
            if int(q) == 0:
                reagents_neutral.add(r_remaining)
            else:
                reagents_charged.append(r_remaining)
    return reagents_neutral, reagents_charged


def preprocess_reagents(reagents):
    """
    inputs: list of str, smiles
    outputs: list of str, smiles
    Rules:
    1. Neutral molecules are splitted from compounds
    2. Try to combine separated charged species
    3. Try to map Pd coordinated compounds with ligands
    4. Canonicalization using hardcoded rules
    """
    assert isinstance(reagents, list)
    for i in range(len(reagents)):
        reagents[i] = canonicalize_smiles_reagent_conv_rules(reagents[i])

    # Rule 1, split neutral
    reagents_neutral, reagents_charged = split_neutral_fragment(reagents)

    # Rule 2, combine charged, reagents_charged --> reagents_neutral
    # q for smiles in reagents_charged
    charges = [get_smiles_charge(s) for s in reagents_charged]
    # sanity check
    # check 1, total charge 0
    total_charge = sum(charges)
    if total_charge != 0:
        raise ProcessingError(
            f"preprocess_reagents(): total charge is not zero, q={str(total_charge)} \
                \nreagents: {reagents}, reagents_neutral: {reagents_neutral}, reagents_charged: {reagents_charged}"
        )
    if len(reagents_charged) > 0:
        reagents_neutral.add(canonicalize_smiles(".".join(reagents_charged)))
    reagents_neutral = list(reagents_neutral)

    # Rule 3, Canonicalization, replace using reagent_conv_rules.json
    res = set()
    for i in reagents_neutral:
        tmp = canonicalize_smiles_reagent_conv_rules(i)
        tmp1, tmp2 = split_neutral_fragment([tmp])
        if len(tmp2) != 0:
            sys.stderr.write(
                "preprocess_reagents(): error: charged fragment, s=" + str(reagents) + "\n"
            )
        for s in tmp1:
            res.add(s)

    return list(res)


def convert_smiles2cnt(s):
    """
    split smiles to {smiles:N}
    """
    res = {}
    for i in s.split("."):
        res[i] = res.get(i, 0) + 1
    return res


def get_solute_solvent(smiles):
    """
    split smiles into solvents and solutes
    Also try to split salts
    SOLUTES, solid, gas
    COMMON_SOLVENTS, liquid
    need special rules for NH3
    return solvents, solutes as dict of str:N
    """
    solutes = {}
    solvents = {}

    # split neutral
    neutral = {}
    r_remaining = {}
    for s in smiles.split("."):
        q = get_smiles_charge(s)
        if int(q) == 0:
            neutral[s] = neutral.get(s, 0) + 1
        else:
            r_remaining[s] = r_remaining.get(s, 0) + 1
    if len(r_remaining) > 0:
        charges = {s: get_smiles_charge(s) for s in r_remaining.keys()}
        # sanity check, total charge 0
        total_charge = 0
        for s, q in charges.items():
            total_charge += q * r_remaining[s]
        if total_charge != 0:
            raise ProcessingError(
                f"get_solute_solvent(): total charge is not zero, smiles={smiles}, q={total_charge}"
            )

        # check if only one cations or one anions
        num_cation = 0
        num_anions = 0
        for s, q in charges.items():
            if q > 0:
                num_cation += r_remaining[s]
            if q < 0:
                num_anions += r_remaining[s]
        if num_anions == 1 or num_cation == 1:
            s_cat = []
            for s in r_remaining:
                s_cat += [s] * r_remaining[s]
            s = ".".join(s_cat)
            s = canonicalize_smiles(s)
            # s = canonicalize_smiles_reagent_conv_rules(s)
            charged_split = {s: 1}
            solutes.update(charged_split)
            r_remaining = []

        # try to divide cations/anions into neutral groups
        charged_split = {}
        while len(r_remaining) > 0:
            match_found = False
            # find the largest negative q
            s_max = 0
            q_max = 0
            for s in r_remaining.keys():
                if charges[s] < q_max:
                    q_max = charges[s]
                    s_max = s
            # try to find cations
            for s in r_remaining.keys():
                q = charges[s]
                if -q_max * r_remaining[s_max] == q * r_remaining[s]:
                    d = math.gcd(abs(q_max), abs(q))
                    q_max /= d
                    q /= d
                    salt = ".".join([s_max] * int(abs(q)) + [s] * int(abs(q_max)))
                    charged_split[salt] = d
                    r_remaining.pop(s)
                    r_remaining.pop(s_max)
                    match_found = True
                    break
            if match_found:
                continue
            # no exact match
            for s in r_remaining.keys():
                q = charges[s]
                if (
                    -q_max * r_remaining[s_max] < q * r_remaining[s]
                    and q > 0
                    and abs(q_max * r_remaining[s_max]) % q == 0
                ):
                    n = int(abs(q_max * r_remaining[s_max] / q))
                    d = math.gcd(r_remaining[s_max], n)
                    salt = ".".join([s_max] * int(r_remaining[s_max] / d) + [s] * int(n / d))
                    charged_split[salt] = d
                    r_remaining[s] -= n
                    r_remaining.pop(s_max)
                    match_found = True
                    break
            if match_found:
                continue
            for s in r_remaining.keys():
                q = charges[s]
                if (
                    -q_max * r_remaining[s_max] > q * r_remaining[s]
                    and q > 0
                    and abs(q * r_remaining[s]) % q_max == 0
                ):
                    n = int(abs(q * r_remaining[s] / q_max))
                    d = math.gcd(r_remaining[s], n)
                    salt = ".".join([s_max] * int(n / d) + [s] * int(r_remaining[s] / d))
                    charged_split[salt] = d
                    r_remaining[s_max] -= n
                    r_remaining.pop(s)
                    match_found = True
                    break
            if match_found:
                continue
            # may need impovement
            raise ProcessingError(f"get_solute_solvent(): matching algo error for smiles={smiles}")

        # salts are solute, not really...
        charged_split_can = {}
        for s, v in charged_split.items():
            s = canonicalize_smiles(s)
            # s = canonicalize_smiles_reagent_conv_rules(s)
            charged_split_can[s] = v
        solutes.update(charged_split_can)

    for s, v in neutral.items():
        s = canonicalize_smiles(s)
        # s = canonicalize_smiles_reagent_conv_rules(s)
        if s in SOLUTES:
            solutes[s] = v
        elif s in COMMON_SOLVENTS_CANONICAL:  # jiannan's list -> pistachio's densitieis
            solvents[s] = v
        else:
            solutes[s] = v
            # phase = get_chemical_phase(s)
            # if phase == 's' or phase == 'g':
            #    solutes[s] = v
            # else:
            #    solvents[s] = v

    return solutes, solvents


special_conc = {
    #'CCO.O': 8.5, # ~50%v/v
    "O.[Na+].[Cl-]": 5.0,  # brine
    #'CCO.ClCCl': 4.0, #~50%v/v
    #'C1CCCO1.CC#N': 15.0, #~50%v/v
    "C[Si](C=[N+]=[N-])(C)C": 2.0,
    "CC(C[AlH]CC(C)C)C": 1.0,
    "COCCO[AlH2-]OCCOC.[Na+]": 3.5,  # ~70%wt in tulene
    "C1CC2CCCC(C1)B2": 0.5,  # in THF
    "CC[BH-](CC)CC.[Li+]": 1.0,  # super-hydride
    "CC[BH](CC)(CC)[Li]": 1.0,
    "Cl[Mg]C(C)C": 2.0,
    "[Li]CCCC": 1.0,
    "[Li]C(C)(C)C": 1.0,
}
