import enum

import sequence_models.constants as constants


class TaskType(enum.Enum):
    LM = "lm"
    OADM = "oadm"
    GLM = "glm"


FIM_MIDDLE = "_"
FIM_PREFIX = "("
FIM_SUFFIX = ")"
MSA_ALPHABET_PLUS = constants.MSA_ALPHABET + FIM_MIDDLE + FIM_PREFIX + FIM_SUFFIX
DNA_ALPHABET_PLUS = (
    constants.DNA
    + "N"
    + constants.MSA_PAD
    + constants.MASK
    + constants.STOP
    + constants.START
    + FIM_MIDDLE
    + FIM_PREFIX
    + FIM_SUFFIX
)

# ensure there are no unintentional character overlaps
assert len(MSA_ALPHABET_PLUS) == len(set(MSA_ALPHABET_PLUS))
assert len(DNA_ALPHABET_PLUS) == len(set(DNA_ALPHABET_PLUS))
