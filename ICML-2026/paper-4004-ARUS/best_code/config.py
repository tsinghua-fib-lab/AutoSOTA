# ---------- Reproduction config for LLaVA-1.5-7B CHAIR ----------
DATA_DIR            = "/datasets/coco"

VQA_QUESTIONS_FILE  = f"{DATA_DIR}/v2_OpenEnded_mscoco_val2014_questions.json"
VQA_ANNOTATIONS_FILE= f"{DATA_DIR}/v2_mscoco_val2014_annotations.json"
IMAGE_DIR           = f"{DATA_DIR}/val2014"

# POPE
POPE_DIR            = f"{DATA_DIR}/POPE"
POPE_SPLITS         = ["adversarial", "popular", "random"]

# CHAIR
COCO_INSTANCES_FILE = f"{DATA_DIR}/annotations/instances_val2014.json"
CHAIR_SPLIT_FILE    = f"{DATA_DIR}/chair_image_ids_500.json"
CHAIR_NUM_IMAGES    = 500

# ---------- Results paths ----------
RESULTS_DIR_CHAIR   = "results/chair_llava"
VIZ_OUT_DIR         = "results/viz"

# ---------- Model ----------
# Use local cached model (llava-hf/llava-1.5-7b-hf format)
MODEL_ID = "/models/llava-v1.5-7b-hf"
CACHE_DIR           = "/autosota_cache/hf"
DEVICE              = "cuda:0"
DTYPE               = "bf16"

BATCH_SIZE          = 4

MAX_NEW_TOKENS_CAP  = 512

SEEDS               = [42]

# ---------- RUDDER hyperparams for LLaVA-1.5-7B (from paper Section 4.1) ----------
# Layer 30, alpha_max=20, k=5.0, c=1.0
EGR_POOLINGS        = ["attn"]
INJECTION_LAYERS    = [30]           # Best layer confirmed by sweep (24,28 worse, 32 invalid)
CARD_PRUNE_RATIO    = 0.0            # Token pruning disabled (regressed at 0.15)
BETA_ALPHA_MAX      = [15, 25]       # Sweep alpha_max: weaker (15) and stronger (25) steering
BETA_K              = [5.0]
BETA_C              = [1.0]
GATE_CLAMP          = (0.05, 1.0)
EGR_RMS_MATCH       = False   # RMS matching disabled — too weak steering, tested CHAIRs=46.0 vs baseline 39.6
MAX_TOKEN_NORM      = 20.0    # Per-token norm cap on steering update (from commented-out SimpleAddHook)
CARD_LAYER_INDICES   = [26, 28, 30]  # Multi-layer CARD aggregation (per Figure 9)
USE_MULTI_LAYER_CARD = False   # Multi-layer CARD tested, worse than single-layer

CHAIR_PROMPT = (
    "USER: <image>\nPlease help me describe the image in detail.\nASSISTANT:"
)

BAD_WORD_STRINGS = [
    "Instruction:", "Image:", "Question:", "Answer:",
    "User:", "Assistant:", "Subject:", "System:",
    "USER:", "ASSISTANT:"
]

COCO_IMG_DIR       = f"{DATA_DIR}/val2014"
COCO_INSTANCES_JSON= f"{DATA_DIR}/annotations/instances_val2014.json"
