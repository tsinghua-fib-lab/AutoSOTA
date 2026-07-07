cd /wyf_project/SeRAG

# musique
SPACY_MODEL="en_core_web_trf"
EMBEDDING_MODEL="model/all-mpnet-base-v2"
DATASET_NAME="musique"
LLM_MODEL="gpt-4o-mini"   
MAX_WORKERS=16


# 2wikimultihop
# SPACY_MODEL="en_core_web_trf"
# EMBEDDING_MODEL="model/all-mpnet-base-v2"
# DATASET_NAME="2wikimultihop"
# LLM_MODEL="gpt-4o-mini"
# MAX_WORKERS=16


# hotpotqa
# SPACY_MODEL="en_core_web_trf"
# EMBEDDING_MODEL="model/all-mpnet-base-v2"
# DATASET_NAME="hotpotqa"
# LLM_MODEL="gpt-4o-mini"
# MAX_WORKERS=16


python run.py \
    --spacy_model ${SPACY_MODEL} \
    --embedding_model ${EMBEDDING_MODEL} \
    --dataset_name ${DATASET_NAME} \
    --llm_model ${LLM_MODEL} \
    --max_workers ${MAX_WORKERS} \
