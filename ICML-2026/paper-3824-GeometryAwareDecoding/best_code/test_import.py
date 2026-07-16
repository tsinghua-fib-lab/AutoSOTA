import sys; sys.path.insert(0, ".")
from logit_processor import TopH_LogitsProcessor
from logit_processor_w1 import TopW_LogitsProcessor
print("Both processors importable")
from huggingface import HFLM
print("HFLM imported successfully from repo")
