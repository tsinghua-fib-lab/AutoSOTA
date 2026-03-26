import torch
import sys

MODEL_PATH = "/tmp/PIGuard_weights"

from transformers import AutoModelForSequenceClassification

print("Loading model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, trust_remote_code=True)
model.to(device)
model.eval()

# Try fvcore with different seq lengths
try:
    from fvcore.nn import FlopCountAnalysis
    
    for seq_len in [35, 100, 175, 200, 256, 400, 512]:
        input_ids = torch.ones(1, seq_len, dtype=torch.long).to(device)
        attention_mask = torch.ones(1, seq_len, dtype=torch.long).to(device)
        
        inputs = (input_ids, attention_mask)
        flops = FlopCountAnalysis(model, inputs)
        total_flops = flops.total()
        gflops = total_flops / 1e9
        print(f"fvcore GFLOPs (seq_len={seq_len}): {gflops:.2f}")
        
except Exception as e:
    print(f"fvcore error: {e}")
    import traceback
    traceback.print_exc()

