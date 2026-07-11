#!/usr/bin/env python3
"""Parse CHAIR metrics from eval output JSON."""
import json, sys, os
def parse(json_path):
    with open(json_path) as f:
        data = json.load(f)
    m = data["metrics"]
    chairs = m["CHAIRs"]
    chairi = m["CHAIRi"]
    return chairs, chairi

if __name__ == "__main__":
    chairs, chairi = parse(sys.argv[1])
    print(f"CHAIRs={chairs} CHAIRi={chairi}")
    # Output machine-readable
    print(f"METRICS_JSON={json.dumps({CHAIRs: chairs, CHAIRi: chairi})}")
