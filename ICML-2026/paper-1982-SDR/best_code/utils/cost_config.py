TOKEN_COSTS = {
    "gpt-oss": {
        "input": 0.16 / 1e6,
        "output": 0.57 / 1e6        
    },
    "qwen-72b": {
        "input": 0.42 / 1e6,
        "output": 0.63 / 1e6
    },    
    "llama-4": {
        "input": 0.29 / 1e6,
        "output": 0.93 / 1e6
    },
    "deepseek-v3": {
        "input": 1.00 / 1e6,
        "output": 1.54 / 1e6
    },
    "o4-mini": {
        "input": 1.1 / 1e6,
        "output": 4.4 / 1e6          
    },    
    "gpt-4o": {
        "input": 2.5 / 1e6,
        "output": 10.0 / 1e6        
    },
}

LATENCY_COSTS = {
    "local": {
        "TTFT": 1.,
        "output": 3000, # token/s
    }
}

AVERAGE_RESPONSE_LENGTH = { # in train set
    "deepseek-v3": {
        "mmlu": 154.40,
        "squad": 6.04,
        "coqa": 5.75,
        "gsm8k": 188.16
    },
    "gpt-4o": {
        "mmlu": 120.43,
        "squad": 5.20,
        "coqa": 5.50,
        "gsm8k": 147.58
    },
    "o4-mini": {
        "mmlu": 49.80,
        "squad": 5.42,
        "coqa": 5.46,
        "gsm8k": 89.61
    },
    "gpt-oss": {
        "mmlu": 25.24,
        "squad": 6.96,
        "coqa": 5.91,
        "gsm8k": 41.28
    },
    "llama-4": {
        "mmlu": 415.69,
        "squad": 6.38,
        "coqa": 6.34,
        "gsm8k": 185.49
    },
    "qwen-72b": {
        "mmlu": 74.88,
        "squad": 7.05,
        "coqa": 17.17,
        "gsm8k": 111.33
    },
}