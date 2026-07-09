from .matmul import matmul_flexible, bmm_flexible

QUANTIZED_FUNCTION_MAP = {
    "matmul": {
        "flexible": matmul_flexible,
    },
    "bmm": {
        "flexible": bmm_flexible,
    },
}


def get_quantized_func(op: str, q_config: str):
    assert op in QUANTIZED_FUNCTION_MAP, f"Unsupported quantized op: {op}"
    assert q_config["name"] in QUANTIZED_FUNCTION_MAP[op], f"Unsupported quantized config: {q_config}"
    return QUANTIZED_FUNCTION_MAP[op][q_config["name"]]
