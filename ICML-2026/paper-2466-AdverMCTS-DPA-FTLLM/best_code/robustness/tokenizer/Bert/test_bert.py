import unicodedata
import pandas as pd
from transformers import AutoTokenizer

# Hugging Face models
# Something is rotten in the realm of bert
HF_TOKENIZERS = [
    "bert-base-uncased",]



def analyze_tokenization(
    chars, 
    output_csv: str, 
    include_names: bool = False
):
    """Analyze token counts for a list of characters and save to CSV."""
    
    # Load Hugging Face tokenizers
    hf_loaded = {}
    for name in HF_TOKENIZERS:
        try:
            hf_loaded[name] = AutoTokenizer.from_pretrained(name)
        except Exception as e:
            print(f"Failed to load {name}: {e}")


    data = []

    for ch in chars:
        cp = ord(ch)
        row = {"Unicode": f"U+{cp:04X}"}

        if include_names:
            row["Character_Name"] = unicodedata.name(ch, "UNKNOWN")

        # Hugging Face tokenizers
        for name, tokenizer in hf_loaded.items():
            try:
                tokens_char = tokenizer.encode(ch, add_special_tokens=False)
                print(tokens_char)
                tokens_base = tokenizer.encode("A ", add_special_tokens=False)
                tokens_with_char = tokenizer.encode("A" + ch + " ", add_special_tokens=False)
                diff = len(tokens_with_char) - len(tokens_base)
                row[f"{name}_char_tokens"] = tokens_char
                row[f"{name}_char_tokens_nb"] = len(tokens_char)
                row[f"{name}_diff"] = diff
            except Exception:
                row[f"{name}_char_tokens"] = "ERR"
                row[f"{name}_char_tokens"] = "ERR"
                row[f"{name}_diff"] = "ERR"


        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")


# --- Invisible / control characters ---
INVISIBLE_CODEPOINTS = [
    0x061C, 0x180E,
    0x200B, 0x200C, 0x200D, 0x200E, 0x200F,
    0x202A, 0x202C, 0x202D,
    0x2060, 0x2061, 0x2062, 0x2063, 0x2064,
    0x2066, 0x2068, 0x2069,
    0x206A, 0x206B, 0x206C, 0x206D, 0x206E, 0x206F,
    0xFEFF,
    0x1D173, 0x1D174, 0x1D175, 0x1D176, 0x1D177, 0x1D178, 0x1D179, 0x1D17A,
    0xE0001,
] + list(range(0xE0020, 0xE0080))
invisible_chars = [chr(cp) for cp in INVISIBLE_CODEPOINTS]

# --- Emojis ---
EMOJI_RANGES = [
    (0x1F300, 0x1F5FF),
    (0x1F600, 0x1F64F),
    (0x1F680, 0x1F6FF),
    (0x1F900, 0x1F9FF),
    (0x1FA70, 0x1FAFF),
]
emoji_chars = [chr(cp) for start, end in EMOJI_RANGES for cp in range(start, end + 1)]


if __name__ == "__main__":
    # Invisible/control characters
    analyze_tokenization(
        invisible_chars, 
        "bert_invisible_char.csv", 
        include_names=True
    )

    # Emojis
    analyze_tokenization(
        emoji_chars, 
        "bert_emoji.csv", 
        include_names=False
    )

