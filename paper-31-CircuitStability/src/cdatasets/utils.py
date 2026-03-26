from transformer_lens.utils import get_attention_mask


def generic_collate(model, xs):
    clean, corrupted_strings, labels = zip(*xs)
    # the clean and corrupted strings together
    batch_size = len(clean)
    all_examples = clean + corrupted_strings
    tokens = model.to_tokens(all_examples, prepend_bos=True, padding_side="left")
    attention_mask = get_attention_mask(model.tokenizer, tokens, True)
    input_lengths = attention_mask.sum(1)
    n_pos = attention_mask.size(1)
    return (
        (
            tokens[:batch_size],
            attention_mask[:batch_size],
            input_lengths[:batch_size],
            n_pos,
        ),
        (
            tokens[batch_size:],
            attention_mask[batch_size:],
            input_lengths[batch_size:],
            n_pos,
        ),
        list(labels),
    )
