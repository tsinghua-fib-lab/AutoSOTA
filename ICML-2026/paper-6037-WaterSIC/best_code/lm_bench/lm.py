from tqdm import tqdm
from lm_eval.api.model import LM
import torch
import torch.nn.functional as F


class My_LM(LM):
    def __init__(self, model, tokenizer, batch_size, device):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size

    def loglikelihood_batch(self, requests):
        B = len(requests)
        pref_tokens = [self.tokenizer.encode(request.args[0], bos=True, eos=False) for request in requests]
        suf_tokens = [self.tokenizer.encode(request.args[1], bos=False, eos=False) for request in requests]

        total_len = [(len(pref_tokens[i]) + len(suf_tokens[i])) for i in range(B)]
        max_len = max(total_len)
        pad_id = self.tokenizer.pad_id
        llm_input = [
            pref_tokens[i] + suf_tokens[i] + [pad_id] * (max_len - total_len[i])
            for i in range(B)]


        llm_input = torch.tensor(llm_input, device=self.device, dtype=torch.long)
        logits = self.model(llm_input, start_pos=0)
        log_probs = F.log_softmax(logits, dim=2)

        len1 = [len(x) for x in pref_tokens]
        len2 = [len(x) for x in suf_tokens]
        greedy = torch.argmax(logits, dim=2).cpu().tolist()
        is_greedy = [greedy[len1[i] - 1: len1[i] - 1 + len2[i]] ==
                     suf_tokens[i] for i in range(B)]
        log_prob_sum = [
            sum(log_probs[i, len1[i] - 1 + j, suf_tokens[i][j]].item()
                for j in range(len2[i]))
            for i in range(B)]
        return list(zip(log_prob_sum, is_greedy))

    def loglikelihood(self, requests):
        results = []
        for i in tqdm(range(0, len(requests), self.batch_size)):
            r = min(len(requests), i + self.batch_size)
            batch = requests[i:r]
            results.extend(self.loglikelihood_batch(batch))
        return results

    def generate_until(self, requests):
        raise NotImplementedError()

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError()


def generate_greedy(model, tokenizer, prompt, device="cuda", max_len=256):
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    logits = model(torch.tensor([tokens]).to(device), start_pos=0)
    for i in tqdm(range(max_len)):
        next_token = logits[0, -1].argmax().item()
        tokens.append(next_token)
        if next_token == tokenizer.eos_id:
            break
        logits = model(torch.tensor([[next_token]]).to(device), start_pos=len(tokens) - 1)
    return tokenizer.decode(tokens)


if __name__ == "__main__":
    from parallel.config import no_q_config
    from parallel.start import start

    ckpt_path = "/home/semyon/quant-bucket/Llama-3-8B"
    model, tokenizer = start(ckpt_path, False, no_q_config)
    device = "cuda"

    text = generate_greedy(model, tokenizer, "The capital of Russia is ")
    print("text:", text)

    class Instance:
        def __init__(self, a, b):
            self.args = (a, b)

    requests = [
        Instance("The capital of Russia is ", "Moscow"),
        Instance("The capital of Russia is ", "St. Petersburg"),
        Instance("The captial of Russia is ", "Paris"),
        Instance("The capital of Russia is ", "cat")
    ]

    batch_size = 16
    lm = My_LM(model, tokenizer, batch_size, device)
    logl = lm.loglikelihood(requests)
    print("logl:")
    print(logl)
