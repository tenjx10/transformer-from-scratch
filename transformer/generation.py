import torch

from transformer.model import softmax


@torch.no_grad()
def decode(model, prompt_ids, max_new_tokens, temperature=1.0, top_p=1.0, eos_id=None):
    device = next(model.parameters()).device
    ids = list(prompt_ids)
    new_ids = []

    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.context_length:]], dtype=torch.long, device=device)

        # next-token logits from last position
        logits = model(x)[0, -1, :]

        if temperature == 0:
            next_id = int(torch.argmax(logits))
        else:
            probs = softmax(logits / temperature, dim=-1)

            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            # a token stays if the total of the tokens before it hasn't reached top_p yet,
            # which also keeps the token that pushes the total over top_p
            keep = (cumulative - sorted_probs) < top_p
            keep[0] = True
            sorted_probs = sorted_probs * keep
            sorted_probs = sorted_probs / sorted_probs.sum()

            choice = torch.multinomial(sorted_probs, num_samples=1)
            next_id = int(sorted_ids[choice])

        ids.append(next_id)
        new_ids.append(next_id)
        if eos_id is not None and next_id == eos_id:
            break

    return new_ids


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 256, temperature: float = 0.8, top_p: float = 0.95) -> str:
    """complete a text prompt; stops at <|endoftext|> or after max_new_tokens"""
    eos_id = tokenizer.token_to_id.get(b"<|endoftext|>")
    new_ids = decode(model, tokenizer.encode(prompt), max_new_tokens, temperature, top_p, eos_id)
    if new_ids and new_ids[-1] == eos_id:
        new_ids = new_ids[:-1]
    return prompt + tokenizer.decode(new_ids)
