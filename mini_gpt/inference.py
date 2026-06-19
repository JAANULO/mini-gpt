import torch
from mini_gpt.utils import DEVICE, top_k_top_p_sampling
from mini_gpt.config import cfg

def build_context(history, new_question):
    """
    Buduje kontekst rozmowy z historii, aby model wiedział o czym rozmawiano.
    """
    parts = []
    for (old_q, old_a) in history[-cfg.memory_window:]:
        parts.append(f"user {old_q} assistant {old_a}")
    parts.append(f"user {new_question} assistant")
    return " ".join(parts)

def generate_response(model, tokenizer, question, history, temp):
    """
    Generuje odpowiedź modelu dla zadanego pytania i historii konwersacji.
    """
    question = question.lower().strip().rstrip("?")
    context = build_context(history, question)
    ids = tokenizer.encode(context)

    with torch.no_grad():
        for _ in range(300):
            input_tensor = torch.tensor([ids[-cfg.max_length:]], dtype=torch.long, device=DEVICE)
            logits, _ = model.forward(input_tensor)
            
            last_logits = logits[0, -1].cpu().numpy()
            next_id = top_k_top_p_sampling(
                last_logits, top_k=50, top_p=0.9, temperature=temp
            )
            ids.append(next_id)

            text = tokenizer.decode(ids)
            if "koniec" in text[-10:]:
                break

    text = tokenizer.decode(ids)

    # Wyodrębnij tylko najnowszą odpowiedź
    if "assistant" in text:
        idx = text.rfind("assistant") + len("assistant")
        response = text[idx:]
    else:
        response = text

    # Utnij przy ewentualnych znacznikach końca
    for stop_word in ["koniec", "user"]:
        if stop_word in response:
            response = response[:response.index(stop_word)]

    response = response.strip()
    return response if response else "..."
