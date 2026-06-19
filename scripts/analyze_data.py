import os
import json

def main():
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print("Biblioteka 'plotly' nie jest zainstalowana. Uruchom 'pip install plotly'")
        return

    input_file = os.path.join("data", "conversations.jsonl")
    if not os.path.exists(input_file):
        print(f"Brak pliku danych: {input_file}")
        return

    ctx_char_lens = []
    ctx_word_lens = []
    res_char_lens = []
    res_word_lens = []

    print("Wczytywanie danych...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            pair = json.loads(line)
            
            ctx = pair.get("context", "")
            res = pair.get("response", "")
            
            ctx_char_lens.append(len(ctx))
            ctx_word_lens.append(len(ctx.split()))
            
            res_char_lens.append(len(res))
            res_word_lens.append(len(res.split()))

    if not ctx_char_lens:
        print("Brak danych do analizy.")
        return

    print("Generowanie wykresów interaktywnych Plotly...")
    
    fig = make_subplots(rows=2, cols=2, 
                        subplot_titles=(
                            "Kontekst (Pytania) - Znaki", 
                            "Odpowiedzi (Ty) - Znaki",
                            "Kontekst (Pytania) - Wyrazy", 
                            "Odpowiedzi (Ty) - Wyrazy"
                        ))

    # Kontekst - znaki
    fig.add_trace(go.Histogram(x=ctx_char_lens, nbinsx=100, name="Kontekst (znaki)", marker_color='blue'), row=1, col=1)
    
    # Odpowiedź - znaki
    fig.add_trace(go.Histogram(x=res_char_lens, nbinsx=100, name="Odpowiedź (znaki)", marker_color='green'), row=1, col=2)
    
    # Kontekst - wyrazy
    fig.add_trace(go.Histogram(x=ctx_word_lens, nbinsx=100, name="Kontekst (wyrazy)", marker_color='lightblue'), row=2, col=1)
    
    # Odpowiedź - wyrazy
    fig.add_trace(go.Histogram(x=res_word_lens, nbinsx=100, name="Odpowiedź (wyrazy)", marker_color='lightgreen'), row=2, col=2)

    fig.update_layout(
        title_text="Analiza Długości Wiadomości z Messengera (EDA)",
        height=800,
        showlegend=False
    )

    output_html = os.path.join("data", "length_analysis.html")
    fig.write_html(output_html)
    
    print(f"Sukces! Otwórz plik '{output_html}' w przeglądarce, aby zobaczyć interaktywne wykresy.")

if __name__ == "__main__":
    main()
