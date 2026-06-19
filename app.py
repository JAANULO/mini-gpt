import os
import sys
import threading
import uuid
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch

from chat import DATA_FILE, CACHE_FILE, EXPORT_FILE
from mini_gpt.transformer import MiniGPT
from mini_gpt.tokenizer import Tokenizer
from mini_gpt.utils import DEVICE, hash_data, load_cache, load_export, top_k_top_p_sampling, logger
from mini_gpt.inference import generate_response

dist_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'app_frontend', 'dist'))
app = Flask(__name__, static_folder=dist_dir)
CORS(app)

from mini_gpt.config import cfg

# Global state
model = None
tokenizer = None
is_training = False
training_status = {"epoch": 0, "loss": 0.0, "perplexity": 0.0, "progress": 0.0}

# Session-based state
sessions_history = {}
sessions_temp = {}

def load_app_model():
    global model, tokenizer
    logger.info("Loading model for API...")
    
    if not os.path.exists(DATA_FILE):
        logger.error(f"Data file {DATA_FILE} missing.")
        return

    import json
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)
    sentences = data.get("zdania", [])
    current_hash = hash_data(DATA_FILE)
    
    tokenizer_temp = Tokenizer()
    loaded = tokenizer_temp.load("exports/tokenizer.model")
    if not loaded:
        logger.error("Brak wytrenowanego tokenizera BPE. Uruchom najpierw train.py")
        return

    model_temp = MiniGPT(
        vocab_size=tokenizer_temp.vocab_size,
        embed_dim=cfg.embed_dim,
        num_layers=cfg.num_layers,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
        max_length=cfg.max_length,
    ).to(DEVICE)

    tokenizer_from_cache, cache_ok = load_cache(model_temp, current_hash, CACHE_FILE)
    if cache_ok:
        tokenizer = tokenizer_from_cache
        model = model_temp
        logger.info("✅ Loaded model from cache!")
    else:
        tokenizer_export, export_ok = load_export(model_temp, EXPORT_FILE)
        if export_ok:
            tokenizer = tokenizer_export
            model = model_temp
            logger.info("✅ Loaded model from export!")
        else:
            logger.warning("⚠️ No trained model found. Ready for training.")
            tokenizer = tokenizer_temp
            model = model_temp

    model.set_training(False)



@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    if model is None or tokenizer is None:
        return jsonify({"error": "Model not ready."}), 500

    data = request.json
    session_id = data.get('session_id', 'default')
    
    if session_id not in sessions_history:
        sessions_temp[session_id] = cfg.default_temp
        sessions_history[session_id] = []

    question = data.get('message', '')
    cmd = data.get('command', '')

    if cmd == 'clear':
        sessions_history[session_id].clear()
        return jsonify({"response": "History cleared."})
        
    if cmd.startswith('temp'):
        try:
            sessions_temp[session_id] = float(cmd.split()[1])
            return jsonify({"response": f"Temperature set to {sessions_temp[session_id]}"})
        except:
            pass

    history = sessions_history[session_id]
    temp = sessions_temp[session_id]

    response = generate_response(model, tokenizer, question, history, temp)
    
    if response != "...":
        question_clean = question.lower().strip().rstrip("?")
        history.append((question_clean, response))
        sessions_history[session_id] = history[-cfg.memory_window:]

    return jsonify({"response": response})

def training_worker():
    global is_training, training_status
    is_training = True
    
    try:
        import subprocess
        process = subprocess.Popen(
            [sys.executable, "train.py"], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True
        )
        
        for line in process.stdout:
            if "Epoch" in line and "loss:" in line:
                try:
                    parts = line.split()
                    # Example: Epoch 100/3000 (3%) loss: 2.1234 perplexity: 8.36
                    epoch_str = parts[1].split('/')[0]
                    loss_str = parts[4]
                    perp_str = parts[6]
                    proc_str = parts[2].strip("()%")
                    
                    training_status["epoch"] = int(epoch_str)
                    training_status["loss"] = float(loss_str)
                    training_status["perplexity"] = float(perp_str)
                    training_status["progress"] = float(proc_str)
                except Exception:
                    pass

        process.wait()
    except Exception as e:
        logger.error(f"Training worker error: {e}")
    finally:
        load_app_model()
        is_training = False

@app.route('/api/train/start', methods=['POST'])
def start_train():
    global is_training
    if is_training:
        return jsonify({"error": "Training already in progress."}), 400
    t = threading.Thread(target=training_worker)
    t.start()
    return jsonify({"status": "started"})

@app.route('/api/train/status', methods=['GET'])
def get_train_status():
    return jsonify({
        "is_training": is_training,
        "status": training_status
    })

@app.route('/outputs/<path:filename>')
def serve_outputs(filename):
    return send_from_directory('outputs', filename)

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/visualize', methods=['POST'])
def run_visualization():
    data = request.json
    script = data.get('script')
    args = data.get('args', [])
    
    import subprocess
    cmd = [sys.executable, f"tools/{script}.py"] + args
    subprocess.run(cmd)
    
    return jsonify({"status": "done"})

@app.route('/api/math_step', methods=['POST'])
def math_step():
    data = request.json
    word = data.get('word', '')
    if not word or model is None:
        return jsonify({"error": "Brak słowa lub modelu"}), 400
    
    ids = tokenizer.encode(word)
    if not ids:
        return jsonify({"error": "Słowo nie występuje w słowniku"}), 400
        
    wejscie = torch.tensor([ids], dtype=torch.long, device=DEVICE)
    
    with torch.no_grad():
        tok = model.tok_emb(wejscie)
        poz = torch.arange(wejscie.shape[1], device=DEVICE)
        pos = model.pos_emb(poz)
        x = tok + pos
        
        x2 = model.blocks[0].ln1(x)
        in_proj_weight = model.blocks[0].attn.in_proj_weight
        in_proj_bias = model.blocks[0].attn.in_proj_bias
        
        qkv = torch.nn.functional.linear(x2, in_proj_weight, in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)
        
        wymiar_glowicy = model.embed_dim // 4
        q0 = q[0, :, :wymiar_glowicy]
        k0 = k[0, :, :wymiar_glowicy]
        v0 = v[0, :, :wymiar_glowicy]
        
        scores = q0 @ k0.transpose(0, 1) / (wymiar_glowicy ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        wyjscie_uwagi = attn @ v0

    def wytnij_macierz(tensor_2d, max_kol=4):
        k = min(tensor_2d.shape[1], max_kol)
        return tensor_2d[:, :k].cpu().numpy().tolist()
        
    def rzut_2d(tensor_2d):
        return tensor_2d[:, :2].cpu().numpy().tolist()

    def to_latex(mat):
        lines = []
        for row in mat:
            lines.append(" & ".join([f"{x:.3f}" for x in row]))
        return "\\begin{bmatrix}\n" + " \\\\\n".join(lines) + "\n\\end{bmatrix}"

    words_list = [tokenizer.id_to_word.get(i, "?") for i in ids]
    
    math_html = f"<h3>Analiza dla słowa: '{word}'</h3>"
    math_html += f"<p>Tokeny (ID): {ids}</p>"
    math_html += f"<p>Rozbite słowa: {words_list}</p>"
    
    math_html += f"<h4>1. Embeddingi (Wycinek)</h4>"
    math_html += f"$$ \\text{{TokEmb}} \\approx {to_latex(wytnij_macierz(tok[0]))} $$"
    math_html += f"$$ \\text{{PosEmb}} \\approx {to_latex(wytnij_macierz(pos))} $$"
    math_html += f"$$ X = \\text{{TokEmb}} + \\text{{PosEmb}} $$"
    
    math_html += f"<h4>2. Projekcja Q, K, V (Głowa 0, Wycinek)</h4>"
    math_html += f"$$ Q \\approx {to_latex(wytnij_macierz(q0))} $$"
    math_html += f"$$ K \\approx {to_latex(wytnij_macierz(k0))} $$"
    math_html += f"$$ V \\approx {to_latex(wytnij_macierz(v0))} $$"
    
    math_html += f"<h4>3. Wynik uwagi (Attention Scores)</h4>"
    math_html += f"$$ \\text{{Scores}} = \\frac{{Q K^T}}{{\\sqrt{{d_k}}}} \\approx {to_latex(scores.cpu().numpy().tolist())} $$"
    math_html += f"$$ \\text{{Attn}} = \\text{{softmax}}(\\text{{Scores}}) \\approx {to_latex(attn.cpu().numpy().tolist())} $$"
        
    return jsonify({
        "tokens": ids,
        "words": words_list,
        "math_html": math_html,
        "embeddings_shape": list(tok.shape),
        "tok_matrix": wytnij_macierz(tok[0]),
        "pos_matrix": wytnij_macierz(pos),
        "q_matrix": wytnij_macierz(q0),
        "k_matrix": wytnij_macierz(k0),
        "v_matrix": wytnij_macierz(v0),
        "scores_matrix": scores.cpu().numpy().tolist(),
        "attn_matrix": attn.cpu().numpy().tolist(),
        "out_matrix": wytnij_macierz(wyjscie_uwagi),
        "q_2d": rzut_2d(q0),
        "k_2d": rzut_2d(k0),
        "v_2d": rzut_2d(v0)
    })

if __name__ == '__main__':
    load_app_model()
    app.run(debug=True, port=5000, use_reloader=False)
