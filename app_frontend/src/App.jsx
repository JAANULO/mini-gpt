import React, { useState, useEffect, useRef } from 'react';
import './index.css';

const SESSION_ID = "web-session-" + Math.floor(Math.random() * 1000000);

function ChatTab() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    if (window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise().catch(err => console.error("MathJax err", err));
    }
  }, [messages]);

  const sendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMsg = input.trim();
    setMessages(prev => [...prev, { text: userMsg, sender: 'user' }]);
    setInput('');
    setLoading(true);

    try {
      const apiUrl = window.location.port === '5173' ? 'http://127.0.0.1:5000/api/chat' : '/api/chat';
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, session_id: SESSION_ID })
      });
      const data = await res.json();
      
      setMessages(prev => [...prev, { text: data.response || "No response", sender: 'bot' }]);
    } catch (err) {
      setMessages(prev => [...prev, { text: "Błąd połączenia z serwerem.", sender: 'bot' }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="chat-window">
        {messages.length === 0 && (
          <div style={{ textAlign: 'center', opacity: 0.5, marginTop: '20%' }}>
            Napisz coś, aby rozpocząć... (np. test mechanizmu uwagi)
          </div>
        )}
        {messages.map((msg, idx) => (
          <div key={idx} className={`message ${msg.sender}`}>
            <div dangerouslySetInnerHTML={{ __html: msg.text.replace(/\n/g, '<br/>') }} />
          </div>
        ))}
        {loading && (
          <div className="message bot">
            <div className="loading-dots">
              <span></span><span></span><span></span>
            </div>
          </div>
        )}
        <div ref={chatEndRef} />
      </div>

      <form className="input-area" onSubmit={sendMessage}>
        <input 
          type="text" 
          value={input} 
          onChange={(e) => setInput(e.target.value)} 
          placeholder="Napisz wiadomość..." 
          disabled={loading}
        />
        <button type="submit" disabled={loading}>Wyślij</button>
      </form>
    </>
  );
}

function TrainTab() {
  const [status, setStatus] = useState({ is_training: false, epoch: 0, loss: 0, perplexity: 0, progress: 0 });

  const fetchStatus = async () => {
    try {
      const apiUrl = window.location.port === '5173' ? 'http://127.0.0.1:5000/api/train/status' : '/api/train/status';
      const res = await fetch(apiUrl);
      const data = await res.json();
      setStatus({
        is_training: data.is_training,
        ...data.status
      });
    } catch (e) {}
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 1000);
    return () => clearInterval(interval);
  }, []);

  const startTraining = async () => {
    try {
      const apiUrl = window.location.port === '5173' ? 'http://127.0.0.1:5000/api/train/start' : '/api/train/start';
      await fetch(apiUrl, { method: 'POST' });
      fetchStatus();
    } catch (e) {
      alert("Nie udało się rozpocząć treningu.");
    }
  };

  return (
    <div className="training-panel">
      <button className="train-btn" onClick={startTraining} disabled={status.is_training}>
        {status.is_training ? 'Trening w toku...' : 'Rozpocznij Trening'}
      </button>
      
      <div className="progress-container">
        <h3>Postęp uczenia: {status.progress}%</h3>
        <div className="progress-bar-bg">
          <div className="progress-bar-fill" style={{ width: `${status.progress}%` }}></div>
        </div>
        
        <div className="stats-grid">
          <div className="stat-box">
            <h3>Epoka</h3>
            <p>{status.epoch}</p>
          </div>
          <div className="stat-box">
            <h3>Loss (Błąd)</h3>
            <p>{status.loss.toFixed(4)}</p>
          </div>
          <div className="stat-box">
            <h3>Perplexity</h3>
            <p>{status.perplexity.toFixed(4)}</p>
          </div>
        </div>
      </div>
    </div>
  );
}

function VizTab() {
  const [iframeSrc, setIframeSrc] = useState('');
  const [loading, setLoading] = useState(false);

  const runViz = async (script, file) => {
    setLoading(true);
    setIframeSrc('');
    try {
      const apiUrl = window.location.port === '5173' ? 'http://127.0.0.1:5000/api/visualize' : '/api/visualize';
      await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ script, args: [] })
      });
      setIframeSrc(window.location.port === '5173' ? `http://127.0.0.1:5000/outputs/${file}?t=${Date.now()}` : `/outputs/${file}?t=${Date.now()}`);
    } catch (e) {
      alert("Błąd podczas generowania wizualizacji.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="visualize-panel">
      <div className="edu-box">
        <h4>📚 Czego uczy nas ta sekcja?</h4>
        <ul>
          <li><strong>Heatmapa Uwagi:</strong> Zobrazowanie macierzy prawdopodobieństw (Attention Score). Jasne pola oznaczają, że sieć uznała dwa słowa za mocno powiązane kontekstowo w zdaniu.</li>
          <li><strong>Wykres Embeddings:</strong> Rzutowanie wielowymiarowych wektorów z przestrzeni 128D na 2D używając algorytmu PCA z wykładów. Słowa o podobnym znaczeniu "zbierają się" blisko siebie!</li>
          <li><strong>Wykres Strat (Loss):</strong> Monitorowanie funkcji celu. Spadek wartości oznacza, że model popełnia coraz mniejszy błąd przy przewidywaniu następnego słowa.</li>
        </ul>
      </div>

      <div className="vis-buttons">
        <button className="vis-btn" onClick={() => runViz('visualize_architecture', 'architecture.html')} disabled={loading}>Architektura</button>
        <button className="vis-btn" onClick={() => runViz('visualize_attention', 'attention_heatmap.html')} disabled={loading}>Heatmapa Uwagi</button>
        <button className="vis-btn" onClick={() => runViz('visualize_embeddings', 'embeddings_2d.html')} disabled={loading}>Wykres Embeddings</button>
        <button className="vis-btn" onClick={() => runViz('plot_metrics', 'loss_curve.html')} disabled={loading}>Wykres Strat (Loss)</button>
      </div>
      
      {loading && <div style={{ color: 'var(--text-accent)' }}>Generowanie interaktywnego wykresu, proszę czekać...</div>}
      
      {iframeSrc && (
        <div className="iframe-container">
          <iframe src={iframeSrc} title="Wizualizacja"></iframe>
        </div>
      )}
    </div>
  );
}

function MathTab() {
  const [word, setWord] = useState('');
  const [result, setResult] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (result && window.MathJax && window.MathJax.typesetPromise) {
      window.MathJax.typesetPromise().catch(err => console.error("MathJax err", err));
    }
  }, [result]);

  const calculate = async (e) => {
    e.preventDefault();
    if (!word.trim() || loading) return;
    setLoading(true);
    setResult('');
    
    try {
      const apiUrl = window.location.port === '5173' ? 'http://127.0.0.1:5000/api/math_step' : '/api/math_step';
      const res = await fetch(apiUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word: word.trim() })
      });
      const data = await res.json();
      
      if (data.error) {
        setResult(`<span style="color:red">${data.error}</span>`);
      } else {
        setResult(data.math_html || "Brak danych");
      }
    } catch (err) {
      setResult("Błąd połączenia z serwerem.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="math-panel">
      <div className="edu-box">
        <h4>📚 Transformery "Pod Maską"</h4>
        <p><strong>Osadzenia (Embeddings):</strong> Sieć nie rozumie tekstu, dlatego każde słowo zamieniane jest na gęsty wektor liczb. Dodatkowo dodajemy "wektor pozycji" (PosEmb), by model wiedział, gdzie to słowo stoi.</p>
        <p><strong>Macierze Q, K, V:</strong> Transformacje liniowe (wymnażanie wektorów przez wyuczone wagi). <b>Q (Query)</b> pyta: "czego szukam?", <b>K (Key)</b> odpowiada: "mam tę cechę!", a <b>V (Value)</b> dostarcza właściwej treści.</p>
        <p><strong>Attention (Uwaga):</strong> Sieć sprawdza stopień dopasowania poprzez mnożenie $Q \times K^T$. Podobieństwo kosinusowe mówi nam, jak ważne dla bieżącego słowa są pozostałe!</p>
      </div>

      <form className="input-area" style={{ borderTop: 'none', background: 'transparent', padding: 0 }} onSubmit={calculate}>
        <input 
          type="text" 
          value={word} 
          onChange={(e) => setWord(e.target.value)} 
          placeholder="Wpisz słowo (ze słownika) do analizy np. test..." 
          disabled={loading}
        />
        <button type="submit" disabled={loading}>Analizuj wektory</button>
      </form>

      {loading && <div className="loading-dots"><span></span><span></span><span></span></div>}

      {result && (
        <div className="math-result" dangerouslySetInnerHTML={{ __html: result }} />
      )}
    </div>
  );
}

function App() {
  const [activeTab, setActiveTab] = useState('chat');

  return (
    <div className="app-container">
      <div className="header">
        <h1>Mini-GPT</h1>
        <p>Professional Neural Network Control Panel</p>
      </div>

      <div className="tabs-container">
        <button className={`tab ${activeTab === 'chat' ? 'active' : ''}`} onClick={() => setActiveTab('chat')}>💬 Czat</button>
        <button className={`tab ${activeTab === 'train' ? 'active' : ''}`} onClick={() => setActiveTab('train')}>⚙️ Trening</button>
        <button className={`tab ${activeTab === 'viz' ? 'active' : ''}`} onClick={() => setActiveTab('viz')}>📊 Wizualizacje</button>
        <button className={`tab ${activeTab === 'math' ? 'active' : ''}`} onClick={() => setActiveTab('math')}>🧮 Analiza Wejścia</button>
      </div>

      <div className="tab-content">
        {activeTab === 'chat' && <ChatTab />}
        {activeTab === 'train' && <TrainTab />}
        {activeTab === 'viz' && <VizTab />}
        {activeTab === 'math' && <MathTab />}
      </div>
    </div>
  );
}

export default App;
