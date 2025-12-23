import { useMemo, useState } from "react";
import "./App.css";

const API_BASE = import.meta.env.VITE_API_URL ?? "http://localhost:8000";

function App() {
  const [text, setText] = useState("");
  const [includeProbs, setIncludeProbs] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const confidencePct = useMemo(() => {
    if (!result?.confidence) return null;
    return (result.confidence * 100).toFixed(2);
  }, [result]);

  const handleSubmit = async () => {
    if (!text.trim()) {
      setError("Please enter some Sinhala text first.");
      return;
    }
    setLoading(true);
    setError("");
    setResult(null);
    try {
      const response = await fetch(`${API_BASE}/classify`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, return_probabilities: includeProbs }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || "Request failed");
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || "Something went wrong");
    } finally {
      setLoading(false);
    }
  };

  const setSample = () => {
    setText("සිංහල භාෂාවෙන් යුතු මනුෂ්‍ය ලියූ වාක්‍යයක් උදාහරණයක් ලෙස මෙහි සදහන් වේ.");
  };

  return (
    <div className="min-h-screen text-slate-50 px-4 py-10">
      <div className="max-w-4xl mx-auto space-y-6">
        <header className="text-center space-y-2">
          <p className="text-sm uppercase tracking-[0.2em] text-slate-400">Sinhala Human vs AI</p>
          <h1 className="text-3xl sm:text-4xl font-semibold text-gradient">Text Classifier</h1>
          <p className="text-slate-400 text-sm sm:text-base">
            Enter Sinhala text, send it to the FastAPI backend, and view the predicted label and confidence.
          </p>
        </header>

        <main className="glass-card rounded-2xl p-6 sm:p-8 space-y-5">
          <div className="flex flex-col gap-3">
            <label className="text-sm font-medium text-slate-200">Text to classify</label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={5}
              className="w-full rounded-xl border border-white/10 bg-white/5 px-4 py-3 text-base text-slate-100 placeholder:text-slate-500 focus:border-cyan-400 focus:outline-none focus:ring-1 focus:ring-cyan-400"
              placeholder="සිංහල පෙළ මෙහි පුරන්න"
            />
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={setSample}
                className="px-3 py-2 text-sm rounded-lg border border-white/10 text-slate-200 hover:border-cyan-400 hover:text-cyan-200 transition"
              >
                Fill sample text
              </button>
              <label className="inline-flex items-center gap-2 text-sm text-slate-200">
                <input
                  type="checkbox"
                  checked={includeProbs}
                  onChange={(e) => setIncludeProbs(e.target.checked)}
                  className="h-4 w-4 rounded border-slate-500 bg-transparent text-cyan-400 focus:ring-cyan-400"
                />
                Return probabilities
              </label>
            </div>
          </div>

          {error && (
            <div className="rounded-lg border border-red-500/40 bg-red-500/10 px-4 py-3 text-sm text-red-100">
              {error}
            </div>
          )}

          <div className="flex flex-col sm:flex-row gap-3 sm:items-center">
            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-cyan-500 px-5 py-3 font-medium text-slate-900 shadow-lg shadow-cyan-500/25 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {loading && (
                <span className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" aria-hidden="true" />
              )}
              {loading ? "Classifying..." : "Classify"}
            </button>
            <p className="text-xs text-slate-400">Backend: {API_BASE}/classify</p>
          </div>

          {result && (
            <section className="rounded-xl border border-white/10 bg-white/5 p-6 space-y-5">
              <div className="flex flex-col items-center gap-6">
                <div className="flex flex-wrap items-center justify-center gap-3">
                  <span className="text-sm uppercase tracking-[0.15em] text-slate-400">Prediction</span>
                  <span className="rounded-full bg-slate-900 px-4 py-1.5 text-base font-semibold text-cyan-200 border border-cyan-400/40">
                    {result.label}
                  </span>
                </div>

                {/* Circular Confidence Indicator */}
                <div className="relative flex items-center justify-center">
                  <svg className="transform -rotate-90" width="160" height="160">
                    {/* Background circle */}
                    <circle
                      cx="80"
                      cy="80"
                      r="70"
                      stroke="rgba(255, 255, 255, 0.1)"
                      strokeWidth="5"
                      fill="none"
                    />
                    {/* Progress circle */}
                    <circle
                      cx="80"
                      cy="80"
                      r="70"
                      stroke={result.label === "HUMAN" ? "#10b981" : "#c084fc"}
                      strokeWidth="12"
                      fill="none"
                      strokeDasharray={`${2 * Math.PI * 70}`}
                      strokeDashoffset={`${2 * Math.PI * 70 * (1 - result.confidence)}`}
                      strokeLinecap="round"
                      className="transition-all duration-1000 ease-out"
                      style={{
                        filter: `drop-shadow(0 0 8px ${result.label === "HUMAN" ? "#10b98180" : "#c084fc80"})`
                      }}
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span className="text-3xl font-bold" style={{ color: result.label === "HUMAN" ? "#10b981" : "#c084fc" }}>
                      {confidencePct}%
                    </span>
                    <span className="text-xs text-slate-400 mt-1">CONFIDENCE</span>
                  </div>
                </div>
              </div>

              {result.probabilities && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm text-slate-200 pt-4 border-t border-white/10">
                  <div className="rounded-lg bg-slate-900/60 border border-white/5 px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                      <span>HUMAN</span>
                    </div>
                    <span className="font-semibold text-emerald-400">{(result.probabilities.HUMAN * 100).toFixed(2)}%</span>
                  </div>
                  <div className="rounded-lg bg-slate-900/60 border border-white/5 px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-purple-400"></div>
                      <span>AI</span>
                    </div>
                    <span className="font-semibold text-purple-400">{(result.probabilities.AI * 100).toFixed(2)}%</span>
                  </div>
                </div>
              )}
            </section>
          )}
        </main>
      </div>
    </div>
  );
}

export default App;
