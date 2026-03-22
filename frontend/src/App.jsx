import { useMemo, useState } from 'react';
import './App.css';
import ModelPerformance from './ModelPerformance';
import ResultsDisplay from './ResultsDisplay';
import LimeExplanation from './LimeExplanation';

const API_BASE = import.meta.env.VITE_API_URL ?? 'https://api.sinxdetect.movindu.com';

function App() {
  const [text, setText] = useState('');
  const [includeProbs, setIncludeProbs] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const wordCount = useMemo(() => {
    const trimmed = text.trim();
    if (!trimmed) return 0;
    return trimmed.split(/\s+/).length;
  }, [text]);

  const handleSubmit = async () => {
    if (!text.trim()) {
      setError('Please enter some Sinhala text first.');
      return;
    }
    setLoading(true);
    setError('');
    setResult(null);
    setExplanation(null);
    try {
      const response = await fetch(`${API_BASE}/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, return_probabilities: includeProbs }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || 'Request failed');
      }
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setLoading(false);
    }
  };

  const handleExplain = async () => {
    if (!text.trim()) {
      setError('Please enter some Sinhala text first.');
      return;
    }
    setExplainLoading(true);
    setError('');
    setExplanation(null);
    try {
      const response = await fetch(`${API_BASE}/explain`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, num_samples: 100 }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || 'Request failed');
      }
      const data = await response.json();
      setExplanation(data);
      setResult({
        label: data.predicted_class === 'AI-generated' ? 'AI' : 'HUMAN',
        confidence: data.confidence,
        probabilities: {
          HUMAN: data.explanation_data.predicted_probability[0],
          AI: data.explanation_data.predicted_probability[1],
        },
      });
    } catch (err) {
      setError(err.message || 'Something went wrong');
    } finally {
      setExplainLoading(false);
    }
  };

  const setSample = () => {
    setText(
      'සිංහල භාෂාවෙන් යුතු මනුෂ්‍ය ලියූ වාක්‍යයක් උදාහරණයක් ලෙස මෙහි සදහන් වේ.'
    );
  };

  return (
    <div className="min-h-screen bg-gray-50 px-4 py-10">
      <div className="max-w-5xl mx-auto space-y-6">
        <header className="text-center space-y-2">
          <p className="text-sm uppercase tracking-[0.2em] text-gray-500">
            Sinhala Human vs AI
          </p>
          <h1 className="text-3xl sm:text-4xl font-bold text-gradient">
            Text Classifier
          </h1>
          <p className="text-gray-600 text-sm sm:text-base">
            Enter Sinhala text and get AI-powered classification with word-level
            explanations.
          </p>
        </header>

        <ModelPerformance />

        <main className="glass-card rounded-2xl p-6 sm:p-8 space-y-5">
          <div className="flex flex-col gap-3">
            <label className="text-sm font-medium text-gray-700">
              Text to classify
            </label>
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={5}
              className="w-full rounded-xl border border-gray-300 bg-white px-4 py-3 text-base text-gray-800 placeholder:text-gray-400 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/20"
              placeholder="සිංහල පෙළ මෙහි පුරන්න"
            />
            <div className="flex items-center justify-between">
              <p className={`text-xs ${wordCount >= 150 ? 'text-cyan-600' : wordCount > 0 ? 'text-gray-500' : 'text-gray-400'}`}>
                {wordCount > 0 ? (
                  <>
                    <span className="font-medium">{wordCount}</span> word{wordCount !== 1 ? 's' : ''}
                    {wordCount < 150 && (
                      <span> &middot; Enter at least 150 words for better results</span>
                    )}
                  </>
                ) : (
                  'Enter at least 150 words for better results'
                )}
              </p>
              {wordCount > 0 && wordCount < 150 && (
                <span className="text-xs text-gray-400 font-medium tabular-nums">
                  {150 - wordCount} more needed
                </span>
              )}
            </div>
            <div className="flex flex-wrap gap-3">
              <button
                type="button"
                onClick={setSample}
                className="px-3 py-2 text-sm rounded-lg border border-gray-300 text-gray-700 hover:border-cyan-500 hover:text-cyan-600 hover:bg-cyan-50 transition"
              >
                Fill sample text
              </button>
              <label className="inline-flex items-center gap-2 text-sm text-gray-700">
                <input
                  type="checkbox"
                  checked={includeProbs}
                  onChange={(e) => setIncludeProbs(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300 text-cyan-600 focus:ring-cyan-500"
                />
                Return probabilities
              </label>
            </div>
          </div>
          {error && (
            <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
              {error}
            </div>
          )}
          <div className="flex flex-col sm:flex-row gap-3 sm:items-center">
            <button
              type="button"
              onClick={handleSubmit}
              disabled={loading}
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-cyan-600 px-5 py-3 font-medium text-white shadow-lg shadow-cyan-500/25 transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {loading && (
                <span
                  className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white"
                  aria-hidden="true"
                />
              )}
              {loading ? 'Classifying...' : 'Classify'}
            </button>
            <button
              type="button"
              onClick={handleExplain}
              disabled={explainLoading}
              className="inline-flex items-center justify-center gap-2 rounded-xl border-2 border-cyan-600 bg-white px-5 py-3 font-medium text-cyan-700 shadow-sm transition hover:bg-cyan-50 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {explainLoading && (
                <span
                  className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-cyan-300 border-t-cyan-600"
                  aria-hidden="true"
                />
              )}
              {explainLoading ? 'Explaining...' : 'Explain with LIME'}
            </button>
            <p className="text-xs text-gray-500">Backend: {API_BASE}</p>
          </div>{' '}
          {result && <ResultsDisplay result={result} text={text} />}
          {explanation && <LimeExplanation explanation={explanation} text={text} />}
        </main>
      </div>
    </div>
  );
}

export default App;
