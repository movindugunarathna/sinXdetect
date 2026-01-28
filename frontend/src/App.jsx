import { useMemo, useState } from 'react';
import './App.css';

const API_BASE = import.meta.env.VITE_API_URL ?? 'https://api.sinxdetect.movindu.com';

function App() {
  const [text, setText] = useState('');
  const [includeProbs, setIncludeProbs] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);

  const confidencePct = useMemo(() => {
    if (!result?.confidence) return null;
    return (result.confidence * 100).toFixed(2);
  }, [result]);
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
      // Also set the basic result
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
              className="inline-flex items-center justify-center gap-2 rounded-xl bg-purple-600 px-5 py-3 font-medium text-white shadow-lg shadow-purple-500/25 transition hover:bg-purple-700 disabled:cursor-not-allowed disabled:opacity-70"
            >
              {explainLoading && (
                <span
                  className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white"
                  aria-hidden="true"
                />
              )}
              {explainLoading ? 'Explaining...' : 'Explain with LIME'}
            </button>
            <p className="text-xs text-gray-500">Backend: {API_BASE}</p>
          </div>{' '}
          {result && (
            <section className="rounded-xl border border-gray-200 bg-white p-6 space-y-5 shadow-sm">
              <div className="flex flex-col items-center gap-6">
                <div className="flex flex-wrap items-center justify-center gap-3">
                  <span className="text-sm uppercase tracking-[0.15em] text-gray-500">
                    Prediction
                  </span>
                  <span className="rounded-full bg-gradient-to-r from-cyan-500 to-purple-500 px-4 py-1.5 text-base font-semibold text-white border border-cyan-300 shadow-md">
                    {result.label}
                  </span>
                </div>

                {/* Circular Confidence Indicator */}
                <div className="relative flex items-center justify-center">
                  <svg
                    className="transform -rotate-90"
                    width="160"
                    height="160"
                  >
                    {/* Background circle */}
                    <circle
                      cx="80"
                      cy="80"
                      r="70"
                      stroke="#e5e7eb"
                      strokeWidth="5"
                      fill="none"
                    />
                    {/* Progress circle */}
                    <circle
                      cx="80"
                      cy="80"
                      r="70"
                      stroke={result.label === 'HUMAN' ? '#10b981' : '#a855f7'}
                      strokeWidth="12"
                      fill="none"
                      strokeDasharray={`${2 * Math.PI * 70}`}
                      strokeDashoffset={`${
                        2 * Math.PI * 70 * (1 - result.confidence)
                      }`}
                      strokeLinecap="round"
                      className="transition-all duration-1000 ease-out"
                      style={{
                        filter: `drop-shadow(0 0 8px ${
                          result.label === 'HUMAN' ? '#10b98180' : '#a855f780'
                        })`,
                      }}
                    />
                  </svg>
                  <div className="absolute inset-0 flex flex-col items-center justify-center">
                    <span
                      className="text-3xl font-bold"
                      style={{
                        color: result.label === 'HUMAN' ? '#10b981' : '#a855f7',
                      }}
                    >
                      {confidencePct}%
                    </span>
                    <span className="text-xs text-gray-500 mt-1">
                      CONFIDENCE
                    </span>
                  </div>
                </div>
              </div>

              {result.probabilities && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm text-gray-700 pt-4 border-t border-gray-200">
                  <div className="rounded-lg bg-emerald-50 border border-emerald-200 px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-emerald-500"></div>
                      <span className="font-medium">HUMAN</span>
                    </div>
                    <span className="font-semibold text-emerald-600">
                      {(result.probabilities.HUMAN * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="rounded-lg bg-purple-50 border border-purple-200 px-4 py-3 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 rounded-full bg-purple-500"></div>
                      <span className="font-medium">AI</span>
                    </div>
                    <span className="font-semibold text-purple-600">
                      {(result.probabilities.AI * 100).toFixed(2)}%
                    </span>
                  </div>
                </div>
              )}
            </section>
          )}
          {/* LIME Explanation Section */}
          {explanation && (
            <section className="rounded-xl border border-gray-200 bg-white p-6 space-y-5 shadow-sm">
              <div className="flex items-center gap-2 border-b border-gray-200 pb-3">
                <svg
                  className="w-5 h-5 text-purple-600"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
                  />
                </svg>
                <h2 className="text-lg font-semibold text-gray-800">
                  LIME Explanation
                </h2>
                {explanation.error && (
                  <span className="text-xs text-amber-600 bg-amber-50 px-2 py-1 rounded-md border border-amber-200">
                    {explanation.error}
                  </span>
                )}
              </div>

              {/* Highlighted Text */}
              {explanation.highlighted_text &&
                explanation.highlighted_text.length > 0 && (
                  <div className="space-y-4">
                    <div className="flex items-center justify-between">
                      <h3 className="text-sm font-medium text-gray-700">
                        Important Words & Phrases
                      </h3>
                      <div className="flex items-center gap-3 text-xs">
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 rounded bg-red-100 border border-red-400"></div>
                          <span className="text-gray-600">AI-generated</span>
                        </div>
                        <div className="flex items-center gap-1">
                          <div className="w-3 h-3 rounded bg-green-100 border border-green-400"></div>
                          <span className="text-gray-600">Human-written</span>
                        </div>
                      </div>
                    </div>

                    <div className="space-y-2">
                      {explanation.highlighted_text
                        .slice(0, 10)
                        .map((item, idx) => (
                          <div
                            key={idx}
                            className={`rounded-lg border p-3 ${
                              item.color === 'red'
                                ? 'bg-red-50 border-red-300'
                                : 'bg-green-50 border-green-300'
                            }`}
                          >
                            <div className="flex items-start justify-between gap-3">
                              <div className="flex-1">
                                <div className="flex items-center gap-2 mb-1">
                                  <span
                                    className={`text-sm font-medium ${
                                      item.color === 'red'
                                        ? 'text-red-800'
                                        : 'text-green-800'
                                    }`}
                                  >
                                    {item.phrase}
                                  </span>
                                  <span
                                    className={`text-xs px-2 py-0.5 rounded-full ${
                                      item.color === 'red'
                                        ? 'bg-red-200 text-red-700'
                                        : 'bg-green-200 text-green-700'
                                    }`}
                                  >
                                    {item.word_count} word
                                    {item.word_count > 1 ? 's' : ''}
                                  </span>
                                </div>
                                <p className="text-xs text-gray-600">
                                  Indicates:{' '}
                                  <span className="font-medium">
                                    {item.indicates}
                                  </span>
                                </p>
                              </div>
                              <div className="text-right">
                                <div
                                  className={`text-lg font-bold ${
                                    item.color === 'red'
                                      ? 'text-red-600'
                                      : 'text-green-600'
                                  }`}
                                >
                                  {(Math.abs(item.weight) * 100).toFixed(1)}%
                                </div>
                                <div className="text-xs text-gray-500">
                                  importance
                                </div>
                              </div>
                            </div>
                          </div>
                        ))}
                    </div>

                    {explanation.highlighted_text.length > 10 && (
                      <p className="text-xs text-gray-500 text-center pt-2">
                        Showing top 10 of {explanation.highlighted_text.length}{' '}
                        important phrases
                      </p>
                    )}
                  </div>
                )}

              {/* Original Text with Inline Highlighting */}
              {explanation.highlighted_text &&
                explanation.highlighted_text.length > 0 && (
                  <div className="space-y-2 border-t border-gray-200 pt-4">
                    <h3 className="text-sm font-medium text-gray-700">
                      Highlighted Text
                    </h3>
                    <div className="rounded-lg bg-gray-50 border border-gray-200 p-4">
                      <div
                        className="text-base leading-relaxed"
                        style={{ direction: 'ltr' }}
                      >
                        {renderHighlightedText(
                          text,
                          explanation.highlighted_text
                        )}
                      </div>
                    </div>
                  </div>
                )}

              {(!explanation.highlighted_text ||
                explanation.highlighted_text.length === 0) &&
                !explanation.error && (
                  <div className="text-center py-8 text-gray-500">
                    <p>
                      No significant word contributions found for this text.
                    </p>
                  </div>
                )}
            </section>
          )}
        </main>
      </div>
    </div>
  );
}

// Helper function to render highlighted text
function renderHighlightedText(originalText, highlights) {
  if (!highlights || highlights.length === 0) {
    return <span className="text-gray-800">{originalText}</span>;
  }

  // Sort highlights by start position
  const sortedHighlights = [...highlights].sort((a, b) => a.start - b.start);

  const elements = [];
  let lastIndex = 0;

  sortedHighlights.forEach((highlight, idx) => {
    // Add text before highlight
    if (highlight.start > lastIndex) {
      elements.push(
        <span key={`text-${idx}`} className="text-gray-800">
          {originalText.substring(lastIndex, highlight.start)}
        </span>
      );
    }

    // Add highlighted text
    const highlightedText = originalText.substring(
      highlight.start,
      highlight.end
    );
    elements.push(
      <span
        key={`highlight-${idx}`}
        className={`px-1 py-0.5 rounded font-medium ${
          highlight.color === 'red'
            ? 'bg-red-200 text-red-900 border-b-2 border-red-500'
            : 'bg-green-200 text-green-900 border-b-2 border-green-500'
        }`}
        title={`${highlight.indicates}: ${(
          Math.abs(highlight.weight) * 100
        ).toFixed(1)}% importance`}
      >
        {highlightedText}
      </span>
    );

    lastIndex = highlight.end;
  });

  // Add remaining text
  if (lastIndex < originalText.length) {
    elements.push(
      <span key="text-end" className="text-gray-800">
        {originalText.substring(lastIndex)}
      </span>
    );
  }

  return elements;
}

export default App;
