import { useMemo, useState } from 'react';
import './App.css';
import ModelPerformance from './ModelPerformance';
import ResultsDisplay from './ResultsDisplay';
import LimeExplanation from './LimeExplanation';
import ShapExplanation from './ShapExplanation';

const API_BASE = import.meta.env.VITE_API_URL ?? 'https://api.sinxdetect.movindu.com';

/** True if the string contains at least one Sinhala script character (Unicode U+0D80–U+0DFF). */
function textContainsSinhalaScript(s) {
  return /[\u0D80-\u0DFF]/.test(s);
}

const LANGUAGE_WARNING =
  'This model is trained for Sinhala text only. Other languages are not supported.';

function App() {
  const [text, setText] = useState('');
  const [includeProbs, setIncludeProbs] = useState(true);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [languageWarning, setLanguageWarning] = useState('');
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [shapExplanation, setShapExplanation] = useState(null);
  const [shapLoading, setShapLoading] = useState(false);

  const wordCount = useMemo(() => {
    const trimmed = text.trim();
    if (!trimmed) return 0;
    return trimmed.split(/\s+/).length;
  }, [text]);

  const handleSubmit = async () => {
    if (!text.trim()) {
      setError('Please enter some Sinhala text first.');
      setLanguageWarning('');
      return;
    }
    if (!textContainsSinhalaScript(text)) {
      setLanguageWarning(LANGUAGE_WARNING);
      setError('');
      setResult(null);
      setExplanation(null);
      return;
    }
    setLoading(true);
    setError('');
    setLanguageWarning('');
    setResult(null);
    setExplanation(null);
    setShapExplanation(null);
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
      setLanguageWarning('');
      return;
    }
    if (!textContainsSinhalaScript(text)) {
      setLanguageWarning(LANGUAGE_WARNING);
      setError('');
      setResult(null);
      setExplanation(null);
      return;
    }
    setExplainLoading(true);
    setError('');
    setLanguageWarning('');
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

  const handleExplainShap = async () => {
    if (!text.trim()) {
      setError('Please enter some Sinhala text first.');
      setLanguageWarning('');
      return;
    }
    if (!textContainsSinhalaScript(text)) {
      setLanguageWarning(LANGUAGE_WARNING);
      setError('');
      setResult(null);
      setShapExplanation(null);
      return;
    }
    setShapLoading(true);
    setError('');
    setLanguageWarning('');
    setShapExplanation(null);
    try {
      const response = await fetch(`${API_BASE}/explain-shap`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, num_samples: 100 }),
      });
      if (!response.ok) {
        const detail = await response.text();
        throw new Error(detail || 'Request failed');
      }
      const data = await response.json();
      setShapExplanation(data);
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
      setShapLoading(false);
    }
  };

  const setSample = () => {
    setLanguageWarning('');
    setText(
      'දේශගුණික විපර්යාසය අද ලෝකය මුහුණ දෙන විශාලතම අභියෝගයන්ගෙන් එකකි. උෂ්ණත්වය ඉහළ යාම, අධික වැසි, වියළි කාල දිගුවීම, මුහුදු මට්ටම ඉහළ යාම සහ ජීව විවිධත්වය අඩුවීම වැනි ගැටලු මේ හේතුවෙන් වැඩිවෙමින් පවතී. එබැවින්, මෙම ගැටලුවට විසඳුම් සෙවීම සෑම රටකම සහ සෑම පුද්ගලයෙකුගේම වගකීමකි. දේශගුණික විපර්යාසයට ප්‍රධාන හේතුව වන්නේ ඉන්ධන දහනයෙන් සහ කර්මාන්ත ක්‍රියාකාරකම්වලින් වායුගෝලයට නිකුත් වන හරිතාගාර වායුය. ඒ නිසා පළමු විසඳුම වන්නේ අලුත්කරගත හැකි බලශක්ති ප්‍රභවයන් භාවිතය වැඩි කිරීමයි. සූර්ය බලශක්තිය, සුළං බලශක්තිය සහ ජල විදුලිය වැනි ක්‍රම භාවිතා කිරීමෙන් ගල්අඟුරු, තෙල් සහ ගෑස් වැනි ඉන්ධන මත ඇති ආශ්‍රිතාව අඩු කළ හැක. මෙය පරිසරයට හිතකර පියවරක් වන අතර දිගුකාලීනව ආර්ථික වශයෙන්ද වාසිදායකය. දෙවනුව, වනාන්තර ආරක්ෂා කිරීම සහ නැවත වනාන්තරකරණය ඉතා වැදගත්ය. ගස් කාබන් ඩයොක්සයිඩ් අවශෝෂණය කරන බැවින්, ගස් කැපීම අඩු කර නව ගස් රෝපණය කළ යුතුය. ගම්මාන, පාසල් සහ නගර මට්ටමින් වෘක්ෂ රෝපණ වැඩසටහන් ක්‍රියාත්මක කළහොත් පරිසරය සුරකින්නට මහත් දායකත්වයක් ලබා දිය හැක. තෙවනුව, අපගේ දෛනික ජීවිතයේ පුරුදු වෙනස් කළ යුතුය. විදුලිය අපතේ නොයවා භාවිතා කිරීම, පොදු ප්‍රවාහනය භාවිතා කිරීම, ප්ලාස්ටික් භාවිතය අඩු කිරීම, අපද්‍රව්‍ය වෙන් කර ප්‍රතිචක්‍රීකරණය කිරීම සහ ජලය සුරකිමින් භාවිතා කිරීම වැනි ක්‍රියා පරමාර්ථවත් වේ. කුඩා පියවරක් සේ පෙනුණත්, බොහෝ දෙනා එකතු වී මේවා අනුගමනය කළහොත් විශාල වෙනසක් ඇති කළ හැක. අවසාන වශයෙන්, දේශගුණික විපර්යාසයට විසඳුම් සෙවීම සරල කාර්යයක් නොවූවත්, එකමුතුව සහ වගකීම් සහිත ක්‍රියාමාර්ග මඟින් එය පාලනය කළ හැක. ආණ්ඩු, පෞද්ගලික ආයතන සහ සාමාන්‍ය ජනතාව එක්ව ක්‍රියා කළහොත්, අනාගත පරපුරට සුරක්ෂිත සහ සෞඛ්‍ය සම්පන්න පෘථිවියක් උරුම කර දිය හැක. දේශගුණික විපර්යාසය වැළැක්වීම සඳහා අද ගන්නා තීරණ හෙට දවසේ ජීවිතය තීරණය කරනු ඇත.'
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
            Enter Sinhala text and get AI-powered classification with SHAP explanations.
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
              onChange={(e) => {
                setText(e.target.value);
                setLanguageWarning('');
              }}
              rows={10}
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
          {languageWarning && (
            <div
              role="alert"
              className="rounded-lg border border-amber-300 bg-amber-50 px-4 py-3 text-sm text-amber-900"
            >
              {languageWarning}
            </div>
          )}
          {error && (
            <div className="rounded-lg border border-red-300 bg-red-50 px-4 py-3 text-sm text-red-800">
              {error}
            </div>
          )}
          <div className="flex flex-col gap-3">
            <div className="flex flex-wrap gap-3 items-center">
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
              {/* <button
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
              </button> */}
              <button
                type="button"
                onClick={handleExplainShap}
                disabled={shapLoading}
                className="inline-flex items-center justify-center gap-2 rounded-xl border-2 border-violet-600 bg-white px-5 py-3 font-medium text-violet-700 shadow-sm transition hover:bg-violet-50 disabled:cursor-not-allowed disabled:opacity-70"
              >
                {shapLoading && (
                  <span
                    className="inline-flex h-4 w-4 animate-spin rounded-full border-2 border-violet-300 border-t-violet-600"
                    aria-hidden="true"
                  />
                )}
                {shapLoading ? 'Explaining...' : 'Explain with SHAP'}
              </button>
            </div>
            <p className="text-xs text-gray-500">Backend: {API_BASE}</p>
          </div>{' '}
          {result && <ResultsDisplay result={result} text={text} />}
          {/* {explanation && <LimeExplanation explanation={explanation} text={text} />} */}
          {shapExplanation && <ShapExplanation explanation={shapExplanation} text={text} />}
        </main>
      </div>
    </div>
  );
}

export default App;
