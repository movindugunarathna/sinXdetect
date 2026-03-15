import { useEffect, useState } from 'react';

const API_BASE =
  import.meta.env.VITE_API_URL ?? 'https://api.sinxdetect.movindu.com';

function MetricPill({ label, value }) {
  const pct = (value * 100).toFixed(2);
  return (
    <div className="flex flex-col items-center rounded-lg border border-gray-200 bg-gray-50 px-4 py-3 min-w-[100px]">
      <span className="text-xs uppercase tracking-wider text-gray-500">
        {label}
      </span>
      <span className="text-xl font-bold text-cyan-700">{pct}%</span>
    </div>
  );
}

function ConfusionMatrix({ data }) {
  if (!data?.matrix || !data?.labels) return null;
  const { labels, matrix } = data;
  const maxVal = Math.max(...matrix.flat());

  return (
    <div className="space-y-2">
      <h4 className="text-xs font-medium uppercase tracking-wider text-gray-500">
        Confusion Matrix
      </h4>
      <div className="overflow-x-auto">
        <table className="mx-auto text-sm">
          <thead>
            <tr>
              <th className="px-2 py-1" />
              <th
                colSpan={labels.length}
                className="px-2 py-1 text-center text-xs text-gray-500 font-medium"
              >
                Predicted
              </th>
            </tr>
            <tr>
              <th className="px-2 py-1" />
              {labels.map((l) => (
                <th
                  key={l}
                  className="px-3 py-1 text-center text-xs font-semibold text-gray-700"
                >
                  {l}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {matrix.map((row, ri) => (
              <tr key={ri}>
                {ri === 0 && (
                  <td
                    rowSpan={matrix.length}
                    className="pr-2 text-xs text-gray-500 font-medium align-middle"
                    style={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}
                  >
                    Actual
                  </td>
                )}
                <td className="px-3 py-1 text-xs font-semibold text-gray-700">
                  {labels[ri]}
                </td>
                {row.map((val, ci) => {
                  const isDiag = ri === ci;
                  const intensity = maxVal > 0 ? val / maxVal : 0;
                  const bg = isDiag
                    ? `rgba(16, 185, 129, ${0.08 + intensity * 0.25})`
                    : `rgba(239, 68, 68, ${0.05 + intensity * 0.15})`;
                  return (
                    <td
                      key={ci}
                      className="px-3 py-2 text-center font-mono font-semibold rounded"
                      style={{ backgroundColor: bg }}
                    >
                      {val}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function ModelPerformance() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const res = await fetch(`${API_BASE}/metrics/current`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json = await res.json();
        if (!cancelled) setData(json);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="glass-card rounded-2xl p-6 animate-pulse">
        <div className="h-4 w-40 bg-gray-200 rounded mb-4" />
        <div className="h-20 bg-gray-100 rounded" />
      </div>
    );
  }

  if (error || !data?.evaluation) {
    return null;
  }

  const { model, evaluation: ev } = data;

  return (
    <div className="glass-card rounded-2xl p-6 sm:p-8 space-y-5">
      <button
        type="button"
        onClick={() => setExpanded((p) => !p)}
        className="w-full flex items-center justify-between text-left"
      >
        <div className="flex items-center gap-3">
          <svg
            className="w-5 h-5 text-cyan-600"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"
            />
          </svg>
          <div>
            <h2 className="text-lg font-semibold text-gray-800">
              Model Performance
            </h2>
            <p className="text-xs text-gray-500">
              {model.version_name}
              {model.base_model && (
                <span className="ml-1 text-gray-400">
                  ({model.base_model})
                </span>
              )}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className="hidden sm:inline-flex items-center gap-1 rounded-full bg-cyan-50 border border-cyan-200 px-3 py-1 text-xs font-medium text-cyan-700">
            F1 {(ev.f1_score * 100).toFixed(1)}%
          </span>
          <svg
            className={`w-5 h-5 text-gray-400 transition-transform duration-200 ${expanded ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </button>

      {expanded && (
        <div className="space-y-6 pt-2">
          <div className="flex flex-wrap justify-center gap-3">
            <MetricPill label="Accuracy" value={ev.accuracy} />
            <MetricPill label="Precision" value={ev.precision} />
            <MetricPill label="Recall" value={ev.recall} />
            <MetricPill label="F1 Score" value={ev.f1_score} />
          </div>

          <ConfusionMatrix data={ev.confusion_matrix} />

          {ev.classification_report && (
            <div className="space-y-2">
              <h4 className="text-xs font-medium uppercase tracking-wider text-gray-500">
                Per-class Metrics
              </h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {Object.entries(ev.classification_report).map(
                  ([label, metrics]) => (
                    <div
                      key={label}
                      className={`rounded-lg border px-4 py-3 ${
                        label === 'HUMAN'
                          ? 'bg-emerald-50 border-emerald-200'
                          : 'bg-red-50 border-red-200'
                      }`}
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <div
                          className={`w-2.5 h-2.5 rounded-full ${
                            label === 'HUMAN'
                              ? 'bg-emerald-500'
                              : 'bg-red-500'
                          }`}
                        />
                        <span className="text-sm font-semibold text-gray-800">
                          {label}
                        </span>
                        <span className="text-xs text-gray-500">
                          ({metrics.support} samples)
                        </span>
                      </div>
                      <div className="grid grid-cols-3 gap-2 text-center text-xs">
                        <div>
                          <div className="text-gray-500">Prec</div>
                          <div className="font-semibold text-gray-800">
                            {(metrics.precision * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500">Rec</div>
                          <div className="font-semibold text-gray-800">
                            {(metrics.recall * 100).toFixed(1)}%
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500">F1</div>
                          <div className="font-semibold text-gray-800">
                            {(metrics.f1_score * 100).toFixed(1)}%
                          </div>
                        </div>
                      </div>
                    </div>
                  )
                )}
              </div>
            </div>
          )}

          <div className="flex flex-wrap items-center justify-between text-xs text-gray-400 border-t border-gray-100 pt-3">
            <span>
              Dataset: {ev.dataset_name} ({ev.total_samples} samples)
            </span>
            <span>
              Evaluated:{' '}
              {new Date(ev.evaluated_at).toLocaleDateString('en-US', {
                year: 'numeric',
                month: 'short',
                day: 'numeric',
              })}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
