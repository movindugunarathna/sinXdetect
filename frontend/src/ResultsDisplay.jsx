import ConfidenceGauge from './ConfidenceGauge';
import FeedbackSection from './FeedbackSection';

function ResultsDisplay({ result, text }) {
  return (
    <section className="rounded-xl border border-gray-200 bg-white p-6 space-y-5 shadow-sm">
      <div className="flex flex-col items-center gap-6">
        <div className="flex flex-wrap items-center justify-center gap-3">
          <span className="text-sm uppercase tracking-[0.15em] text-gray-500">
            Prediction
          </span>
          <span className={`rounded-full px-4 py-1.5 text-base font-semibold text-white shadow-md ${result.label === 'HUMAN' ? 'bg-emerald-500' : 'bg-red-500'}`}>
            {result.label}
          </span>
        </div>

        <ConfidenceGauge label={result.label} confidence={result.confidence} />
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
          <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span className="font-medium">AI</span>
            </div>
            <span className="font-semibold text-red-600">
              {(result.probabilities.AI * 100).toFixed(2)}%
            </span>
          </div>
        </div>
      )}

      <FeedbackSection result={result} text={text} />
    </section>
  );
}

export default ResultsDisplay;
