import { useMemo } from 'react';

function ConfidenceGauge({ label, confidence }) {
  const confidencePct = useMemo(() => {
    if (!confidence) return null;
    return (confidence * 100).toFixed(2);
  }, [confidence]);

  const isHuman = label === 'HUMAN';
  const color = isHuman ? '#10b981' : '#ef4444';
  const textColor = isHuman ? 'text-emerald-600' : 'text-red-600';
  const glowColor = isHuman ? 'rgba(16,185,129,0.5)' : 'rgba(239,68,68,0.5)';

  return (
    <div className="relative flex items-center justify-center">
      <svg className="transform -rotate-90" width="160" height="160">
        <circle
          cx="80"
          cy="80"
          r="70"
          stroke="#e5e7eb"
          strokeWidth="5"
          fill="none"
        />
        <circle
          cx="80"
          cy="80"
          r="70"
          stroke={color}
          strokeWidth="12"
          fill="none"
          strokeDasharray={`${2 * Math.PI * 70}`}
          strokeDashoffset={`${2 * Math.PI * 70 * (1 - confidence)}`}
          strokeLinecap="round"
          className="transition-all duration-1000 ease-out"
          style={{ filter: `drop-shadow(0 0 8px ${glowColor})` }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={`text-3xl font-bold ${textColor}`}>
          {confidencePct}%
        </span>
        <span className="text-xs text-gray-500 mt-1">CONFIDENCE</span>
      </div>
    </div>
  );
}

export default ConfidenceGauge;
