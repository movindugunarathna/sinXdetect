function ShapExplanation({ explanation, text }) {
  const summary = explanation.evidence_summary;
  const sentences = explanation.sentence_explanations || [];
  const significantSentences = sentences
    .filter((s) => s.color !== 'neutral')
    .sort((a, b) => Math.abs(b.net_weight) - Math.abs(a.net_weight));
  const hasContent = sentences.length > 0;

  // SHAP-specific data
  const shapData = explanation.explanation_data || {};
  const baseValues = shapData.base_values;
  const shapTokens = shapData.tokens || [];
  const aiShapValues = shapData.shap_values?.['AI-generated'] || [];

  // Top contributing tokens by absolute Shapley value
  const topTokens = aiShapValues
    .map((val, idx) => ({ token: shapTokens[idx], value: val, idx }))
    .filter((t) => t.token && Math.abs(t.value) > 0.001)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
    .slice(0, 10);

  return (
    <section className="rounded-xl border border-gray-200 bg-white p-6 space-y-5 shadow-sm">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-gray-200 pb-3">
        <svg
          className="w-5 h-5 text-violet-600"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"
          />
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"
          />
        </svg>
        <h2 className="text-lg font-semibold text-gray-800">
          SHAP Explanation
        </h2>
        <span className="text-xs text-violet-600 bg-violet-50 px-2 py-0.5 rounded-md border border-violet-200 font-medium">
          Shapley Values
        </span>
        {explanation.error && (
          <span className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded-md border border-gray-200">
            {explanation.error}
          </span>
        )}
      </div>

      {/* Evidence Summary Bar
      {summary && (
        <div className="space-y-2">
          <div className="flex items-center justify-between text-xs font-medium text-gray-600">
            <span>Evidence Balance</span>
            <span>{summary.total_important_words} important words detected</span>
          </div>
          <div className="flex h-4 rounded-full overflow-hidden border border-gray-200">
            <div
              className="bg-emerald-400 transition-all duration-700"
              style={{ width: `${(summary.human_ratio * 100).toFixed(1)}%` }}
              title={`Human: ${(summary.human_ratio * 100).toFixed(1)}%`}
            />
            <div
              className="bg-red-400 transition-all duration-700"
              style={{ width: `${(summary.ai_ratio * 100).toFixed(1)}%` }}
              title={`AI: ${(summary.ai_ratio * 100).toFixed(1)}%`}
            />
          </div>
          <div className="flex justify-between text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-2.5 h-2.5 rounded-full bg-emerald-400"></div>
              <span className="text-emerald-700 font-medium">
                Human-written {(summary.human_ratio * 100).toFixed(0)}%
              </span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="text-red-700 font-medium">
                AI-generated {(summary.ai_ratio * 100).toFixed(0)}%
              </span>
              <div className="w-2.5 h-2.5 rounded-full bg-red-400"></div>
            </div>
          </div>
        </div>
      )} */}

      {/* SHAP Base Values
      {baseValues && baseValues.length === 2 && (
        <div className="rounded-lg bg-violet-50 border border-violet-200 px-4 py-3">
          <p className="text-xs font-medium text-violet-700 mb-1">
            SHAP Baseline (expected output with no tokens)
          </p>
          <div className="flex gap-6 text-sm">
            <span className="text-emerald-700">
              Human: <span className="font-semibold">{(baseValues[0] * 100).toFixed(1)}%</span>
            </span>
            <span className="text-red-700">
              AI: <span className="font-semibold">{(baseValues[1] * 100).toFixed(1)}%</span>
            </span>
          </div>
        </div>
      )} */}

      {/* Top Contributing Tokens (SHAP-specific)
      {topTokens.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">
            Top Token Shapley Values
          </h3>
          <div className="space-y-1.5">
            {topTokens.map((t, idx) => {
              const isAI = t.value > 0;
              const maxAbsVal = Math.abs(topTokens[0].value) || 0.01;
              const barWidth = (Math.abs(t.value) / maxAbsVal) * 100;
              return (
                <div key={idx} className="flex items-center gap-3 text-sm">
                  <span className="w-36 truncate text-right font-medium text-gray-700" title={t.token}>
                    {t.token}
                  </span>
                  <div className="flex-1 flex items-center h-5">
                    <div
                      className={`h-4 rounded-sm transition-all duration-500 ${
                        isAI ? 'bg-red-400' : 'bg-emerald-400'
                      }`}
                      style={{ width: `${barWidth.toFixed(1)}%`, minWidth: '4px' }}
                    />
                  </div>
                  <span
                    className={`w-20 text-right text-xs font-medium tabular-nums ${
                      isAI ? 'text-red-600' : 'text-emerald-600'
                    }`}
                  >
                    {t.value > 0 ? '+' : ''}{t.value.toFixed(4)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="text-xs text-gray-400 pt-1">
            Positive values push toward AI-generated, negative toward Human-written.
          </p>
        </div>
      )} */}

      {/* Highlighted Full Text */}
      {sentences.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700">
            Highlighted Text
          </h3>
          <div className="rounded-lg bg-gray-50 border border-gray-200 p-4">
            <div
              className="text-base leading-[1.9]"
              style={{ direction: 'ltr' }}
            >
              {renderSentenceHighlightedText(text, sentences)}
            </div>
          </div>
          <div className="flex items-center justify-center gap-4 text-xs text-gray-500 pt-1">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-red-200 border border-red-300"></div>
              <span>AI-generated</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-emerald-200 border border-emerald-300"></div>
              <span>Human-written</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 rounded bg-white border border-gray-300"></div>
              <span>Neutral</span>
            </div>
          </div>
        </div>
      )}

      {/* Sentence-Level Breakdown */}
      {significantSentences.length > 0 && (
        <div className="space-y-3 border-t border-gray-200 pt-4">
          <h3 className="text-sm font-medium text-gray-700">
            Sentence-by-Sentence Analysis
          </h3>
          <div className="space-y-2">
            {significantSentences.map((sent, idx) => (
              <div
                key={idx}
                className={`rounded-lg border p-3 ${
                  sent.color === 'red'
                    ? 'bg-red-50 border-red-200'
                    : 'bg-emerald-50 border-emerald-200'
                }`}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="flex-1 min-w-0">
                    <p
                      className={`text-sm leading-relaxed ${
                        sent.color === 'red' ? 'text-red-900' : 'text-emerald-900'
                      }`}
                    >
                      {sent.sentence}
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      {sent.important_words} of {sent.total_words} words contribute
                      {' \u2192 '}
                      <span className="font-medium">{sent.indicates}</span>
                    </p>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <div
                      className={`text-lg font-bold ${
                        sent.color === 'red' ? 'text-red-600' : 'text-emerald-600'
                      }`}
                    >
                      {(Math.abs(sent.net_weight) * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-500">strength</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {!hasContent && !explanation.error && (
        <div className="text-center py-8 text-gray-500">
          <p>No significant word contributions found for this text.</p>
        </div>
      )}
    </section>
  );
}

/**
 * Render the full text with sentence-level background highlighting.
 * Replicates the same approach used in LimeExplanation for visual consistency.
 */
function renderSentenceHighlightedText(originalText, sentenceHighlights) {
  if (!sentenceHighlights || sentenceHighlights.length === 0) {
    return <span className="text-gray-800">{originalText}</span>;
  }

  const sorted = [...sentenceHighlights].sort((a, b) => a.start - b.start);

  const cleaned = [];
  for (const s of sorted) {
    if (cleaned.length === 0 || s.start >= cleaned[cleaned.length - 1].end) {
      cleaned.push(s);
    }
  }

  const maxWeight = Math.max(...cleaned.map((s) => Math.abs(s.net_weight)), 0.01);

  const elements = [];
  let lastIndex = 0;

  cleaned.forEach((sent, idx) => {
    if (sent.start > lastIndex) {
      elements.push(
        <span key={`gap-${idx}`} className="text-gray-800">
          {originalText.substring(lastIndex, sent.start)}
        </span>
      );
    }

    const sentText = originalText.substring(sent.start, sent.end);

    if (sent.color === 'neutral') {
      elements.push(
        <span key={`sent-${idx}`} className="text-gray-800">
          {sentText}
        </span>
      );
      lastIndex = sent.end;
      return;
    }

    const ratio = Math.abs(sent.net_weight) / maxWeight;
    const opacity = 0.15 + ratio * 0.4;

    const bgColor =
      sent.color === 'red'
        ? `rgba(239, 68, 68, ${opacity.toFixed(2)})`
        : `rgba(16, 185, 129, ${opacity.toFixed(2)})`;

    const borderColor =
      sent.color === 'red'
        ? `rgba(239, 68, 68, ${(opacity + 0.15).toFixed(2)})`
        : `rgba(16, 185, 129, ${(opacity + 0.15).toFixed(2)})`;

    elements.push(
      <span
        key={`sent-${idx}`}
        className="rounded-sm py-0.5"
        style={{
          backgroundColor: bgColor,
          borderBottom: `2px solid ${borderColor}`,
          WebkitBoxDecorationBreak: 'clone',
          boxDecorationBreak: 'clone',
        }}
        title={`${sent.indicates}: ${(Math.abs(sent.net_weight) * 100).toFixed(1)}% strength (${sent.important_words}/${sent.total_words} words)`}
      >
        {sentText}
      </span>
    );

    lastIndex = sent.end;
  });

  if (lastIndex < originalText.length) {
    elements.push(
      <span key="text-end" className="text-gray-800">
        {originalText.substring(lastIndex)}
      </span>
    );
  }

  return elements;
}

export default ShapExplanation;
