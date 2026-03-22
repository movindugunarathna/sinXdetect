import { useState } from 'react';

function LimeExplanation({ explanation, text }) {
  const [showWordDetail, setShowWordDetail] = useState(false);

  const summary = explanation.evidence_summary;
  const sentences = explanation.sentence_explanations || [];
  const phrases = explanation.highlighted_text || [];
  // For the analysis list, filter out neutrals and sort by strength
  const significantSentences = sentences
    .filter((s) => s.color !== 'neutral')
    .sort((a, b) => Math.abs(b.net_weight) - Math.abs(a.net_weight));
  const hasContent = sentences.length > 0 || phrases.length > 0;

  return (
    <section className="rounded-xl border border-gray-200 bg-white p-6 space-y-5 shadow-sm">
      {/* Header */}
      <div className="flex items-center gap-2 border-b border-gray-200 pb-3">
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
            d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"
          />
        </svg>
        <h2 className="text-lg font-semibold text-gray-800">
          LIME Explanation
        </h2>
        {explanation.error && (
          <span className="text-xs text-gray-600 bg-gray-100 px-2 py-1 rounded-md border border-gray-200">
            {explanation.error}
          </span>
        )}
      </div>

      {/* Evidence Summary Bar */}
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
      )}

      {/* Highlighted Full Text — sentence-level highlighting like GPTZero */}
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

      {/* Sentence-Level Breakdown (only significant sentences) */}
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

      {/* Word-Level Detail (collapsible) */}
      {phrases.length > 0 && (
        <div className="border-t border-gray-200 pt-3">
          <button
            type="button"
            onClick={() => setShowWordDetail(!showWordDetail)}
            className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-cyan-600 transition"
          >
            <svg
              className={`w-3.5 h-3.5 transition-transform ${showWordDetail ? 'rotate-90' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            {showWordDetail ? 'Hide' : 'Show'} word-level detail ({phrases.length} phrases)
          </button>
          {showWordDetail && (
            <div className="space-y-2 mt-3">
              {phrases.slice(0, 15).map((item, idx) => (
                <div
                  key={idx}
                  className={`rounded-lg border p-3 ${
                    item.color === 'red'
                      ? 'bg-red-50 border-red-200'
                      : 'bg-emerald-50 border-emerald-200'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-1">
                        <span
                          className={`text-sm font-medium ${
                            item.color === 'red'
                              ? 'text-red-800'
                              : 'text-emerald-800'
                          }`}
                        >
                          {item.phrase}
                        </span>
                        <span
                          className={`text-xs px-2 py-0.5 rounded-full ${
                            item.color === 'red'
                              ? 'bg-red-100 text-red-700'
                              : 'bg-emerald-100 text-emerald-700'
                          }`}
                        >
                          {item.word_count} word{item.word_count > 1 ? 's' : ''}
                        </span>
                      </div>
                      <p className="text-xs text-gray-600">
                        Indicates:{' '}
                        <span className="font-medium">{item.indicates}</span>
                      </p>
                    </div>
                    <div className="text-right">
                      <div
                        className={`text-lg font-bold ${
                          item.color === 'red' ? 'text-red-600' : 'text-emerald-600'
                        }`}
                      >
                        {(Math.abs(item.weight) * 100).toFixed(1)}%
                      </div>
                      <div className="text-xs text-gray-500">importance</div>
                    </div>
                  </div>
                </div>
              ))}
              {phrases.length > 15 && (
                <p className="text-xs text-gray-500 text-center pt-1">
                  Showing top 15 of {phrases.length} phrases
                </p>
              )}
            </div>
          )}
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
 * Render the full text with sentence-level background highlighting (GPTZero style).
 * Each sentence is a continuous highlighted span. Gaps between sentences stay neutral.
 * Highlight intensity scales with signal strength.
 */
function renderSentenceHighlightedText(originalText, sentenceHighlights) {
  if (!sentenceHighlights || sentenceHighlights.length === 0) {
    return <span className="text-gray-800">{originalText}</span>;
  }

  // Sort sentences by their position in the text
  const sorted = [...sentenceHighlights].sort((a, b) => a.start - b.start);

  // Remove overlaps (keep first occurrence)
  const cleaned = [];
  for (const s of sorted) {
    if (cleaned.length === 0 || s.start >= cleaned[cleaned.length - 1].end) {
      cleaned.push(s);
    }
  }

  // Compute max absolute weight for intensity scaling
  const maxWeight = Math.max(...cleaned.map((s) => Math.abs(s.net_weight)), 0.01);

  const elements = [];
  let lastIndex = 0;

  cleaned.forEach((sent, idx) => {
    // Unhighlighted gap before this sentence
    if (sent.start > lastIndex) {
      elements.push(
        <span key={`gap-${idx}`} className="text-gray-800">
          {originalText.substring(lastIndex, sent.start)}
        </span>
      );
    }

    const sentText = originalText.substring(sent.start, sent.end);

    // Neutral sentences render as plain text
    if (sent.color === 'neutral') {
      elements.push(
        <span key={`sent-${idx}`} className="text-gray-800">
          {sentText}
        </span>
      );
      lastIndex = sent.end;
      return;
    }

    // Intensity: stronger signal = more opaque highlight (min 20%, max 55%)
    const ratio = Math.abs(sent.net_weight) / maxWeight;
    const opacity = 0.15 + ratio * 0.4;

    const bgColor =
      sent.color === 'red'
        ? `rgba(239, 68, 68, ${opacity.toFixed(2)})`   // red-500
        : `rgba(16, 185, 129, ${opacity.toFixed(2)})`;  // emerald-500

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

  // Trailing unhighlighted text
  if (lastIndex < originalText.length) {
    elements.push(
      <span key="text-end" className="text-gray-800">
        {originalText.substring(lastIndex)}
      </span>
    );
  }

  return elements;
}

export default LimeExplanation;
