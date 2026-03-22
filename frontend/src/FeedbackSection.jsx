import { useState } from 'react';

const API_BASE = import.meta.env.VITE_API_URL ?? 'https://api.sinxdetect.movindu.com';

function sha256(text) {
  const encoder = new TextEncoder();
  return crypto.subtle.digest('SHA-256', encoder.encode(text)).then((buf) =>
    Array.from(new Uint8Array(buf))
      .map((b) => b.toString(16).padStart(2, '0'))
      .join('')
  );
}

function FeedbackSection({ result, text }) {
  const [feedbackOpen, setFeedbackOpen] = useState(false);
  const [feedbackLabel, setFeedbackLabel] = useState('');
  const [feedbackComment, setFeedbackComment] = useState('');
  const [feedbackName, setFeedbackName] = useState('');
  const [feedbackEmail, setFeedbackEmail] = useState('');
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [feedbackDone, setFeedbackDone] = useState(false);
  const [feedbackError, setFeedbackError] = useState('');

  const handleFeedbackSubmit = async () => {
    if (!result) return;
    if (!feedbackLabel) {
      setFeedbackError('Please select what you think the correct label is.');
      return;
    }
    setFeedbackLoading(true);
    setFeedbackError('');
    try {
      const predictedLabel = result.label === 'AI' || result.label === 'AI-generated' ? 'AI' : 'HUMAN';
      const textHash = await sha256(text);

      const response = await fetch(`${API_BASE}/feedback`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          predicted_label: predictedLabel,
          corrected_label: feedbackLabel,
          text_hash: textHash,
          comment: feedbackComment || null,
          raw_text: text,
          user_name: feedbackName || null,
          user_email: feedbackEmail || null,
        }),
      });
      if (!response.ok) {
        const detail = await response.json().catch(() => ({}));
        throw new Error(detail.detail || 'Failed to submit feedback');
      }
      setFeedbackDone(true);
      setFeedbackOpen(false);
      setFeedbackComment('');
      setFeedbackLabel('');
      setFeedbackName('');
      setFeedbackEmail('');
    } catch (err) {
      setFeedbackError(err.message || 'Something went wrong');
    } finally {
      setFeedbackLoading(false);
    }
  };

  return (
    <div className="pt-4 border-t border-gray-200">
      {feedbackDone ? (
        <div className="flex items-center gap-2 text-sm text-cyan-700 bg-cyan-50 border border-cyan-200 rounded-lg px-4 py-3">
          <svg className="w-4 h-4 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
          </svg>
          Thank you! Your feedback has been submitted for review.
        </div>
      ) : !feedbackOpen ? (
        <button
          type="button"
          onClick={() => setFeedbackOpen(true)}
          className="inline-flex items-center gap-1.5 text-sm text-gray-500 hover:text-cyan-600 transition"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
          </svg>
          Share feedback
        </button>
      ) : (
        <div className="space-y-3 rounded-lg border border-gray-200 bg-gray-50 p-4">
          <div className="flex items-start justify-between">
            <p className="text-sm font-medium text-gray-800">
              Share your feedback
            </p>
            <button
              type="button"
              onClick={() => { setFeedbackOpen(false); setFeedbackError(''); }}
              className="text-gray-400 hover:text-gray-600"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
          <div>
            <label className="block text-xs font-medium text-gray-600 mb-1.5">
              What do you think the correct label is?
            </label>
            <div className="flex gap-2">
              <button
                type="button"
                onClick={() => setFeedbackLabel('HUMAN')}
                className={`flex-1 rounded-lg border px-3 py-2 text-sm font-medium transition ${
                  feedbackLabel === 'HUMAN'
                    ? 'border-emerald-500 bg-emerald-50 text-emerald-700 ring-2 ring-emerald-500/20'
                    : 'border-gray-300 bg-white text-gray-600 hover:border-gray-400'
                }`}
              >
                HUMAN
              </button>
              <button
                type="button"
                onClick={() => setFeedbackLabel('AI')}
                className={`flex-1 rounded-lg border px-3 py-2 text-sm font-medium transition ${
                  feedbackLabel === 'AI'
                    ? 'border-red-500 bg-red-50 text-red-700 ring-2 ring-red-500/20'
                    : 'border-gray-300 bg-white text-gray-600 hover:border-gray-400'
                }`}
              >
                AI
              </button>
            </div>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <input
              type="text"
              value={feedbackName}
              onChange={(e) => setFeedbackName(e.target.value)}
              className="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800 placeholder:text-gray-400 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/20"
              placeholder="Your name (optional)"
            />
            <input
              type="email"
              value={feedbackEmail}
              onChange={(e) => setFeedbackEmail(e.target.value)}
              className="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800 placeholder:text-gray-400 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/20"
              placeholder="Your email (optional)"
            />
          </div>
          <textarea
            value={feedbackComment}
            onChange={(e) => setFeedbackComment(e.target.value)}
            rows={2}
            className="w-full rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm text-gray-800 placeholder:text-gray-400 focus:border-cyan-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/20"
            placeholder="Any additional comments (optional)"
          />
          {feedbackError && (
            <p className="text-xs text-red-600">{feedbackError}</p>
          )}
          <button
            type="button"
            onClick={handleFeedbackSubmit}
            disabled={feedbackLoading}
            className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-600 px-4 py-2 text-sm font-medium text-white shadow transition hover:bg-cyan-700 disabled:cursor-not-allowed disabled:opacity-70"
          >
            {feedbackLoading && (
              <span className="inline-flex h-3.5 w-3.5 animate-spin rounded-full border-2 border-white/30 border-t-white" />
            )}
            {feedbackLoading ? 'Submitting...' : 'Submit feedback'}
          </button>
        </div>
      )}
    </div>
  );
}

export default FeedbackSection;
