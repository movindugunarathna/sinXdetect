# Light Mode Interface - Visual Description

## Overall Layout

### Header Section

```
┌─────────────────────────────────────────────────────────┐
│              SINHALA HUMAN VS AI                        │
│          Text Classifier                                │
│  Enter Sinhala text and get AI-powered classification  │
│           with word-level explanations                  │
└─────────────────────────────────────────────────────────┘
```

- Gradient text title (cyan to purple)
- Clean, centered layout
- Descriptive subtitle in gray

### Input Section

```
┌─────────────────────────────────────────────────────────┐
│ Text to classify                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │                                                       │ │
│ │  [Text input area - white with gray border]          │ │
│ │                                                       │ │
│ └─────────────────────────────────────────────────────┘ │
│                                                           │
│ [Fill sample text] ☑ Return probabilities               │
│                                                           │
│ [Classify 🔵]  [Explain with LIME 🟣]  Backend: ...     │
└─────────────────────────────────────────────────────────┘
```

- White card with subtle shadow
- Two action buttons (Cyan and Purple)
- Clear button states with loading spinners

## Results Display

### Classification Results Card

```
┌─────────────────────────────────────────────────────────┐
│                                                           │
│              Prediction    [ HUMAN ]                     │
│                                                           │
│                      ⚪○○○○○○                            │
│                    ○○   85.42%  ○○                      │
│                   ○○  CONFIDENCE ○○                     │
│                    ○○          ○○                        │
│                      ○○○○○○○○                            │
│                                                           │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │ 🟢 HUMAN 85.42% │  │ 🟣 AI    14.58% │            │
│  └──────────────────┘  └──────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

- White card with border
- Animated circular progress ring
- Color-coded probability cards (emerald/purple)

### LIME Explanation Card

```
┌─────────────────────────────────────────────────────────┐
│ 💡 LIME Explanation                                      │
├─────────────────────────────────────────────────────────┤
│                                                           │
│ Important Words & Phrases    🟥 AI-gen  🟩 Human-written│
│                                                           │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 🟥 යෝජනා කරන ලදී         [2 words]        45.3% │ │
│ │     Indicates: AI-generated                importance│ │
│ └─────────────────────────────────────────────────────┘ │
│                                                           │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ 🟩 ඔබට                  [1 word]           32.1% │ │
│ │     Indicates: Human-written              importance│ │
│ └─────────────────────────────────────────────────────┘ │
│                                                           │
│ ... [more phrase cards]                                  │
│                                                           │
├─────────────────────────────────────────────────────────┤
│ Highlighted Text                                         │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ මෙය [කෘතිම බුද්ධි] මගින් [යෝජනා කරන ලදී]        │ │
│ │     ↑ red highlight       ↑ red highlight            │ │
│ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## Color Scheme

### Background

- Page background: Light gradient (cyan → purple → gray tints)
- Card background: White with 95% opacity
- Soft shadows for depth

### Interactive Elements

- **Primary Button (Classify)**:

  - Background: #0891b2 (cyan-600)
  - Hover: #0e7490 (cyan-700)
  - Text: White
  - Shadow: Cyan glow

- **Secondary Button (Explain)**:
  - Background: #9333ea (purple-600)
  - Hover: #7e22ce (purple-700)
  - Text: White
  - Shadow: Purple glow

### Text Colors

- **Primary text**: #1e293b (slate-800)
- **Secondary text**: #6b7280 (gray-500)
- **Labels**: #374151 (gray-700)

### Highlights

- **AI-generated phrases**:

  - Background: #fee2e2 (red-50)
  - Border: #f87171 (red-400)
  - Text: #991b1b (red-800)

- **Human-written phrases**:
  - Background: #dcfce7 (green-50)
  - Border: #4ade80 (green-400)
  - Text: #166534 (green-800)

## Responsive Behavior

### Desktop (>640px)

- Maximum width: 1280px
- Buttons side-by-side
- Two-column probability display
- Centered layout with margins

### Mobile (<640px)

- Full width with padding
- Stacked buttons
- Single-column probability display
- Touch-friendly button sizes

## Accessibility Features

- ✅ High contrast text (WCAG AA compliant)
- ✅ Clear focus states on interactive elements
- ✅ Semantic HTML structure
- ✅ Loading indicators for async operations
- ✅ Error messages in accessible red
- ✅ Descriptive button labels
- ✅ Keyboard navigation support

## Animation & Transitions

1. **Circular progress ring**: 1-second ease-out animation
2. **Button hover**: Smooth color transition
3. **Loading spinner**: Continuous rotation
4. **Cards**: Subtle entrance (if implemented)

## Typography

- **Font family**: Inter (with fallbacks)
- **Heading sizes**:
  - H1: 2.25rem (3xl) - 3rem (4xl)
  - H2: 1.125rem (lg)
  - Body: 0.875rem (sm) - 1rem (base)
- **Font weights**:
  - Regular: 400
  - Medium: 500
  - Semibold: 600
  - Bold: 700

## Spacing & Layout

- **Card padding**: 1.5rem (6) - 2rem (8)
- **Section gaps**: 1.25rem (5)
- **Element gaps**: 0.75rem (3)
- **Border radius**:
  - Cards: 1rem (2xl)
  - Buttons: 0.75rem (xl)
  - Pills: 9999px (full)

## Visual Hierarchy

1. **Page title** (gradient, large, bold)
2. **Input area** (prominent white card)
3. **Action buttons** (colorful, clear CTAs)
4. **Results** (structured cards with visual emphasis)
5. **Details** (smaller text, supporting information)

This creates a clear, professional, and user-friendly interface that makes the LIME explanations easy to understand!
