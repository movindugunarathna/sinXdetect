# Frontend Refactor Summary - Light Mode with LIME Explanations

## Overview

The frontend has been completely refactored to display LIME explanation results in a modern light mode design.

## Major Changes

### 1. **Light Mode Design**

- Changed from dark theme to clean, modern light theme
- Updated color scheme:
  - Background: Soft gradient from cyan to purple tints (#f0f9ff → #faf5ff → #f9fafb)
  - Cards: White with subtle shadows and borders
  - Text: Dark gray (#1e293b) for readability
  - Accents: Cyan (#0891b2) and Purple (#9333ea)

### 2. **New LIME Explanation Features**

#### Added State Management

- `explanation`: Stores LIME explanation data
- `explainLoading`: Tracks explanation request status

#### New API Endpoint Integration

- Added `handleExplain()` function that calls `/explain` endpoint
- Processes and displays:
  - Word/phrase importance scores
  - Color-coded highlights (red for AI, green for Human)
  - Confidence and prediction data

#### Explanation Display Components

**A. Important Words & Phrases Section**

- Shows top 10 most influential phrases
- Each phrase card displays:
  - The phrase text in Sinhala
  - Word count badge
  - Importance percentage
  - Color indicator (red/green)
  - What it indicates (AI-generated or Human-written)

**B. Highlighted Text View**

- Renders original text with inline highlights
- Color-coded highlighting:
  - Red background with red border: Indicates AI-generated
  - Green background with green border: Indicates Human-written
- Hover tooltips show importance percentage
- Helper function `renderHighlightedText()` handles the rendering logic

**C. Legend**

- Clear visual legend explaining color meanings
- Red = AI-generated indicators
- Green = Human-written indicators

### 3. **UI Components Updated**

#### Buttons

- **Classify Button**: Cyan background (#0891b2)
- **Explain with LIME Button**: Purple background (#9333ea)
- Both have loading states with spinners
- Shadow effects for depth

#### Results Cards

- White background with subtle borders
- Color-coded probability bars:
  - Emerald for Human
  - Purple for AI
- Circular confidence indicator with animated progress ring

#### Form Elements

- Light borders and backgrounds
- Focus states with cyan accent color
- Improved placeholder text visibility

### 4. **Color Scheme**

#### Primary Colors

- **Cyan**: #0891b2 (Primary actions, Human indicators)
- **Purple**: #9333ea (Secondary actions, AI indicators)
- **Emerald**: #10b981 (Human probability)
- **Red**: Red-50 to Red-600 (AI-generated highlights)
- **Green**: Green-50 to Green-600 (Human-written highlights)

#### Neutral Colors

- **Background**: #f9fafb
- **Card Background**: White with 95% opacity
- **Text**: #1e293b (primary), #6b7280 (secondary)
- **Borders**: #e5e7eb

### 5. **Key Functions Added**

#### `renderHighlightedText(originalText, highlights)`

- Takes original text and highlight data
- Sorts highlights by position
- Renders text with inline colored highlights
- Preserves text that's not highlighted
- Adds tooltips with importance scores

#### `handleExplain()`

- Calls `/explain` API endpoint
- Handles loading states
- Updates both `explanation` and `result` states
- Error handling with user-friendly messages

### 6. **Responsive Design**

- Maintains responsive layout for mobile and desktop
- Flexible grid layouts for probability displays
- Stacked buttons on mobile, inline on desktop
- Maximum width of 5xl (1280px) for better readability

## Files Modified

1. **frontend/src/App.jsx**

   - Added explanation state and loading state
   - Added `handleExplain()` function
   - Added `renderHighlightedText()` helper function
   - Added LIME Explanation section
   - Updated all styling classes for light mode
   - Added "Explain with LIME" button

2. **frontend/src/App.css**

   - Updated `.glass-card` for light theme
   - Updated `.text-gradient` with new colors
   - Added standard `background-clip` property

3. **frontend/src/index.css**
   - Changed background to light gradient
   - Updated text colors for light theme
   - Removed dark radial gradients

## Features

### Classification View

- ✅ Quick classification with confidence score
- ✅ Probability breakdown (Human vs AI)
- ✅ Animated circular progress indicator
- ✅ Color-coded results

### Explanation View

- ✅ LIME-based word importance analysis
- ✅ Top 10 most important phrases display
- ✅ Color-coded phrase cards with importance scores
- ✅ Original text with inline highlights
- ✅ Hover tooltips on highlighted words
- ✅ Clear visual legend
- ✅ Error handling for edge cases

## User Experience Improvements

1. **Visual Clarity**: Light mode provides better readability
2. **Color Psychology**: Red (caution/AI) and Green (natural/Human) are intuitive
3. **Information Hierarchy**: Clear sections with proper spacing
4. **Loading States**: Both classify and explain have loading indicators
5. **Error Handling**: Graceful fallbacks if LIME fails
6. **Accessibility**: High contrast text, clear labels, semantic HTML

## Testing Recommendations

1. Test with various Sinhala text lengths
2. Verify highlights render correctly for RTL and LTR text
3. Test error handling with empty or invalid text
4. Verify responsive design on mobile devices
5. Check performance with long texts (>500 words)

## Future Enhancements

- Add export functionality for explanations
- Allow users to adjust num_samples and num_features
- Add comparison view for multiple texts
- Save explanation history
- Add dark mode toggle option
