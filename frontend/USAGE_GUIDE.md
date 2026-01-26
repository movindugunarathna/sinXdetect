# Frontend Usage Guide

## Overview

The Sinhala Human vs AI Text Classifier now features a beautiful light mode interface with LIME-based explainability.

## How to Use

### 1. **Start the Application**

```bash
# Terminal 1 - Backend
cd backend
python app.py

# Terminal 2 - Frontend
cd frontend
npm run dev
```

### 2. **Open in Browser**

Navigate to: `http://localhost:5173/`

### 3. **Enter Text**

- Type or paste Sinhala text into the text area
- Or click "Fill sample text" to use a pre-filled example

### 4. **Choose Your Action**

#### Option A: Quick Classification

- Click **"Classify"** button (cyan)
- Get instant prediction: HUMAN or AI
- View confidence percentage
- See probability breakdown

#### Option B: Detailed Explanation

- Click **"Explain with LIME"** button (purple)
- Get classification PLUS detailed word-level analysis
- See which words indicate AI vs Human writing
- View highlighted text with importance scores

## Understanding the Results

### Classification Results

- **Circular Progress**: Shows confidence percentage
  - Green ring = Human prediction
  - Purple ring = AI prediction
- **Probability Cards**: Exact percentages for each class

### LIME Explanation Results

#### 1. Important Words & Phrases

- **Red cards**: Words that indicate AI-generated text
- **Green cards**: Words that indicate Human-written text
- **Importance %**: How much this phrase influenced the prediction
- **Word count**: Number of words in the phrase

#### 2. Highlighted Text

- Original text with color-coded highlights
- **Red highlight with red border**: Supports AI-generated classification
- **Green highlight with green border**: Supports Human-written classification
- **Hover over highlights**: See importance tooltip

#### 3. Color Legend

- 🟥 Red = AI-generated indicators
- 🟩 Green = Human-written indicators

## Tips

1. **Longer texts** provide more accurate explanations
2. **At least 2 words** required for LIME analysis
3. **Top 10 phrases** are shown (if more than 10 are found)
4. **Importance threshold**: Only shows phrases with >1% importance

## Features

✅ **Light Mode Design**: Easy on the eyes, professional look  
✅ **Real-time Classification**: Fast prediction results  
✅ **LIME Explanations**: Understand WHY the model made its prediction  
✅ **Word Highlighting**: Visual feedback on important words  
✅ **Responsive Design**: Works on desktop and mobile  
✅ **Error Handling**: Graceful fallbacks for edge cases  
✅ **Loading States**: Clear feedback during processing

## Troubleshooting

### No highlights showing?

- The text might be too short (need at least 2 words)
- All words might have importance below the 1% threshold
- Try longer or more distinctive text

### Explanation taking long?

- LIME analysis requires 100 samples by default
- Longer texts take more time to analyze
- This is normal for accurate explanations

### Backend connection error?

- Ensure backend is running on port 8000
- Check `VITE_API_URL` in frontend/.env if using different port

## API Endpoints Used

- `POST /classify`: Quick classification only
- `POST /explain`: Classification + LIME explanation

## Color Meanings

| Color             | Meaning          | Usage                              |
| ----------------- | ---------------- | ---------------------------------- |
| Cyan (#0891b2)    | Primary action   | Classify button, Human indicators  |
| Purple (#9333ea)  | Secondary action | Explain button, AI indicators      |
| Emerald (#10b981) | Human class      | Human probability, confidence ring |
| Red (various)     | AI indicators    | Phrases that suggest AI-generated  |
| Green (various)   | Human indicators | Phrases that suggest Human-written |

## Example Workflow

1. Enter Sinhala text: "මෙය කෘතිම බුද්ධි මගින් ලියන ලද වාක්‍යයකි"
2. Click "Explain with LIME"
3. Wait for analysis (few seconds)
4. Review results:
   - See prediction and confidence
   - Check which words influenced the decision
   - Understand the model's reasoning

## Best Practices

- ✅ Use complete sentences for better accuracy
- ✅ Include context in your text
- ✅ Try both "Classify" and "Explain" to compare
- ✅ Use "Explain" when you need to understand the decision
- ✅ Use "Classify" for quick checks

## Support

For issues or questions, check:

- Backend logs in the terminal
- Browser console for frontend errors
- API documentation at http://localhost:8000/docs
