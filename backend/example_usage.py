# -*- coding: utf-8 -*-
"""
Example usage of the Sinhala Text Classifier
"""

from classify_text import SinhalaTextClassifier

# Initialize the classifier
classifier = SinhalaTextClassifier(model_path='ml/models/bert_multilingual_model')

# Example 1: Classify a single Sinhala text
print("Example 1: Single text classification")
print("-" * 50)

sinhala_text = "ශ්‍රී ලංකාව ඉන්දියන් සාගරයේ පිහිටි සුන්දර දූපත් රාජ්‍යයක් වන අතර දිගු ඉතිහාසයක් සහ විවිධ සංස්කෘතික උරුමයක් ඇති රටකි. බුද්ධාගම, හින්දු, ඉස්ලාම් සහ ක්‍රිස්තියානි ආගමන්ගේ බලපෑමෙන් මෙහි සමාජය ගොඩනැගී ඇත. පැරණි නගර, විහාරස්ථාන, කොලනියල් ගොඩනැගිලි සහ සම්ප්‍රදායික උත්සව ශ්‍රී ලංකාවේ විශේෂත්වය වේ. කන්දුවැටි, වනාන්තර, වෙරළ සහ වනජීවී උද්‍යානයන් රටට අලංකාරයක් එක් කරයි. තේ, කුළුබඩු සහ මැණික් කර්මාන්තය ආර්ථිකයට වැදගත් වේ. අභියෝග තිබුණද, ශ්‍රී ලාංකික ජනතාවගේ ආත්මීය ශක්තිය සහ ආගන්තුක සත්කාරය රටේ අනාගතයට බලාපොරොත්තුවක් ගෙනෙයි."

result = classifier.classify(sinhala_text, return_probabilities=True)

print(f"Text: {sinhala_text}")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: HUMAN={result['probabilities']['HUMAN']:.2%}, AI={result['probabilities']['AI']:.2%}")
print()
