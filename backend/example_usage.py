# -*- coding: utf-8 -*-
"""
Example usage of the Sinhala Text Classifier
"""

from classify_text import SinhalaTextClassifier

# Initialize the classifier
classifier = SinhalaTextClassifier(model_path='ml/models/sinbert_sinhala_classifier')

# Example 1: Classify a single Sinhala text
print("Example 1: Single text classification")
print("-" * 50)

sinhala_text = "මෙම යුගයේ මුහුණ දෙන ලොකුම ගෝලීය ගැටලුවක් ලෙස දේශගුණික වෙනස්වීම හැඳින්විය හැක. කාර්මික ක්‍රියාකාරකම්, වනාන්තර විනාශය, ඉන්ධන දහනය සහ අධික කාබන් ඩයොක්සයිඩ් විමෝචනය නිසා පෘථිවියේ උෂ්ණත්වය ක්‍රමයෙන් ඉහළ යමින් පවතී. මෙය වර්ෂා රටා වෙනස් වීම, වියළි කාල වැඩි වීම, මුහුදු මට්ටම ඉහළ යාම සහ ස්වභාවික විපත් වැඩි වීම වැනි බරපතල ප්‍රතිවිපාක ඇති කරයි. දේශගුණික වෙනස්වීමට විසඳුම් සොයන විට පළමුවම පිරිසිදු හා නැවත නවීකරණය කළ හැකි බලශක්ති භාවිතය වැඩි කිරීම අත්‍යවශ්‍ය වේ. සූර්ය බලශක්ති, සුළං බලශක්ති සහ ජල විදුලිය වැනි බලශක්ති මූලාශ්‍ර භාවිතයෙන් කාබන් විමෝචනය අවම කළ හැක. එසේම පොදු ප්‍රවාහනය භාවිතය වැඩි කිරීම, විදුලි වාහන භාවිතයට උනන්දු වීමද වැදගත් වේ. වනාන්තර සංරක්ෂණය හා නව වෘක්ෂ රෝපණ වැඩසටහන් ක්‍රියාත්මක කිරීම දේශගුණික වෙනස්වීම පාලනයට විශාල දායකත්වයක් ලබා දෙයි. වෘක්ෂ කාබන් ඩයොක්සයිඩ් අවශෝෂණය කර වායුගෝලය පිරිසිදු කරයි. එමෙන්ම අපි දෛනික ජීවිතයේදී බලශක්ති සුරැකුම්කරණයට අවධානය යොමු කළ යුතුය. අවසන් වශයෙන්, රජය, සමාජය සහ පුද්ගලයන් එක්ව කටයුතු කළහොත් දේශගුණික වෙනස්වීමේ බලපෑම් අවම කර ගැනීම සාර්ථකව සිදු කළ හැක."

result = classifier.classify(sinhala_text, return_probabilities=True)

print(f"Text: {sinhala_text}")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: HUMAN={result['probabilities']['HUMAN']:.2%}, AI={result['probabilities']['AI']:.2%}")
print()
