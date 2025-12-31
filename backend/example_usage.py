# -*- coding: utf-8 -*-
"""
Example usage of the Sinhala Text Classifier
"""

from classify_text import SinhalaTextClassifier

# Initialize the classifier
classifier = SinhalaTextClassifier(model_path='models/bilstm_sinhala_model')

# Example 1: Classify a single Sinhala text
print("Example 1: Single text classification")
print("-" * 50)

sinhala_text = "තැපැල් සේවාවේ අවම තැපැල් ගාස්තුව රුපියල් 5 කින් වැඩි කිරීමට ගත් තීන්දුව ඇතුළත් ගැසට් නිවේදනය පාර්ලිමේන්තුවට අද (20) ඉදිරිපත් කළේය. තැපැල්, තැපැල් සේවා හා මුස්ලිම් ආගමික කටයුතු ඇමැති එම්.එච්.ඒ හලිම් මහතා නිකුත් කළ මෙම ගැසට් නිවේදනය අනුව ජුනි 15 දා සිට තැපැල් මුද්දරයක ගාස්තුව වැඩි වී ඇත. රුපියල් 10 ක් අවම තැපැල් ගාස්තුව රුපියල් 15 දක්වා වැඩි වන අතර ග්‍රෑම් 100 දක්වා වැඩිවන ග්‍රෑම් 20 ක් හෝ ඉන් කොටසක් සඳහා රුපියල් 10 බැගින් වැඩි වන අතර ග්‍රෑම් 100 සිට ග්‍රෑම් 250 දක්වා වැඩිවන සෑම ග්‍රෑම් 50 ක් හෝ ඉන් කොටසක් සඳහා රුපියල් 5 බැගින් වැඩිවේ. ලියාපදිංචි තැපැල් ගාස්තුවද රුපියල් 25 සිට රුපියල් 30 දක්වා වැඩි කර ඇති අතර තැපැල් පාර්සල් ගාස්තුවද බර අනුව ඉහළ දමා ඇත. ව්‍යාපාරික ලිපි සඳහා පැවැති ගාස්තුවල සංශෝධනයක් සිදුවී නැත. තැපැල් පතක මිල ද රුපියල් 8 සිට රුපියල් 10 දක්වා ඉහළ දැමීමට කටයුතු කර ඇත. ටෙලිමේල් සඳහා ස්ථාවර ගාස්තු වෙනස් නොවන අතර පළමු වචන 10 සඳහා පැවැති රුපියල් 20 ක ගාස්තුව රුපියල් 30 දක්වාද වැඩිවන සෑම වචනයක් සඳහාම රුපියල් 1.50 ක් වූ ගාස්තුව රුපියල් 2 දක්වාද වැඩිවේ. පණිවුඩය බෙදීමේ රුපියල් 15 ක් වූ සහතික ගාස්තුව රුපියල් 20 දක්වාද වැඩිවේ. පිරිවැය ඉහළයාම තුළ මෙලෙස ගාස්තු වැඩි කළ බවද ඉහත ගැසට් නිවේදනයේ සඳහන් කර ඇත."

result = classifier.classify(sinhala_text, return_probabilities=True)

print(f"Text: {sinhala_text}")
print(f"Prediction: {result['label']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: HUMAN={result['probabilities']['HUMAN']:.2%}, AI={result['probabilities']['AI']:.2%}")
print()
