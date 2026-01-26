#!/usr/bin/env python3
"""
Example usage of the LIME explanation endpoint
"""
import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

def test_classification(text: str):
    """Test basic classification endpoint"""
    print("\n" + "="*60)
    print("Testing /classify endpoint")
    print("="*60)
    
    response = requests.post(
        f"{BASE_URL}/classify",
        json={"text": text, "return_probabilities": True}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"Text: {text[:100]}...")
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        if 'probabilities' in result:
            print(f"Probabilities: HUMAN={result['probabilities']['HUMAN']:.2%}, AI={result['probabilities']['AI']:.2%}")
    else:
        print(f"Error: {response.status_code} - {response.text}")


def test_explanation(text: str, num_samples: int = 100, num_features: int = None):
    """Test LIME explanation endpoint"""
    print("\n" + "="*60)
    print("Testing /explain endpoint (LIME)")
    print("="*60)
    
    payload = {"text": text, "num_samples": num_samples}
    if num_features:
        payload["num_features"] = num_features
    
    response = requests.post(
        f"{BASE_URL}/explain",
        json=payload
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"\nText: {text[:100]}...")
        print(f"\nPrediction: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        
        if result.get('error'):
            print(f"\nWarning: {result['error']}")
        
        if result['highlighted_text']:
            print(f"\nTop Important Phrases ({len(result['highlighted_text'])} total):")
            print("-" * 60)
            for i, phrase in enumerate(result['highlighted_text'][:5], 1):
                indicator = "🔴" if phrase['color'] == 'red' else "🟢"
                print(f"{i}. {indicator} '{phrase['phrase']}'")
                print(f"   Weight: {phrase['weight']:.3f} | Indicates: {phrase['indicates']}")
                print()
        else:
            print("\nNo significant phrases identified.")
        
        # Print class probabilities
        probs = result['explanation_data']['predicted_probability']
        print(f"\nClass Probabilities:")
        print(f"  HUMAN: {probs[0]:.2%}")
        print(f"  AI:    {probs[1]:.2%}")
        
    else:
        print(f"Error: {response.status_code} - {response.text}")


def main():
    # Example Sinhala texts
    sinhala_text = """
    අද මම පාසලට ගියෙමි. එතැනදී මගේ යාළුවන් හමුවිය. 
    අපි එකට පන්තියට ගියෙමු. ගුරුවරයා ඉතා හොඳින් පාඩම උගන්වන්න විදියට.
    """
    
    # Test both endpoints
    test_classification(sinhala_text.strip())
    test_explanation(sinhala_text.strip(), num_samples=100)


if __name__ == "__main__":
    try:
        # Check if server is running
        response = requests.get(f"{BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            print("⚠️  Server is not healthy!")
            exit(1)
    except requests.exceptions.RequestException:
        print("❌ Server is not running!")
        print(f"Please start the server first: python app.py")
        exit(1)
    
    main()
