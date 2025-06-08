#!/usr/bin/env python3
"""
Quick Test - 10 Cases Only
"""

import pandas as pd
import os
import sys
import time

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from clinical_bert_openai_predictor import ClinicalBERTOpenAIPredictor

def main():
    print("🧠 Quick Test - Enhanced Clinical AI")
    print("Testing 10 diverse cases from tests.csv")
    
    # Load test data
    df = pd.read_csv('data/tests.csv')
    test_sample = df.head(10)  # Just first 10 cases
    
    print(f"✅ Loaded {len(test_sample)} test cases")
    print(f"   Expected outcomes: {test_sample['Outcome'].value_counts().to_dict()}")
    
    # Initialize AI
    clinical_ai = ClinicalBERTOpenAIPredictor()
    
    results = []
    correct = 0
    
    for i, (_, row) in enumerate(test_sample.iterrows(), 1):
        print(f"\n🔍 Test {i}: {row['Claim ID']} - {row['Reason Code']}")
        
        claim = {
            'Procedure Code': row['Procedure Code'],
            'Diagnosis Code': row['Diagnosis Code'],
            'Insurance Type': row['Insurance Type'],
            'Billed Amount': float(row['Billed Amount']),
            'Allowed Amount': float(row['Allowed Amount']),
            'Paid Amount': float(row['Paid Amount'])
        }
        
        expected = 'APPROVED' if row['Outcome'] == 'Approved' else 'DENIED' if row['Outcome'] == 'Denied' else 'PENDING_REVIEW'
        
        try:
            result = clinical_ai.predict_with_clinical_ai(claim)
            prediction = result['prediction']
            confidence = result['confidence']
            
            is_correct = prediction == expected
            if is_correct:
                correct += 1
            
            print(f"   Expected: {expected}")
            print(f"   Predicted: {prediction} ({confidence:.3f})")
            print(f"   {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
            
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
    
    accuracy = correct / len(test_sample)
    print(f"\n📊 Quick Test Results:")
    print(f"   Accuracy: {accuracy:.1%} ({correct}/{len(test_sample)})")

if __name__ == "__main__":
    main() 