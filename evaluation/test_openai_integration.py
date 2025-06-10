#!/usr/bin/env python3
"""
Test OpenAI Integration for Healthcare Claim Denial Prediction
"""

import os
from single_model_openai_denial_predictor import SingleModelOpenAIDenialPredictor
import pandas as pd

def test_openai_integration():
    """Test the OpenAI integration with sample claims"""
    
    print("🧪 Testing OpenAI Integration")
    print("=" * 50)
    
    # Check for API key
    api_key = input("Enter your OpenAI API key (or press Enter to skip): ").strip()
    if not api_key:
        print("⚠️ No API key provided. Testing traditional model only.")
        predictor = SingleModelOpenAIDenialPredictor()
    else:
        os.environ['OPENAI_API_KEY'] = api_key
        predictor = SingleModelOpenAIDenialPredictor(api_key)
        print("✅ OpenAI API key configured")
    
    # Load training data
    print("\n📂 Loading training data...")
    df = pd.read_csv('enhanced_claim_data.csv')
    print(f"Loaded {len(df)} claims")
    
    # Train traditional model
    predictor.train_traditional_model(df)
    
    # Test claims with different risk profiles
    test_claims = [
        {
            'name': 'High Risk Claim',
            'data': {
                'Procedure Code': '99285',  # Emergency visit
                'Diagnosis Code': 'F32.9',  # Mental health
                'Insurance Type': 'Self-Pay',
                'Billed Amount': 1500,
                'Allowed Amount': 800,
                'Paid Amount': 0,
                'Date of Service': '06/15/2024',
                'Reason Code': 'Authorization not obtained'
            }
        },
        {
            'name': 'Low Risk Claim',
            'data': {
                'Procedure Code': '99213',  # Office visit
                'Diagnosis Code': 'I10',    # Hypertension
                'Insurance Type': 'Medicare',
                'Billed Amount': 120,
                'Allowed Amount': 100,
                'Paid Amount': 95,
                'Date of Service': '07/20/2024',
                'Reason Code': 'Missing documentation'
            }
        },
        {
            'name': 'Medium Risk Claim',
            'data': {
                'Procedure Code': '99232',  # Hospital care
                'Diagnosis Code': 'J18.9',  # Pneumonia
                'Insurance Type': 'Commercial',
                'Billed Amount': 650,
                'Allowed Amount': 500,
                'Paid Amount': 475,
                'Date of Service': '08/10/2024',
                'Reason Code': 'Pre-existing condition'
            }
        }
    ]
    
    print("\n🎯 Testing Claims...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_claims, 1):
        claim_name = test_case['name']
        claim_data = test_case['data']
        
        print(f"\n--- {claim_name} (Test {i}) ---")
        print(f"Procedure: {claim_data['Procedure Code']}")
        print(f"Diagnosis: {claim_data['Diagnosis Code']}")
        print(f"Insurance: {claim_data['Insurance Type']}")
        print(f"Amount: ${claim_data['Billed Amount']:,}")
        
        # Get prediction
        result = predictor.predict_with_ensemble(claim_data)
        
        print(f"\n🎯 Final Prediction: {result['ensemble_prediction']}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"🔍 Risk Factors: {result['risk_factors']}")
        print(f"💭 Reasoning:")
        for reason in result['reasoning']:
            print(f"   • {reason}")
        
        # Show model breakdown
        if result['traditional_prediction']:
            trad_pred = result['traditional_prediction']['prediction']
            trad_conf = result['traditional_prediction']['confidence']
            print(f"🤖 Traditional ML: {trad_pred} (confidence: {trad_conf:.2f})")
        
        if result['openai_prediction']:
            openai_pred = result['openai_prediction']['prediction']
            openai_conf = result['openai_prediction']['confidence']
            print(f"🧠 OpenAI LLM: {openai_pred} (confidence: {openai_conf:.2f})")
        
        print("-" * 50)
    
    print(f"\n✅ Testing complete!")
    print(f"📈 Performance Metrics: {predictor.performance_metrics}")
    
    # Show data insights
    print(f"\n📊 Dataset Insights:")
    print(f"Total Claims: {len(df)}")
    print(f"Denial Rate: {(df['Outcome'] == 'Denied').mean():.1%}")
    print(f"Insurance Distribution:")
    print(df['Insurance Type'].value_counts())

if __name__ == "__main__":
    test_openai_integration() 