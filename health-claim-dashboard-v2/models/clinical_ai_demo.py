#!/usr/bin/env python3
"""
Clinical AI Demo - ClinicalBERT + OpenAI Healthcare Prediction

Interactive demonstration of the advanced clinical AI system combining:
- ClinicalBERT (MIMIC-III trained medical knowledge)
- OpenAI GPT-4 (clinical reasoning and medical interpretation)
- Rule-based clinical guidelines (evidence-based protocols)

Shows detailed clinical analysis and reasoning for healthcare claim predictions.
"""

import pandas as pd
import json
from clinical_bert_openai_predictor import ClinicalBERTOpenAIPredictor

def demo_clinical_ai():
    """Demonstrate the Clinical AI system with sample claims"""
    
    print("🏥 CLINICAL AI DEMONSTRATION")
    print("ClinicalBERT + OpenAI Healthcare Claim Analysis")
    print("=" * 65)
    
    # Initialize the Clinical AI
    print("\n🚀 Initializing Advanced Clinical AI...")
    clinical_ai = ClinicalBERTOpenAIPredictor()
    
    # Load sample data
    try:
        test_df = pd.read_csv('../data/test_claims.csv')
        print(f"✅ Loaded {len(test_df)} test claims for demonstration")
    except FileNotFoundError:
        print("⚠️ Test data not found. Creating sample claims...")
        # Create sample claims for demo
        test_df = pd.DataFrame([
            {
                'Procedure Code': '99213',
                'Diagnosis Code': 'F32.1',
                'Insurance Type': 'Medicare',
                'Billed Amount': 2500.00,
                'Allowed Amount': 2000.00,
                'Paid Amount': 1800.00,
                'Outcome': 'Approved'
            },
            {
                'Procedure Code': '27447',
                'Diagnosis Code': 'M17.11',
                'Insurance Type': 'Private',
                'Billed Amount': 45000.00,
                'Allowed Amount': 35000.00,
                'Paid Amount': 30000.00,
                'Outcome': 'Denied'
            },
            {
                'Procedure Code': '99285',
                'Diagnosis Code': 'R06.02',
                'Insurance Type': 'Medicaid',
                'Billed Amount': 8500.00,
                'Allowed Amount': 7200.00,
                'Paid Amount': 6800.00,
                'Outcome': 'Approved'
            }
        ])
    
    # Train the Clinical AI
    print("\n🎓 Training Clinical AI on patterns...")
    if len(test_df) > 3:
        training_data = test_df.iloc[3:]  # Use most data for training
        demo_claims = test_df.iloc[:3]   # Use first 3 for demo
    else:
        training_data = test_df
        demo_claims = test_df
    
    clinical_ai.train_on_claims_data(training_data)
    
    # Demonstrate detailed clinical analysis
    print(f"\n🔍 DETAILED CLINICAL AI ANALYSIS")
    print("=" * 65)
    
    for i, (_, claim) in enumerate(demo_claims.iterrows(), 1):
        print(f"\n{'='*20} CLINICAL CASE {i} {'='*20}")
        
        # Get comprehensive prediction
        result = clinical_ai.predict_with_clinical_ai(claim.to_dict())
        
        # Display clinical information
        print(f"📋 PATIENT INFORMATION:")
        print(f"   Procedure Code: {claim['Procedure Code']} - {clinical_ai._get_procedure_description(str(claim['Procedure Code']))}")
        print(f"   Diagnosis Code: {claim['Diagnosis Code']} - {clinical_ai._get_diagnosis_description(str(claim['Diagnosis Code']))}")
        print(f"   Insurance Type: {claim['Insurance Type']}")
        print(f"   Billed Amount: ${claim['Billed Amount']:,.2f}")
        
        # Display AI prediction
        print(f"\n🤖 AI CLINICAL ASSESSMENT:")
        print(f"   Prediction: {result['prediction']}")
        print(f"   Overall Confidence: {result['confidence']:.3f}")
        print(f"   Medical Necessity: {result['medical_necessity']}")
        
        # Display component analysis
        print(f"\n🔬 COMPONENT ANALYSIS:")
        print(f"   ClinicalBERT Confidence: {result['clinical_bert_confidence']:.3f}")
        print(f"   OpenAI Confidence: {result['openai_confidence']:.3f}")
        print(f"   Rule-based Confidence: {result['rule_based_confidence']:.3f}")
        print(f"   Active AI Models: {', '.join(result['ai_models_used'])}")
        
        # Display clinical reasoning
        print(f"\n🧠 CLINICAL REASONING:")
        if result['clinical_reasoning']:
            for j, reason in enumerate(result['clinical_reasoning'], 1):
                print(f"   {j}. {reason}")
        else:
            print("   Standard clinical assessment applied")
        
        # Display risk factors
        if result['risk_factors']:
            print(f"\n⚠️ RISK FACTORS IDENTIFIED:")
            for j, risk in enumerate(result['risk_factors'], 1):
                print(f"   {j}. {risk}")
        
        # Display clinical notes
        if result['clinical_notes']:
            print(f"\n📝 CLINICAL NOTES:")
            print(f"   {result['clinical_notes']}")
        
        # Show actual outcome
        actual = 'DENIED' if claim['Outcome'] == 'Denied' else 'APPROVED'
        print(f"\n✅ ACTUAL OUTCOME: {actual}")
        
        # Accuracy indicator
        correct = "✅ CORRECT" if result['prediction'] == actual else "❌ INCORRECT"
        print(f"🎯 PREDICTION ACCURACY: {correct}")
        
        # Show clinical narrative preview
        print(f"\n📄 CLINICAL NARRATIVE PREVIEW:")
        clinical_text = clinical_ai.create_clinical_narrative(claim.to_dict())
        print(f"   {clinical_text[:200]}...")
        
        print(f"\n{'-'*65}")
    
    # Show overall performance summary
    print(f"\n📊 CLINICAL AI PERFORMANCE SUMMARY")
    print("=" * 65)
    
    # Evaluate on larger sample if available
    if len(test_df) > 10:
        evaluation_results = clinical_ai.evaluate_performance(test_df, num_samples=20)
        print(f"📈 Evaluation Results (20 test claims):")
        print(f"   Accuracy: {evaluation_results['accuracy']:.3f}")
        print(f"   Average Confidence: {evaluation_results['avg_confidence']:.3f}")
        print(f"   Total Predictions: {evaluation_results['total_predictions']}")
        print(f"   Correct Predictions: {evaluation_results['correct_predictions']}")
        print(f"   Medical Necessity Distribution: {evaluation_results['medical_necessity_distribution']}")
    
    # Show AI capabilities
    print(f"\n🏥 CLINICAL AI CAPABILITIES:")
    active_models = clinical_ai._get_active_models()
    print(f"   Active AI Models: {len(active_models)}")
    for model in active_models:
        print(f"   ✓ {model}")
    
    print(f"\n   Medical Knowledge Base:")
    print(f"   ✓ {len(clinical_ai.medical_patterns['high_risk_procedures'])} High-risk procedures")
    print(f"   ✓ {len(clinical_ai.medical_patterns['emergency_procedures'])} Emergency procedures")
    print(f"   ✓ {len(clinical_ai.medical_patterns['preventive_care'])} Preventive care codes")
    print(f"   ✓ {len(clinical_ai.medical_patterns['mental_health_procedures'])} Mental health procedures")
    print(f"   ✓ {len(clinical_ai.medical_patterns['mental_health_diagnoses'])} Mental health diagnoses")
    
    print(f"\n   Clinical Guidelines:")
    guidelines = clinical_ai.clinical_guidelines
    print(f"   ✓ Evidence-based approval criteria")
    print(f"   ✓ Cost threshold analysis")
    print(f"   ✓ Medical necessity factors")
    
    print(f"\n🌟 CLINICAL AI ADVANTAGES:")
    print(f"   ✓ Multi-modal AI ensemble (ClinicalBERT + OpenAI + Rules)")
    print(f"   ✓ MIMIC-III trained medical knowledge")
    print(f"   ✓ GPT-4 clinical reasoning capabilities")
    print(f"   ✓ Evidence-based clinical guidelines")
    print(f"   ✓ Comprehensive medical narrative generation")
    print(f"   ✓ Detailed clinical reasoning and risk assessment")
    print(f"   ✓ Fallback mode for high availability")
    
    print(f"\n🎉 Clinical AI Demonstration Complete!")
    print(f"🏥 Ready for advanced healthcare claim analysis!")

if __name__ == "__main__":
    demo_clinical_ai() 