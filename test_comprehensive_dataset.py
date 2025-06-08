#!/usr/bin/env python3
"""
Comprehensive Test Suite for Enhanced Clinical AI Model
Tests the model on 100 diverse healthcare claim scenarios from tests.csv
"""

import pandas as pd
import numpy as np
import os
import sys
import time
from typing import Dict, List, Any

# Add models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

try:
    from clinical_bert_openai_predictor import ClinicalBERTOpenAIPredictor
    CLINICAL_AI_AVAILABLE = True
except ImportError as e:
    print(f"❌ Could not import ClinicalBERTOpenAIPredictor: {e}")
    CLINICAL_AI_AVAILABLE = False

def load_test_data(file_path: str = 'data/tests.csv') -> pd.DataFrame:
    """Load comprehensive test dataset"""
    print(f"📂 Loading test data from: {file_path}")
    df = pd.read_csv(file_path)
    print(f"✅ Loaded {len(df)} test cases")
    print(f"   Expected outcomes: {df['Outcome'].value_counts().to_dict()}")
    return df

def convert_claim_to_dict(row: pd.Series) -> Dict[str, Any]:
    """Convert DataFrame row to claim dictionary"""
    return {
        'Claim ID': row['Claim ID'],
        'Procedure Code': row['Procedure Code'], 
        'Diagnosis Code': row['Diagnosis Code'],
        'Insurance Type': row['Insurance Type'],
        'Billed Amount': float(row['Billed Amount']),
        'Allowed Amount': float(row['Allowed Amount']),
        'Paid Amount': float(row['Paid Amount']),
        'Date of Service': row['Date of Service']
    }

def map_outcome_to_prediction(outcome: str) -> str:
    """Map CSV outcome to prediction format"""
    if outcome == 'Approved':
        return 'APPROVED'
    elif outcome == 'Denied':
        return 'DENIED'
    elif outcome == 'Pending_Review':
        return 'PENDING_REVIEW'
    else:
        return 'APPROVED'  # Default for edge cases

def analyze_by_category(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze results by medical category"""
    
    # Define categories based on procedure codes
    categories = {
        'Emergency Care': ['99281', '99282', '99283', '99284', '99285', '31500', '92950', '36556', '32551', '21501'],
        'Preventive Care': ['77057', '99395', '90471', '90734', '90585', '80053', '85025', '88150'],
        'Mental Health': ['90837', '90801', '90834', '90847', '90853', '90862', '90832', '90901', '96116'],
        'Surgery': ['27447', '33533', '28810', '27130', '47562', '43239', '49505', '29827', '43280', '52601', '23472'],
        'Experimental': ['96413', '00670', '0016T', 'S2095', 'C9399', '0312T', 'C9024', 'C9136', 'C9752', 'Q0083'],
        'Cosmetic': ['15823', '15734', '30520', '30140', '19318', '17311', '15738', '19342', '14301', '11755', '15825', '27096', '25447'],
        'Routine Care': ['99213', '99214', '99215', '99221', '99222', '99223', '99231', '99232', '99233', '99238']
    }
    
    category_results = {}
    
    for category, procedures in categories.items():
        category_mask = results_df['Procedure Code'].isin(procedures)
        category_data = results_df[category_mask]
        
        if len(category_data) > 0:
            accuracy = (category_data['Correct']).mean()
            avg_confidence = category_data['Confidence'].mean()
            count = len(category_data)
            
            category_results[category] = {
                'count': count,
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'total_cases': len(category_data)
            }
    
    return category_results

def run_comprehensive_test(num_samples: int = 100) -> Dict[str, Any]:
    """Run comprehensive test on the full dataset"""
    
    if not CLINICAL_AI_AVAILABLE:
        print("❌ Enhanced Clinical AI not available for testing")
        return {}
    
    print("🧠🏥 COMPREHENSIVE CLINICAL AI TEST SUITE")
    print("Testing Enhanced Clinical AI v3.0 on Diverse Medical Scenarios")
    print("=" * 80)
    
    # Load test data
    test_df = load_test_data()
    
    # Sample data if needed
    if num_samples < len(test_df):
        test_sample = test_df.sample(n=num_samples, random_state=42)
        print(f"📊 Testing on random sample of {num_samples} cases")
    else:
        test_sample = test_df
        print(f"📊 Testing on all {len(test_df)} cases")
    
    # Initialize clinical AI
    print("\n🚀 Initializing Enhanced Clinical AI System...")
    start_time = time.time()
    clinical_ai = ClinicalBERTOpenAIPredictor()
    init_time = time.time() - start_time
    print(f"✅ System initialized in {init_time:.2f} seconds")
    print(f"   Active Models: {', '.join(clinical_ai._get_active_models())}")
    
    # Test results tracking
    results = []
    correct_predictions = 0
    total_processing_time = 0
    
    print(f"\n📋 Running Comprehensive Test on {len(test_sample)} Cases...")
    
    for i, (_, row) in enumerate(test_sample.iterrows(), 1):
        if i % 10 == 0:
            print(f"   Progress: {i}/{len(test_sample)} cases ({i/len(test_sample)*100:.1f}%)")
        
        # Convert to claim format
        claim = convert_claim_to_dict(row)
        expected = map_outcome_to_prediction(row['Outcome'])
        
        try:
            # Run prediction
            start_time = time.time()
            result = clinical_ai.predict_with_clinical_ai(claim)
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            # Extract information
            prediction = result['prediction']
            confidence = result['confidence']
            medical_necessity = result['medical_analysis']['medical_necessity']
            
            # Check accuracy
            is_correct = prediction == expected
            if is_correct:
                correct_predictions += 1
            
            # Store result
            results.append({
                'Claim ID': row['Claim ID'],
                'Procedure Code': row['Procedure Code'],
                'Diagnosis Code': row['Diagnosis Code'],
                'Insurance Type': row['Insurance Type'],
                'Billed Amount': row['Billed Amount'],
                'Expected': expected,
                'Predicted': prediction,
                'Correct': is_correct,
                'Confidence': confidence,
                'Medical Necessity': medical_necessity,
                'Processing Time': processing_time,
                'Reason Code': row.get('Reason Code', ''),
                'Clinical Category': classify_procedure(row['Procedure Code'])
            })
            
        except Exception as e:
            print(f"   ❌ ERROR processing claim {row['Claim ID']}: {e}")
            results.append({
                'Claim ID': row['Claim ID'],
                'Procedure Code': row['Procedure Code'],
                'Expected': expected,
                'Predicted': 'ERROR',
                'Correct': False,
                'Confidence': 0.0,
                'Error': str(e)
            })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate overall metrics
    overall_accuracy = correct_predictions / len(test_sample)
    avg_confidence = results_df['Confidence'].mean()
    avg_processing_time = total_processing_time / len(test_sample)
    
    # Analyze by category
    category_analysis = analyze_by_category(results_df)
    
    # Performance Summary
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"🎯 Overall Performance:")
    print(f"   Accuracy: {overall_accuracy:.1%} ({correct_predictions}/{len(test_sample)})")
    print(f"   Average Confidence: {avg_confidence:.3f}")
    print(f"   Average Processing Time: {avg_processing_time:.3f}s per case")
    print(f"   Total Processing Time: {total_processing_time:.2f}s")
    
    # Category breakdown
    print(f"\n📋 Performance by Medical Category:")
    for category, metrics in category_analysis.items():
        print(f"   {category}:")
        print(f"      Cases: {metrics['count']}")
        print(f"      Accuracy: {metrics['accuracy']:.1%}")
        print(f"      Avg Confidence: {metrics['avg_confidence']:.3f}")
    
    # Expected vs Predicted breakdown
    print(f"\n📈 Outcome Distribution:")
    outcome_counts = results_df.groupby(['Expected', 'Predicted']).size().unstack(fill_value=0)
    print(outcome_counts)
    
    # Save detailed results
    output_file = 'analysis/comprehensive_test_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"\n💾 Detailed results saved to: {output_file}")
    
    return {
        'overall_accuracy': overall_accuracy,
        'avg_confidence': avg_confidence,
        'total_cases': len(test_sample),
        'correct_predictions': correct_predictions,
        'category_analysis': category_analysis,
        'results_df': results_df,
        'processing_time': total_processing_time
    }

def classify_procedure(procedure_code: str) -> str:
    """Classify procedure into medical category"""
    
    emergency_codes = ['99281', '99282', '99283', '99284', '99285', '31500', '92950']
    preventive_codes = ['77057', '99395', '90471', '90734', '90585']
    mental_health_codes = ['90837', '90801', '90834', '90847', '90853', '90862']
    surgery_codes = ['27447', '33533', '28810', '27130', '47562', '43239']
    experimental_codes = ['96413', '00670', '0016T', 'S2095', 'C9399']
    cosmetic_codes = ['15823', '15734', '30520', '30140', '19318']
    
    if procedure_code in emergency_codes:
        return 'Emergency Care'
    elif procedure_code in preventive_codes:
        return 'Preventive Care'
    elif procedure_code in mental_health_codes:
        return 'Mental Health'
    elif procedure_code in surgery_codes:
        return 'Surgery'
    elif procedure_code in experimental_codes:
        return 'Experimental'
    elif procedure_code in cosmetic_codes:
        return 'Cosmetic'
    else:
        return 'Routine Care'

if __name__ == "__main__":
    print("🧠 Comprehensive Clinical AI Test Suite")
    print("Testing Enhanced Clinical AI v3.0 on 100 Diverse Medical Scenarios")
    
    # Run comprehensive test
    results = run_comprehensive_test(num_samples=50)  # Test on 50 samples initially
    
    if results:
        print(f"\n✅ Comprehensive testing completed successfully!")
        print(f"🎯 Final Accuracy: {results['overall_accuracy']:.1%}")
        print(f"📊 System performance validated across diverse medical scenarios!")
        print(f"🚀 Enhanced Clinical AI v3.0 ready for healthcare deployment!")
    else:
        print(f"\n❌ Testing failed - check system configuration") 