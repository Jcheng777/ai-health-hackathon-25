#!/usr/bin/env python3
"""
Analyze Real Data Performance Results
"""

import pandas as pd
import numpy as np

def analyze_performance():
    df = pd.read_csv('real_data_performance_results.csv')
    
    print("🔍 REAL DATA PERFORMANCE ANALYSIS")
    print("=" * 50)
    
    print(f"\n📊 OVERALL PERFORMANCE:")
    print(f"   Total Claims: {len(df)}")
    print(f"   Correct Predictions: {df['correct'].sum()}")
    print(f"   Accuracy: {df['correct'].mean():.1%}")
    print(f"   Average Confidence: {df['confidence'].mean():.3f}")
    
    print(f"\n🏥 BY MEDICAL NECESSITY:")
    necessity_perf = df.groupby('medical_necessity')['correct'].agg(['count', 'sum', 'mean'])
    for necessity, data in necessity_perf.iterrows():
        print(f"   {necessity}: {data['mean']:.1%} ({int(data['sum'])}/{int(data['count'])})")
    
    print(f"\n💼 BY INSURANCE TYPE:")
    insurance_perf = df.groupby('insurance_type')['correct'].agg(['count', 'sum', 'mean'])
    for insurance, data in insurance_perf.iterrows():
        print(f"   {insurance}: {data['mean']:.1%} ({int(data['sum'])}/{int(data['count'])})")
    
    print(f"\n❌ PREDICTION ERRORS:")
    errors = df[df['correct'] == False]
    print(f"   Total Errors: {len(errors)}")
    
    error_patterns = errors.groupby(['predicted_outcome', 'actual_outcome']).size()
    print("   Error Patterns:")
    for (pred, actual), count in error_patterns.items():
        print(f"     Predicted {pred}, Actually {actual}: {count} cases")
    
    print(f"\n🔝 TOP PROCEDURE CODES (by frequency):")
    proc_perf = df.groupby('procedure_code')['correct'].agg(['count', 'sum', 'mean']).sort_values('count', ascending=False)
    for proc, data in proc_perf.head(8).iterrows():
        print(f"   {proc}: {data['mean']:.1%} ({int(data['sum'])}/{int(data['count'])})")
    
    print(f"\n🎯 KEY INSIGHTS:")
    
    # Check if model is too conservative or too liberal
    predictions = df['predicted_outcome'].value_counts()
    actuals = df['actual_outcome'].value_counts()
    
    pred_denial_rate = predictions.get('DENIED', 0) / len(df)
    actual_denial_rate = actuals.get('DENIED', 0) / len(df)
    
    print(f"   Predicted Denial Rate: {pred_denial_rate:.1%}")
    print(f"   Actual Denial Rate: {actual_denial_rate:.1%}")
    
    if pred_denial_rate > actual_denial_rate + 0.1:
        print("   ⚠️  Model is TOO CONSERVATIVE (over-denying)")
    elif pred_denial_rate < actual_denial_rate - 0.1:
        print("   ⚠️  Model is TOO LIBERAL (under-denying)")
    else:
        print("   ✅ Model denial rate is reasonably balanced")
    
    # Check confidence patterns
    correct_confidence = df[df['correct'] == True]['confidence'].mean()
    incorrect_confidence = df[df['correct'] == False]['confidence'].mean()
    
    print(f"   Avg Confidence (Correct): {correct_confidence:.3f}")
    print(f"   Avg Confidence (Incorrect): {incorrect_confidence:.3f}")
    
    if correct_confidence > incorrect_confidence:
        print("   ✅ Higher confidence on correct predictions (good calibration)")
    else:
        print("   ⚠️  Confidence calibration needs improvement")

if __name__ == "__main__":
    analyze_performance() 