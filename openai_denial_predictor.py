#!/usr/bin/env python3
"""
OpenAI-Enhanced Healthcare Claim Denial Predictor

Uses OpenAI GPT models combined with traditional ML for improved
denial prediction accuracy and reasoning.
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import openai
from openai import OpenAI
import time
import warnings
warnings.filterwarnings('ignore')

class OpenAIDenialPredictor:
    def __init__(self, openai_api_key=None):
        """Initialize the predictor with OpenAI integration"""
        # Set up OpenAI
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            # Try to get from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("⚠️ Warning: No OpenAI API key provided. Using fallback model only.")
                self.client = None
        
        # Traditional ML components
        self.rf_model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        
        # Performance tracking
        self.performance_metrics = {}
        
    def create_clinical_context(self, claim_data):
        """Create clinical context for OpenAI analysis"""
        context = f"""
        Healthcare Claim Analysis:
        
        Patient Information:
        - Procedure Code: {claim_data.get('Procedure Code', 'N/A')} 
        - Diagnosis Code: {claim_data.get('Diagnosis Code', 'N/A')}
        - Insurance Type: {claim_data.get('Insurance Type', 'N/A')}
        - Billed Amount: ${claim_data.get('Billed Amount', 0):,}
        - Allowed Amount: ${claim_data.get('Allowed Amount', 0):,}
        - Date of Service: {claim_data.get('Date of Service', 'N/A')}
        - Reason Code: {claim_data.get('Reason Code', 'N/A')}
        
        Clinical Context:
        - This is a healthcare insurance claim that needs to be evaluated for approval/denial
        - Consider medical necessity, insurance coverage policies, and billing accuracy
        - Look for red flags like unusual procedure-diagnosis combinations or excessive billing
        """
        return context
    
    def get_openai_prediction(self, claim_data, max_retries=3):
        """Get prediction and reasoning from OpenAI"""
        if not self.client:
            return None
            
        context = self.create_clinical_context(claim_data)
        
        prompt = f"""
        {context}
        
        As a healthcare insurance expert, analyze this claim and provide:
        
        1. Prediction: Should this claim be "APPROVED", "DENIED", or "NEEDS_REVIEW"?
        2. Confidence: Rate your confidence from 0.0 to 1.0
        3. Risk Factors: List specific risk factors (if any)
        4. Reasoning: Explain your decision in 2-3 bullet points
        
        Consider these factors:
        - Medical necessity of the procedure for the diagnosis
        - Appropriateness of billing amount for the procedure
        - Insurance type coverage patterns
        - Common denial reasons for this procedure/diagnosis combination
        
        Respond in this JSON format:
        {{
            "prediction": "APPROVED|DENIED|NEEDS_REVIEW",
            "confidence": 0.85,
            "risk_factors": ["factor1", "factor2"],
            "reasoning": ["reason1", "reason2", "reason3"]
        }}
        """
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4",  # Use GPT-4 for better medical reasoning
                    messages=[
                        {"role": "system", "content": "You are a healthcare insurance claim analyst with extensive medical and billing expertise."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,  # Lower temperature for more consistent medical decisions
                    max_tokens=500
                )
                
                # Parse the response
                content = response.choices[0].message.content.strip()
                
                # Try to extract JSON from response
                if '```json' in content:
                    json_str = content.split('```json')[1].split('```')[0]
                elif '{' in content:
                    json_start = content.find('{')
                    json_end = content.rfind('}') + 1
                    json_str = content[json_start:json_end]
                else:
                    json_str = content
                
                result = json.loads(json_str)
                
                # Validate and normalize the result
                prediction = result.get('prediction', 'NEEDS_REVIEW').upper()
                if prediction not in ['APPROVED', 'DENIED', 'NEEDS_REVIEW']:
                    prediction = 'NEEDS_REVIEW'
                
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'risk_factors': result.get('risk_factors', []),
                    'reasoning': result.get('reasoning', ['OpenAI analysis completed'])
                }
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                    
            except Exception as e:
                print(f"OpenAI API error (attempt {attempt + 1}): {e}")
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)  # Wait before retry
        
        return None
    
    def train_traditional_model(self, df):
        """Train the traditional Random Forest model"""
        print("🤖 Training traditional ML model...")
        
        # Preprocess data
        df_processed = self.preprocess_data(df.copy())
        
        # Create binary target (focusing on Denied vs others)
        df_processed['Binary_Target'] = (df_processed['Outcome'] == 'Denied').astype(int)
        
        # Prepare features
        categorical_features = ['Diagnosis Code', 'Procedure Code', 'Insurance Type']
        numerical_features = ['Billed Amount', 'Allowed Amount', 'Paid Amount', 'Days_Since_Min']
        
        # Encode categorical features
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
                df_processed[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(
                    df_processed[feature].astype(str)
                )
            else:
                df_processed[f'{feature}_encoded'] = self.label_encoders[feature].transform(
                    df_processed[feature].astype(str)
                )
        
        # Scale numerical features
        if self.scaler is None:
            self.scaler = StandardScaler()
            df_processed[numerical_features] = self.scaler.fit_transform(df_processed[numerical_features])
        else:
            df_processed[numerical_features] = self.scaler.transform(df_processed[numerical_features])
        
        # Prepare feature matrix
        encoded_features = [f'{feature}_encoded' for feature in categorical_features]
        self.feature_names = encoded_features + numerical_features
        
        X = df_processed[self.feature_names]
        y = df_processed['Binary_Target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train Random Forest
        self.rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Traditional model trained. Accuracy: {accuracy:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Approved', 'Denied']))
        
        self.performance_metrics['traditional_accuracy'] = accuracy
        
        return X_test, y_test, y_pred
    
    def preprocess_data(self, df):
        """Preprocess the dataframe"""
        # Convert date
        df['Date of Service'] = pd.to_datetime(df['Date of Service'], format='%m/%d/%Y', errors='coerce')
        min_date = df['Date of Service'].min()
        df['Days_Since_Min'] = (df['Date of Service'] - min_date).dt.days
        df['Days_Since_Min'] = df['Days_Since_Min'].fillna(0)
        
        # Fill missing values
        numeric_columns = ['Billed Amount', 'Allowed Amount', 'Paid Amount']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        return df
    
    def predict_with_ensemble(self, claim_data):
        """Make prediction using both OpenAI and traditional model"""
        results = {
            'traditional_prediction': None,
            'openai_prediction': None,
            'ensemble_prediction': None,
            'confidence': 0.5,
            'reasoning': [],
            'risk_factors': []
        }
        
        # Get traditional ML prediction
        if self.rf_model is not None:
            try:
                traditional_result = self.predict_traditional(claim_data)
                results['traditional_prediction'] = traditional_result
            except Exception as e:
                print(f"Traditional model error: {e}")
        
        # Get OpenAI prediction
        openai_result = self.get_openai_prediction(claim_data)
        if openai_result:
            results['openai_prediction'] = openai_result
        
        # Ensemble logic
        if results['traditional_prediction'] and results['openai_prediction']:
            # Both models available - use weighted ensemble
            trad_pred = results['traditional_prediction']['prediction']
            openai_pred = results['openai_prediction']['prediction']
            
            # Convert to common format
            trad_denial_prob = results['traditional_prediction']['denial_probability']
            openai_confidence = results['openai_prediction']['confidence']
            
            # Weight: 60% OpenAI (better reasoning), 40% traditional (more data-driven)
            if openai_pred == 'DENIED':
                openai_denial_score = openai_confidence
            elif openai_pred == 'APPROVED':
                openai_denial_score = 1 - openai_confidence
            else:  # NEEDS_REVIEW
                openai_denial_score = 0.5
            
            ensemble_denial_score = 0.6 * openai_denial_score + 0.4 * trad_denial_prob
            
            if ensemble_denial_score > 0.7:
                final_prediction = 'DENIED'
            elif ensemble_denial_score < 0.3:
                final_prediction = 'APPROVED'
            else:
                final_prediction = 'NEEDS_REVIEW'
            
            results['ensemble_prediction'] = final_prediction
            results['confidence'] = abs(ensemble_denial_score - 0.5) * 2  # Convert to confidence
            results['reasoning'] = results['openai_prediction']['reasoning']
            results['risk_factors'] = results['openai_prediction']['risk_factors']
            
        elif results['openai_prediction']:
            # Only OpenAI available
            openai_pred = results['openai_prediction']['prediction']
            results['ensemble_prediction'] = openai_pred
            results['confidence'] = results['openai_prediction']['confidence']
            results['reasoning'] = results['openai_prediction']['reasoning']
            results['risk_factors'] = results['openai_prediction']['risk_factors']
            
        elif results['traditional_prediction']:
            # Only traditional available
            trad_pred = results['traditional_prediction']['prediction']
            results['ensemble_prediction'] = trad_pred
            results['confidence'] = results['traditional_prediction']['confidence']
            results['reasoning'] = ['Traditional ML model prediction']
            results['risk_factors'] = results['traditional_prediction']['risk_factors']
            
        else:
            # Fallback
            results['ensemble_prediction'] = 'NEEDS_REVIEW'
            results['confidence'] = 0.5
            results['reasoning'] = ['Unable to generate prediction - manual review required']
            results['risk_factors'] = ['Prediction system unavailable']
        
        return results
    
    def predict_traditional(self, claim_data):
        """Get prediction from traditional model"""
        if self.rf_model is None:
            return None
        
        # Create dataframe
        df = pd.DataFrame([claim_data])
        df_processed = self.preprocess_data(df)
        
        # Encode features
        categorical_features = ['Diagnosis Code', 'Procedure Code', 'Insurance Type']
        for feature in categorical_features:
            if feature in self.label_encoders:
                try:
                    df_processed[f'{feature}_encoded'] = self.label_encoders[feature].transform(
                        df_processed[feature].astype(str)
                    )
                except ValueError:
                    # Unknown category
                    df_processed[f'{feature}_encoded'] = 0
            else:
                df_processed[f'{feature}_encoded'] = 0
        
        # Scale numerical features
        numerical_features = ['Billed Amount', 'Allowed Amount', 'Paid Amount', 'Days_Since_Min']
        df_processed[numerical_features] = self.scaler.transform(df_processed[numerical_features])
        
        # Predict
        X = df_processed[self.feature_names]
        denial_prob = self.rf_model.predict_proba(X)[0][1]  # Probability of denial
        prediction = 'DENIED' if denial_prob > 0.5 else 'APPROVED'
        
        # Get feature importance for risk factors
        feature_importance = dict(zip(self.feature_names, self.rf_model.feature_importances_))
        top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
        
        risk_factors = []
        for feature, importance in top_features:
            if importance > 0.1:
                if 'Amount' in feature:
                    risk_factors.append('Financial factors')
                elif 'Procedure' in feature:
                    risk_factors.append('Procedure type risk')
                elif 'Insurance' in feature:
                    risk_factors.append('Insurance coverage issue')
                elif 'Diagnosis' in feature:
                    risk_factors.append('Diagnosis-related risk')
        
        return {
            'prediction': prediction,
            'denial_probability': denial_prob,
            'confidence': abs(denial_prob - 0.5) * 2,
            'risk_factors': risk_factors
        }
    
    def evaluate_model(self, test_df):
        """Evaluate the ensemble model on test data"""
        print("📊 Evaluating ensemble model...")
        
        correct_predictions = 0
        total_predictions = 0
        results = []
        
        # Sample subset for evaluation (OpenAI API has rate limits)
        eval_df = test_df.sample(n=min(20, len(test_df)), random_state=42)
        
        for idx, row in eval_df.iterrows():
            if total_predictions % 5 == 0:
                print(f"Evaluated {total_predictions}/{len(eval_df)} claims...")
            
            # Get prediction
            prediction_result = self.predict_with_ensemble(row.to_dict())
            ensemble_pred = prediction_result['ensemble_prediction']
            
            # Compare with actual outcome
            actual_outcome = row['Outcome']
            actual_binary = 'DENIED' if actual_outcome == 'Denied' else 'APPROVED'
            
            is_correct = (ensemble_pred == actual_binary) or (ensemble_pred == 'NEEDS_REVIEW' and actual_outcome == 'Under Review')
            
            if is_correct:
                correct_predictions += 1
            
            results.append({
                'actual': actual_binary,
                'predicted': ensemble_pred,
                'confidence': prediction_result['confidence'],
                'correct': is_correct
            })
            
            total_predictions += 1
            
            # Rate limiting for OpenAI
            time.sleep(1)
        
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        avg_confidence = np.mean([r['confidence'] for r in results])
        
        print(f"\n✅ Ensemble Model Evaluation:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Average Confidence: {avg_confidence:.3f}")
        print(f"Total Evaluated: {total_predictions}")
        
        self.performance_metrics['ensemble_accuracy'] = accuracy
        self.performance_metrics['average_confidence'] = avg_confidence
        
        return results

def main():
    """Main function to test the OpenAI denial predictor"""
    print("🚀 OpenAI Healthcare Claim Denial Predictor")
    print("=" * 50)
    
    # Initialize predictor
    predictor = OpenAIDenialPredictor()
    
    # Generate data if needed
    if not os.path.exists('enhanced_claim_data.csv'):
        print("📊 Generating synthetic data...")
        os.system('python generate_synthetic_data.py')
    
    # Load data
    print("📂 Loading training data...")
    if os.path.exists('enhanced_claim_data.csv'):
        df = pd.read_csv('enhanced_claim_data.csv')
    else:
        df = pd.read_csv('claim_data.csv')
    print(f"Loaded {len(df)} claims")
    
    # Train traditional model
    X_test, y_test, y_pred = predictor.train_traditional_model(df)
    
    # Test with sample claims
    print("\n🧪 Testing with sample claims...")
    test_claims = [
        {
            'Procedure Code': '99285',  # High-cost emergency visit
            'Diagnosis Code': 'F32.9',  # Mental health (higher denial rate)
            'Insurance Type': 'Self-Pay',  # Higher risk
            'Billed Amount': 1200,
            'Allowed Amount': 800,
            'Paid Amount': 0,
            'Date of Service': '06/15/2024',
            'Reason Code': 'Authorization not obtained'
        },
        {
            'Procedure Code': '99213',  # Standard office visit
            'Diagnosis Code': 'I10',    # Hypertension (common, low risk)
            'Insurance Type': 'Medicare',
            'Billed Amount': 150,
            'Allowed Amount': 120,
            'Paid Amount': 115,
            'Date of Service': '07/20/2024',
            'Reason Code': 'Missing documentation'
        }
    ]
    
    for i, claim in enumerate(test_claims, 1):
        print(f"\n--- Test Claim {i} ---")
        print(f"Procedure: {claim['Procedure Code']}, Diagnosis: {claim['Diagnosis Code']}")
        print(f"Insurance: {claim['Insurance Type']}, Amount: ${claim['Billed Amount']}")
        
        result = predictor.predict_with_ensemble(claim)
        
        print(f"🎯 Prediction: {result['ensemble_prediction']}")
        print(f"📊 Confidence: {result['confidence']:.2f}")
        print(f"🔍 Risk Factors: {result['risk_factors']}")
        print(f"💭 Reasoning: {result['reasoning']}")
    
    print(f"\n✅ Model training and testing complete!")
    print(f"📈 Performance Metrics: {predictor.performance_metrics}")

if __name__ == "__main__":
    main() 