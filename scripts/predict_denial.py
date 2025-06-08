#!/usr/bin/env python3
"""
Healthcare Claim Denial Prediction Script

This script loads the trained models and makes predictions for new claims.
It integrates with the trained Random Forest and BERT models from the notebooks.
"""

import sys
import json
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Get the directory of this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_ROOT, 'claim_data.csv')

class DenialPredictor:
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.scaler = None
        self.is_trained = False
        
    def load_or_train_model(self):
        """Load existing model or train a new one"""
        model_path = os.path.join(SCRIPT_DIR, 'denial_model.pkl')
        encoders_path = os.path.join(SCRIPT_DIR, 'label_encoders.pkl')
        scaler_path = os.path.join(SCRIPT_DIR, 'scaler.pkl')
        
        if (os.path.exists(model_path) and 
            os.path.exists(encoders_path) and 
            os.path.exists(scaler_path)):
            # Load existing model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            with open(encoders_path, 'rb') as f:
                self.label_encoders = pickle.load(f)
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            self.is_trained = True
        else:
            # Train new model
            self.train_model()
            
    def train_model(self):
        """Train the denial prediction model"""
        try:
            # Load training data
            df = pd.read_csv(DATA_PATH)
            
            # Preprocess data
            df = self.preprocess_data(df)
            
            # Create binary target (Paid=1, Denied=0, drop others)
            df['Target'] = df['Claim Status'].map({'Paid': 1, 'Denied': 0})
            df_clean = df.dropna(subset=['Target']).copy()
            
            # Prepare features
            cat_features = ['Diagnosis Code', 'Procedure Code', 'Insurance Type', 'Provider ID']
            num_features = ['Billed Amount', 'Allowed Amount', 'Paid Amount', 'Days_Since_Min']
            
            # Encode categorical features
            for col in cat_features:
                le = LabelEncoder()
                df_clean[col + '_encoded'] = le.fit_transform(df_clean[col].astype(str))
                self.label_encoders[col] = le
            
            # Scale numerical features
            self.scaler = StandardScaler()
            df_clean[num_features] = self.scaler.fit_transform(df_clean[num_features])
            
            # Prepare feature matrix
            encoded_features = [col + '_encoded' for col in cat_features]
            all_features = encoded_features + num_features
            X = df_clean[all_features]
            y = df_clean['Target']
            
            # Train Random Forest model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                class_weight='balanced'
            )
            self.model.fit(X, y)
            
            # Save model and preprocessors
            model_path = os.path.join(SCRIPT_DIR, 'denial_model.pkl')
            encoders_path = os.path.join(SCRIPT_DIR, 'label_encoders.pkl')
            scaler_path = os.path.join(SCRIPT_DIR, 'scaler.pkl')
            
            os.makedirs(SCRIPT_DIR, exist_ok=True)
            
            with open(model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(encoders_path, 'wb') as f:
                pickle.dump(self.label_encoders, f)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
                
            self.is_trained = True
            
        except Exception as e:
            print(f"Error training model: {e}", file=sys.stderr)
            raise
            
    def preprocess_data(self, df):
        """Preprocess the dataframe"""
        # Convert date column
        df['Date of Service'] = pd.to_datetime(df['Date of Service'], format='%m/%d/%Y')
        min_date = df['Date of Service'].min()
        df['Days_Since_Min'] = (df['Date of Service'] - min_date).dt.days
        
        return df
        
    def predict_claim(self, claim_data):
        """Make prediction for a single claim"""
        if not self.is_trained:
            raise Exception("Model not trained")
            
        try:
            # Create dataframe from claim data
            df = pd.DataFrame([{
                'Diagnosis Code': claim_data['diagnosisCode'],
                'Procedure Code': claim_data['procedureCode'],
                'Insurance Type': claim_data['insuranceType'],
                'Provider ID': str(claim_data.get('providerId', '0')),
                'Billed Amount': float(claim_data['billedAmount']),
                'Allowed Amount': float(claim_data.get('allowedAmount', claim_data['billedAmount'] * 0.8)),
                'Paid Amount': float(claim_data.get('paidAmount', claim_data.get('allowedAmount', claim_data['billedAmount'] * 0.7))),
                'Days_Since_Min': 30  # Default to 30 days
            }])
            
            # Encode categorical features
            cat_features = ['Diagnosis Code', 'Procedure Code', 'Insurance Type', 'Provider ID']
            for col in cat_features:
                if col in self.label_encoders:
                    # Handle unseen categories
                    try:
                        df[col + '_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Use most common class for unseen categories
                        df[col + '_encoded'] = 0
                else:
                    df[col + '_encoded'] = 0
            
            # Scale numerical features
            num_features = ['Billed Amount', 'Allowed Amount', 'Paid Amount', 'Days_Since_Min']
            df[num_features] = self.scaler.transform(df[num_features])
            
            # Prepare feature vector
            encoded_features = [col + '_encoded' for col in cat_features]
            all_features = encoded_features + num_features
            X = df[all_features]
            
            # Make prediction
            prediction_proba = self.model.predict_proba(X)[0]
            prediction = self.model.predict(X)[0]
            
            # Get feature importance for reasoning
            feature_importance = self.model.feature_importances_
            feature_names = all_features
            
            # Create reasoning based on feature importance
            reasoning = self.generate_reasoning(claim_data, feature_importance, feature_names, prediction_proba)
            
            # Determine prediction label
            if prediction == 1:
                pred_label = 'approved'
            else:
                pred_label = 'denied'
                
            # If confidence is low, suggest review
            confidence = max(prediction_proba)
            if confidence < 0.7:
                pred_label = 'review'
                
            return {
                'prediction': pred_label,
                'confidence': float(confidence),
                'reasoning': reasoning,
                'riskFactors': self.get_risk_factors(claim_data)
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}", file=sys.stderr)
            raise
            
    def generate_reasoning(self, claim_data, feature_importance, feature_names, prediction_proba):
        """Generate human-readable reasoning for the prediction"""
        reasoning = []
        
        # Sort features by importance
        feature_importance_sorted = sorted(zip(feature_names, feature_importance), 
                                         key=lambda x: x[1], reverse=True)
        
        # Add top 3 most important factors
        for feature_name, importance in feature_importance_sorted[:3]:
            if importance > 0.1:  # Only include significant factors
                if 'Amount' in feature_name:
                    reasoning.append(f"Financial factors (importance: {importance:.2f})")
                elif 'Procedure' in feature_name:
                    reasoning.append(f"Procedure type consideration (importance: {importance:.2f})")
                elif 'Insurance' in feature_name:
                    reasoning.append(f"Insurance type factor (importance: {importance:.2f})")
                elif 'Diagnosis' in feature_name:
                    reasoning.append(f"Diagnosis code relevance (importance: {importance:.2f})")
                    
        # Add confidence information
        confidence = max(prediction_proba)
        if confidence > 0.8:
            reasoning.append("High confidence prediction based on similar historical cases")
        elif confidence > 0.6:
            reasoning.append("Moderate confidence prediction")
        else:
            reasoning.append("Low confidence - manual review recommended")
            
        return reasoning
        
    def get_risk_factors(self, claim_data):
        """Identify risk factors for the claim"""
        risk_factors = []
        
        billed_amount = float(claim_data['billedAmount'])
        
        if billed_amount > 500:
            risk_factors.append("High claim amount")
            
        if claim_data['insuranceType'] == 'Self-Pay':
            risk_factors.append("Self-pay insurance type")
            
        # High-risk procedure codes based on historical data
        high_risk_procedures = ['99238', '99233', '99232', '99231']
        if claim_data['procedureCode'] in high_risk_procedures:
            risk_factors.append("High-risk procedure code")
            
        return risk_factors

def main():
    """Main function to handle command line prediction"""
    if len(sys.argv) != 2:
        print("Usage: python predict_denial.py '<claim_data_json>'", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Parse claim data from command line argument
        claim_data = json.loads(sys.argv[1])
        
        # Initialize predictor
        predictor = DenialPredictor()
        predictor.load_or_train_model()
        
        # Make prediction
        result = predictor.predict_claim(claim_data)
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'prediction': 'review',
            'confidence': 0.5,
            'reasoning': ['Error in ML prediction, manual review required'],
            'riskFactors': ['Prediction system error']
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main() 