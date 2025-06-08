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
import sys
from dotenv import load_dotenv
load_dotenv(dotenv_path='/full/path/to/.env.local')
warnings.filterwarnings('ignore')

class OpenAIDenialPredictor:
    def __init__(self, openai_api_key=None):
        """Initialize the Enhanced Traditional ML + OpenAI predictor with advanced medical knowledge"""
        # Set up OpenAI
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            # Try to get from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("⚠️ Warning: No OpenAI API key provided. Using enhanced fallback model only.")
                self.client = None
        
        # Traditional ML components
        self.rf_model = None
        self.label_encoders = {}
        self.scaler = None
        self.feature_names = []
        
        # Performance tracking
        self.performance_metrics = {}
        
        # ENHANCED: Comprehensive medical knowledge base
        self.medical_patterns = self._initialize_enhanced_medical_knowledge()
        self.clinical_guidelines = self._initialize_enhanced_clinical_guidelines()
        
        # ENHANCED: Prediction history for dynamic learning
        self.prediction_history = []
        
    def _initialize_enhanced_medical_knowledge(self):
        """Initialize comprehensive medical knowledge base - ENHANCED VERSION"""
        return {
            'high_risk_procedures': {
                '99213': 'Office visit, established patient, moderate complexity',
                '99214': 'Office visit, established patient, moderate-high complexity', 
                '99215': 'Office visit, established patient, high complexity',
                '27447': 'Total knee arthroplasty',
                '27130': 'Total hip arthroplasty',
                '64483': 'Transforaminal epidural injection, lumbar',
                '64484': 'Transforaminal epidural injection, cervical',
                '12032': 'Layer closure of wounds 2.6-7.5 cm',
                '12034': 'Layer closure of wounds 7.6-12.5 cm',
                '93000': 'Electrocardiogram, routine ECG with interpretation',
                # Enhanced additions
                '33533': 'Coronary artery bypass, single arterial graft',
                '47562': 'Laparoscopic cholecystectomy',
                '43239': 'Upper endoscopy with biopsy',
                '49505': 'Inguinal hernia repair, initial',
                '29827': 'Arthroscopy, shoulder, surgical with rotator cuff repair'
            },
            'emergency_procedures': {
                '99281': 'Emergency department visit, problem-focused',
                '99282': 'Emergency department visit, expanded problem-focused',
                '99283': 'Emergency department visit, detailed',
                '99284': 'Emergency department visit, comprehensive',
                '99285': 'Emergency department visit, comprehensive high complexity',
                '36415': 'Routine venipuncture',
                '71020': 'Radiologic examination, chest, 2 views',
                # Enhanced additions
                '92950': 'Cardiopulmonary resuscitation',
                '31500': 'Emergency intubation',
                '36556': 'Central venous catheter insertion',
                '32551': 'Tube thoracostomy for pneumothorax'
            },
            'preventive_care': {
                '99391': 'Preventive medicine, infant (age under 1 year)',
                '99392': 'Preventive medicine, early childhood (age 1-4 years)',
                '99393': 'Preventive medicine, late childhood (age 5-11 years)',
                '99394': 'Preventive medicine, adolescent (age 12-17 years)',
                '99395': 'Preventive medicine, 18-39 years',
                '86803': 'Hepatitis B surface antibody',
                '82465': 'Cholesterol, serum, total',
                # Enhanced additions
                '80053': 'Comprehensive metabolic panel',
                '85025': 'Complete blood count with differential',
                '77057': 'Screening mammography, bilateral',
                '88150': 'Cervical cytology, manual screening'
            },
            'mental_health_procedures': {
                '90834': 'Psychotherapy, 45 minutes',
                '90837': 'Psychotherapy, 60 minutes', 
                '90847': 'Family psychotherapy with patient present',
                # Enhanced additions
                '90801': 'Psychiatric diagnostic interview',
                '90862': 'Medication management',
                '90832': 'Psychotherapy, 30 minutes',
                '90853': 'Group psychotherapy'
            },
            # NEW: High-value diagnoses
            'critical_diagnoses': {
                'I21.9': 'Acute myocardial infarction',
                'I50.9': 'Heart failure',
                'J44.1': 'COPD with acute exacerbation',
                'F32.9': 'Major depressive disorder',
                'F33.1': 'Major depression, recurrent',
                'E11.9': 'Type 2 diabetes mellitus'
            }
        }
    
    def _initialize_enhanced_clinical_guidelines(self):
        """Initialize enhanced evidence-based clinical guidelines"""
        return {
            'approval_criteria': {
                'preventive_care': 0.95,
                'emergency_care': 0.92,  # Increased from 0.85
                'mental_health': 0.82,   # Increased from 0.75
                'complex_procedures': 0.75, # Increased from 0.65
                'routine_office_visits': 0.88, # NEW category
                'diagnostic_procedures': 0.85   # NEW category
            },
            'cost_thresholds': {
                'low_cost': 500.0,      # Lowered for better granularity
                'moderate_cost': 2500.0, # Adjusted
                'high_cost': 10000.0,   # Lowered from 15000
                'very_high_cost': 35000.0 # Lowered from 50000
            },
            'risk_stratification': {
                'low_risk': {
                    'approval_probability': 0.92,
                    'criteria': ['routine_care', 'preventive', 'stable_condition']
                },
                'moderate_risk': {
                    'approval_probability': 0.78,
                    'criteria': ['complex_diagnosis', 'multiple_comorbidities']
                },
                'high_risk': {
                    'approval_probability': 0.65,
                    'criteria': ['experimental_treatment', 'very_high_cost']
                },
                'critical_risk': {
                    'approval_probability': 0.95,
                    'criteria': ['life_threatening', 'emergency', 'standard_of_care']
                }
            }
        }
    
    def create_enhanced_clinical_context(self, claim_data):
        """Create enhanced clinical context with comprehensive medical intelligence"""
        procedure_code = str(claim_data.get('Procedure Code', 'N/A'))
        diagnosis_code = str(claim_data.get('Diagnosis Code', 'N/A'))
        
        # Enhanced medical pattern recognition
        procedure_type = self._classify_procedure_type(procedure_code)
        diagnosis_severity = self._assess_diagnosis_severity(diagnosis_code)
        medical_necessity = self._evaluate_medical_necessity(procedure_code, diagnosis_code)
        
        context = f"""
        ENHANCED HEALTHCARE CLAIM ANALYSIS
        ==================================
        
        PATIENT CLINICAL DATA:
        - Procedure Code: {procedure_code} ({self._get_procedure_description(procedure_code)})
        - Diagnosis Code: {diagnosis_code} ({self._get_diagnosis_description(diagnosis_code)})
        - Insurance Type: {claim_data.get('Insurance Type', 'N/A')}
        - Billed Amount: ${claim_data.get('Billed Amount', 0):,}
        - Allowed Amount: ${claim_data.get('Allowed Amount', 0):,}
        - Date of Service: {claim_data.get('Date of Service', 'N/A')}
        - Reason Code: {claim_data.get('Reason Code', 'N/A')}
        
        CLINICAL INTELLIGENCE ASSESSMENT:
        - Procedure Category: {procedure_type}
        - Diagnosis Severity: {diagnosis_severity}
        - Medical Necessity Level: {medical_necessity}
        - Procedure-Diagnosis Alignment: {self._assess_procedure_diagnosis_alignment(procedure_code, diagnosis_code)}
        - Cost Appropriateness: {self._assess_cost_appropriateness(claim_data)}
        - Insurance Coverage Pattern: {self._assess_insurance_pattern(claim_data)}
        """
        return context
    
    def _classify_procedure_type(self, procedure_code):
        """Classify procedure type using enhanced medical knowledge"""
        if procedure_code in self.medical_patterns['emergency_procedures']:
            return "EMERGENCY_CARE"
        elif procedure_code in self.medical_patterns['preventive_care']:
            return "PREVENTIVE_CARE"  
        elif procedure_code in self.medical_patterns['mental_health_procedures']:
            return "MENTAL_HEALTH"
        elif procedure_code in self.medical_patterns['high_risk_procedures']:
            return "COMPLEX_PROCEDURE"
        elif procedure_code.startswith('99'):
            return "OFFICE_VISIT"
        else:
            return "DIAGNOSTIC_PROCEDURE"
    
    def _assess_diagnosis_severity(self, diagnosis_code):
        """Assess diagnosis severity using clinical guidelines"""
        if diagnosis_code in self.medical_patterns['critical_diagnoses']:
            return "CRITICAL"
        elif diagnosis_code.startswith(('I2', 'I5', 'J44')):  # Cardiac/respiratory
            return "HIGH"
        elif diagnosis_code.startswith(('F3', 'F4')):  # Mental health
            return "MODERATE"
        else:
            return "LOW"
    
    def _evaluate_medical_necessity(self, procedure_code, diagnosis_code):
        """Evaluate medical necessity using evidence-based criteria"""
        # Emergency procedures are always highly necessary
        if procedure_code in self.medical_patterns['emergency_procedures']:
            return "HIGH"
        
        # Critical diagnoses require high medical necessity
        if diagnosis_code in self.medical_patterns['critical_diagnoses']:
            return "HIGH"
        
        # Mental health has moderate-high necessity
        if procedure_code in self.medical_patterns['mental_health_procedures']:
            return "MODERATE_HIGH"
        
        # Preventive care has moderate necessity but high value
        if procedure_code in self.medical_patterns['preventive_care']:
            return "MODERATE"
        
        return "MODERATE"
    
    def _get_procedure_description(self, procedure_code):
        """Get procedure description from medical knowledge base"""
        all_procedures = {**self.medical_patterns['emergency_procedures'],
                         **self.medical_patterns['preventive_care'],
                         **self.medical_patterns['mental_health_procedures'],
                         **self.medical_patterns['high_risk_procedures']}
        return all_procedures.get(procedure_code, "Unknown procedure")
    
    def _get_diagnosis_description(self, diagnosis_code):
        """Get diagnosis description from medical knowledge base"""
        return self.medical_patterns['critical_diagnoses'].get(diagnosis_code, "Common diagnosis")
    
    def _assess_procedure_diagnosis_alignment(self, procedure_code, diagnosis_code):
        """Assess if procedure matches diagnosis appropriately"""
        # Emergency procedures align with critical diagnoses
        if (procedure_code in self.medical_patterns['emergency_procedures'] and 
            diagnosis_code in self.medical_patterns['critical_diagnoses']):
            return "EXCELLENT_ALIGNMENT"
        
        # Mental health procedures with mental health diagnoses
        if (procedure_code in self.medical_patterns['mental_health_procedures'] and
            diagnosis_code.startswith('F')):
            return "GOOD_ALIGNMENT"
        
        return "STANDARD_ALIGNMENT"
    
    def _assess_cost_appropriateness(self, claim_data):
        """Assess if billing amount is appropriate"""
        billed = float(claim_data.get('Billed Amount', 0))
        allowed = float(claim_data.get('Allowed Amount', 0))
        
        if allowed > 0:
            ratio = billed / allowed
            if ratio <= 1.1:
                return "APPROPRIATE"
            elif ratio <= 1.5:
                return "MODERATELY_HIGH"
            else:
                return "EXCESSIVE"
        return "UNABLE_TO_ASSESS"
    
    def _assess_insurance_pattern(self, claim_data):
        """Assess insurance type patterns"""
        insurance = claim_data.get('Insurance Type', '')
        if insurance == 'Medicare':
            return "GOVERNMENT_COVERAGE"
        elif insurance == 'Medicaid':
            return "GOVERNMENT_ASSISTANCE"
        elif insurance == 'Commercial':
            return "PRIVATE_COVERAGE"
        else:
            return "OTHER_COVERAGE"
    
    def get_enhanced_openai_prediction(self, claim_data, max_retries=3):
        """Get advanced prediction and reasoning from OpenAI with board-certified expertise"""
        if not self.client:
            return None
            
        context = self.create_enhanced_clinical_context(claim_data)
        
        prompt = f"""
        {context}
        
        CLINICAL DECISION FRAMEWORK:
        As a board-certified physician with expertise in healthcare administration and insurance policy, 
        perform a comprehensive medical review of this claim using evidence-based medicine principles.
        
        ANALYSIS REQUIREMENTS:
        1. MEDICAL NECESSITY ASSESSMENT:
           - Evaluate clinical appropriateness of procedure for diagnosis
           - Consider standard of care and clinical guidelines
           - Assess potential for improved patient outcomes
        
        2. REGULATORY COMPLIANCE:
           - Check compliance with insurance coverage policies
           - Verify documentation requirements
           - Assess billing code accuracy and appropriateness
        
        3. COST-EFFECTIVENESS ANALYSIS:
           - Compare billed vs allowed amounts for reasonableness
           - Evaluate cost vs clinical benefit ratio
           - Consider alternative treatment options
        
        4. RISK ASSESSMENT:
           - Identify potential patient harm if denied
           - Assess denial appeal likelihood
           - Consider regulatory compliance issues
        
        CLINICAL DECISION CRITERIA:
        - Emergency care with life-threatening conditions: APPROVE unless clear fraud
        - Preventive care within guidelines: APPROVE for quality metrics
        - Mental health within standard frequency: APPROVE for patient safety
        - Complex procedures: Evaluate medical necessity and outcomes data
        - Routine office visits: Verify frequency and medical need
        
        Provide your expert clinical assessment in this JSON format:
        {{
            "prediction": "APPROVED|DENIED|NEEDS_REVIEW",
            "confidence": 0.85,
            "medical_necessity_score": 0.90,
            "cost_appropriateness_score": 0.75,
            "risk_factors": ["factor1", "factor2"],
            "clinical_reasoning": ["reason1", "reason2", "reason3"],
            "denial_risk_assessment": "LOW|MODERATE|HIGH",
            "patient_harm_risk": "MINIMAL|MODERATE|SIGNIFICANT",
            "quality_metrics_impact": "POSITIVE|NEUTRAL|NEGATIVE",
            "appeal_likelihood": "LOW|MODERATE|HIGH"
        }}
        """
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",  # Use latest GPT-4 for enhanced medical reasoning
                    messages=[
                        {"role": "system", "content": "You are a board-certified physician specializing in healthcare administration, insurance policy, and evidence-based medicine. You provide thorough clinical assessments with detailed reasoning."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,  # Very low temperature for consistent clinical decisions
                    max_tokens=800    # Increased for comprehensive analysis
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
                
                # Enhanced validation and normalization
                prediction = result.get('prediction', 'NEEDS_REVIEW').upper()
                if prediction not in ['APPROVED', 'DENIED', 'NEEDS_REVIEW']:
                    prediction = 'NEEDS_REVIEW'
                
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                
                # Enhanced output with clinical scores
                enhanced_result = {
                    'prediction': prediction,
                    'confidence': confidence,
                    'medical_necessity_score': float(result.get('medical_necessity_score', 0.5)),
                    'cost_appropriateness_score': float(result.get('cost_appropriateness_score', 0.5)),
                    'risk_factors': result.get('risk_factors', []),
                    'clinical_reasoning': result.get('clinical_reasoning', result.get('reasoning', ['Enhanced OpenAI analysis completed'])),
                    'denial_risk_assessment': result.get('denial_risk_assessment', 'MODERATE'),
                    'patient_harm_risk': result.get('patient_harm_risk', 'MINIMAL'),
                    'quality_metrics_impact': result.get('quality_metrics_impact', 'NEUTRAL'),
                    'appeal_likelihood': result.get('appeal_likelihood', 'MODERATE')
                }
                
                return enhanced_result
                
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
    
    def _calculate_dynamic_weights(self, claim_data, openai_result=None):
        """Calculate dynamic ensemble weights based on claim characteristics"""
        procedure_code = str(claim_data.get('Procedure Code', ''))
        diagnosis_code = str(claim_data.get('Diagnosis Code', ''))
        
        # Base weights: OpenAI, Traditional ML, Rules
        base_weights = [0.60, 0.30, 0.10]
        
        # Emergency procedures: Trust OpenAI more (clinical expertise)
        if procedure_code in self.medical_patterns['emergency_procedures']:
            weights = [0.80, 0.15, 0.05]
            
        # Preventive care: Balance OpenAI with rules (policy-driven)
        elif procedure_code in self.medical_patterns['preventive_care']:
            weights = [0.55, 0.15, 0.30]
            
        # Mental health: High OpenAI weight (clinical complexity)
        elif procedure_code in self.medical_patterns['mental_health_procedures']:
            weights = [0.75, 0.20, 0.05]
            
        # Office visits: Traditional ML better for patterns
        elif procedure_code.startswith('99'):
            weights = [0.50, 0.45, 0.05]
            
        # Complex procedures: Balanced approach
        elif procedure_code in self.medical_patterns['high_risk_procedures']:
            weights = [0.65, 0.30, 0.05]
            
        else:
            weights = base_weights
        
        # Adjust based on OpenAI confidence if available
        if openai_result and 'confidence' in openai_result:
            confidence = openai_result['confidence']
            if confidence > 0.8:  # High confidence - trust OpenAI more
                weights[0] += 0.1
                weights[1] -= 0.05
                weights[2] -= 0.05
            elif confidence < 0.6:  # Low confidence - trust ML more
                weights[0] -= 0.1
                weights[1] += 0.1
        
        # Normalize weights
        total = sum(weights)
        weights = [w/total for w in weights]
        
        return weights
    
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
    
    def predict_with_enhanced_ensemble(self, claim_data):
        """Enhanced ensemble prediction with dynamic weighting and comprehensive analysis"""
        results = {
            'traditional_prediction': None,
            'openai_prediction': None,
            'rules_prediction': None,
            'ensemble_prediction': None,
            'confidence': 0.5,
            'clinical_reasoning': [],
            'risk_factors': [],
            'ensemble_weights': None,
            'clinical_scores': {}
        }
        
        # Get traditional ML prediction
        if self.rf_model is not None:
            try:
                traditional_result = self.predict_traditional(claim_data)
                results['traditional_prediction'] = traditional_result
            except Exception as e:
                print(f"Traditional model error: {e}")
        
        # Get enhanced OpenAI prediction
        openai_result = self.get_enhanced_openai_prediction(claim_data)
        if openai_result:
            results['openai_prediction'] = openai_result
        
        # Get rules-based prediction
        rules_result = self._predict_with_enhanced_rules(claim_data)
        results['rules_prediction'] = rules_result
        
        # Calculate dynamic weights based on claim characteristics
        weights = self._calculate_dynamic_weights(claim_data, openai_result)
        results['ensemble_weights'] = {
            'openai_weight': weights[0],
            'traditional_weight': weights[1], 
            'rules_weight': weights[2]
        }
        
        # Enhanced ensemble logic with dynamic weighting
        if results['traditional_prediction'] and results['openai_prediction'] and results['rules_prediction']:
            # All three models available - use dynamic weighted ensemble
            
            # Convert predictions to denial probability scores
            scores = []
            
            # OpenAI score
            if openai_result['prediction'] == 'DENIED':
                openai_score = openai_result['confidence']
            elif openai_result['prediction'] == 'APPROVED':
                openai_score = 1 - openai_result['confidence']
            else:  # NEEDS_REVIEW
                openai_score = 0.5
            scores.append(openai_score)
            
            # Traditional ML score
            trad_score = results['traditional_prediction']['denial_probability']
            scores.append(trad_score)
            
            # Rules score
            rules_score = results['rules_prediction']['denial_probability']
            scores.append(rules_score)
            
            # Calculate weighted ensemble score
            ensemble_denial_score = sum(w * s for w, s in zip(weights, scores))
            
            # Dynamic thresholds based on claim type
            procedure_code = str(claim_data.get('Procedure Code', ''))
            if procedure_code in self.medical_patterns['emergency_procedures']:
                # Lower threshold for emergency denials (favor approval)
                approve_threshold = 0.25
                deny_threshold = 0.80
            elif procedure_code in self.medical_patterns['preventive_care']:
                # Favor approval for preventive care
                approve_threshold = 0.20
                deny_threshold = 0.75
            else:
                # Standard thresholds
                approve_threshold = 0.30
                deny_threshold = 0.70
            
            if ensemble_denial_score > deny_threshold:
                final_prediction = 'DENIED'
            elif ensemble_denial_score < approve_threshold:
                final_prediction = 'APPROVED'
            else:
                final_prediction = 'NEEDS_REVIEW'
            
            results['ensemble_prediction'] = final_prediction
            results['confidence'] = abs(ensemble_denial_score - 0.5) * 2  # Convert to confidence
            
            # Enhanced reasoning and clinical scores
            if openai_result:
                results['clinical_reasoning'] = openai_result.get('clinical_reasoning', [])
                results['risk_factors'] = openai_result.get('risk_factors', [])
                results['clinical_scores'] = {
                    'medical_necessity': openai_result.get('medical_necessity_score', 0.5),
                    'cost_appropriateness': openai_result.get('cost_appropriateness_score', 0.5),
                    'denial_risk': openai_result.get('denial_risk_assessment', 'MODERATE'),
                    'patient_harm_risk': openai_result.get('patient_harm_risk', 'MINIMAL'),
                    'ensemble_denial_score': ensemble_denial_score
                }
            
        elif results['openai_prediction']:
            # Fallback to OpenAI only
            results = self._fallback_to_single_model(results, 'openai')
            
        elif results['traditional_prediction']:
            # Fallback to traditional only  
            results = self._fallback_to_single_model(results, 'traditional')
            
        elif results['rules_prediction']:
            # Fallback to rules only
            results = self._fallback_to_single_model(results, 'rules')
            
        else:
            # Ultimate fallback
            results['ensemble_prediction'] = 'NEEDS_REVIEW'
            results['confidence'] = 0.5
            results['clinical_reasoning'] = ['Unable to generate prediction - manual review required']
            results['risk_factors'] = ['All prediction systems unavailable']
        
        # Store prediction for learning
        self.prediction_history.append({
            'claim_data': claim_data,
            'prediction': results['ensemble_prediction'],
            'confidence': results['confidence'],
            'weights': weights
        })
        
        return results
    
    def _predict_with_enhanced_rules(self, claim_data):
        """Enhanced rules-based prediction using clinical guidelines"""
        procedure_code = str(claim_data.get('Procedure Code', ''))
        diagnosis_code = str(claim_data.get('Diagnosis Code', ''))
        billed_amount = float(claim_data.get('Billed Amount', 0))
        insurance_type = claim_data.get('Insurance Type', '')
        
        # Initialize denial probability
        denial_prob = 0.3  # Base rate
        
        # Apply clinical guidelines
        procedure_type = self._classify_procedure_type(procedure_code)
        approval_criteria = self.clinical_guidelines['approval_criteria']
        
        # Adjust based on procedure type
        if procedure_type in approval_criteria:
            expected_approval_rate = approval_criteria[procedure_type]
            denial_prob = 1 - expected_approval_rate
        
        # Cost-based adjustments
        cost_thresholds = self.clinical_guidelines['cost_thresholds']
        if billed_amount > cost_thresholds['very_high_cost']:
            denial_prob += 0.2
        elif billed_amount > cost_thresholds['high_cost']:
            denial_prob += 0.1
        elif billed_amount < cost_thresholds['low_cost']:
            denial_prob -= 0.1
        
        # Insurance type adjustments
        if insurance_type == 'Commercial':
            denial_prob += 0.05  # Slightly higher denial rate
        elif insurance_type in ['Medicare', 'Medicaid']:
            denial_prob -= 0.05  # Government programs more consistent
        
        # Clamp to valid range
        denial_prob = max(0.0, min(1.0, denial_prob))
        
        prediction = 'DENIED' if denial_prob > 0.5 else 'APPROVED'
        
        return {
            'prediction': prediction,
            'denial_probability': denial_prob,
            'confidence': abs(denial_prob - 0.5) * 2,
            'reasoning': [f'Rules-based assessment using {procedure_type} guidelines']
        }
    
    def _fallback_to_single_model(self, results, model_type):
        """Fallback when only one model is available"""
        if model_type == 'openai' and results['openai_prediction']:
            pred = results['openai_prediction']
            results['ensemble_prediction'] = pred['prediction']
            results['confidence'] = pred['confidence']
            results['clinical_reasoning'] = pred.get('clinical_reasoning', pred.get('reasoning', []))
            results['risk_factors'] = pred.get('risk_factors', [])
            if 'medical_necessity_score' in pred:
                results['clinical_scores'] = {
                    'medical_necessity': pred['medical_necessity_score'],
                    'cost_appropriateness': pred['cost_appropriateness_score']
                }
                
        elif model_type == 'traditional' and results['traditional_prediction']:
            pred = results['traditional_prediction']
            results['ensemble_prediction'] = pred['prediction']
            results['confidence'] = pred['confidence']
            results['clinical_reasoning'] = ['Traditional ML model prediction']
            results['risk_factors'] = pred['risk_factors']
            
        elif model_type == 'rules' and results['rules_prediction']:
            pred = results['rules_prediction']
            results['ensemble_prediction'] = pred['prediction']
            results['confidence'] = pred['confidence']
            results['clinical_reasoning'] = pred['reasoning']
            results['risk_factors'] = ['Rules-based assessment']
        
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
            
            # Get enhanced prediction
            prediction_result = self.predict_with_enhanced_ensemble(row.to_dict())
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

    def get_denial_prediction_json(self, claim_data):
        """
        Get a clean JSON prediction output with likelihood, prediction, reasons, and next steps
        
        Args:
            claim_data (dict): Claim information including procedure code, diagnosis, amounts, etc.
            
        Returns:
            dict: JSON with likelihood (%), prediction (accepted/denied), denial_reasons, and next_steps
        """
        try:
            # Get the comprehensive ensemble prediction
            ensemble_results = self.predict_with_enhanced_ensemble(claim_data)
            
            # Extract key information
            prediction = ensemble_results.get('ensemble_prediction', 'NEEDS_REVIEW')
            confidence = ensemble_results.get('confidence', 0.5)
            clinical_reasoning = ensemble_results.get('clinical_reasoning', [])
            risk_factors = ensemble_results.get('risk_factors', [])
            clinical_scores = ensemble_results.get('clinical_scores', {})
            
            # Convert prediction format
            if prediction == 'APPROVED':
                prediction_label = 'accepted'
                likelihood_percent = round((1 - confidence) * 100, 1)  # Lower likelihood of denial
            elif prediction == 'DENIED':
                prediction_label = 'denied'
                likelihood_percent = round(confidence * 100, 1)  # Higher likelihood of denial
            else:  # NEEDS_REVIEW
                prediction_label = 'review_required'
                likelihood_percent = 50.0
            
            # Generate denial reasons if prediction is denied or needs review
            denial_reasons = []
            if prediction in ['DENIED', 'NEEDS_REVIEW']:
                # Extract reasons from clinical reasoning and risk factors
                if clinical_reasoning:
                    denial_reasons.extend([reason for reason in clinical_reasoning if reason])
                
                if risk_factors:
                    denial_reasons.extend([f"Risk factor: {factor}" for factor in risk_factors if factor])
                
                # Add specific denial reasons based on clinical analysis
                if clinical_scores:
                    medical_necessity = clinical_scores.get('medical_necessity', 0.5)
                    cost_appropriateness = clinical_scores.get('cost_appropriateness', 0.5)
                    
                    if medical_necessity < 0.6:
                        denial_reasons.append("Low medical necessity score - procedure may not be clinically required")
                    
                    if cost_appropriateness < 0.6:
                        denial_reasons.append("Cost concerns - billed amount exceeds typical allowable limits")
                
                # Add procedure-specific reasons
                procedure_code = str(claim_data.get('Procedure Code', ''))
                billed_amount = float(claim_data.get('Billed Amount', 0))
                
                if billed_amount > 10000:
                    denial_reasons.append("High-cost claim requiring additional documentation")
                
                if not denial_reasons:
                    denial_reasons = ["Requires additional review based on claim characteristics"]
            
            # Generate next steps based on prediction
            next_steps = []
            if prediction_label == 'accepted':
                next_steps = [
                    "Claim approved for processing",
                    "Verify beneficiary eligibility",
                    "Process payment according to contract terms"
                ]
            elif prediction_label == 'denied':
                next_steps = [
                    "Notify provider of denial decision",
                    "Send denial letter with specific reasons",
                    "Provider may submit additional documentation for reconsideration",
                    "Patient may appeal decision within 60 days"
                ]
            else:  # review_required
                next_steps = [
                    "Flag claim for manual medical review",
                    "Request additional clinical documentation from provider",
                    "Assign to clinical reviewer for assessment",
                    "Expected review completion within 5-7 business days"
                ]
            
            # Clean up denial reasons (remove duplicates and empty strings)
            denial_reasons = list(set([reason.strip() for reason in denial_reasons if reason and reason.strip()]))
            
            # Create the final JSON response
            json_response = {
                "likelihood_percent": likelihood_percent,
                "prediction": prediction_label,
                "denial_reasons": denial_reasons if denial_reasons else None,
                "next_steps": next_steps,
                "confidence_score": round(confidence * 100, 1),
                "analysis_details": {
                    "procedure_code": claim_data.get('Procedure Code', 'N/A'),
                    "diagnosis_code": claim_data.get('Diagnosis Code', 'N/A'),
                    "billed_amount": claim_data.get('Billed Amount', 0),
                    "insurance_type": claim_data.get('Insurance Type', 'N/A')
                }
            }
            
            return json_response
            
        except Exception as e:
            # Fallback response in case of errors
            return {
                "likelihood_percent": 50.0,
                "prediction": "error",
                "denial_reasons": [f"Error processing claim: {str(e)}"],
                "next_steps": ["Manual review required due to processing error"],
                "confidence_score": 0.0,
                "analysis_details": {
                    "error": str(e)
                }
            }

def format_denial_reasons(claim_data, denial_reasons):
    # Check for missing codes
    procedure_code = claim_data.get('procedure_code') or claim_data.get('Procedure Code')
    diagnosis_code = claim_data.get('diagnosis_code') or claim_data.get('Diagnosis Code')
    formatted = []
    for reason in denial_reasons:
        # Remove underscores and capitalize
        if reason == "unknown_procedure":
            formatted.append("Risk factor: Unknown procedure")
        elif reason == "unknown_diagnosis":
            formatted.append("Risk factor: Unknown diagnosis")
        elif reason == "lack_of_documentation":
            formatted.append("Risk factor: Lack of documentation")
        elif "procedure and diagnosis codes are not provided" in reason:
            if not procedure_code or not diagnosis_code:
                formatted.append("The procedure and diagnosis codes are not provided, making it difficult to assess the clinical appropriateness.")
            # else: skip this reason
        else:
            # Capitalize first letter, replace underscores with spaces
            formatted.append(reason.replace("_", " ").capitalize())
    return formatted

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
        
        # Test the new JSON prediction function
        json_result = predictor.get_denial_prediction_json(claim)
        
        print(f"🎯 Prediction: {json_result['prediction']}")
        print(f"📊 Likelihood of Denial: {json_result['likelihood_percent']}%")
        print(f"🔍 Confidence Score: {json_result['confidence_score']}%")
        if json_result['denial_reasons']:
            print(f"💭 Denial Reasons: {json_result['denial_reasons']}")
        print(f"📋 Next Steps: {json_result['next_steps']}")
        
        # Pretty print the full JSON
        print(f"\n📄 Full JSON Output:")
        print(json.dumps(json_result, indent=2))
    
    print(f"\n✅ Model training and testing complete!")
    print(f"📈 Performance Metrics: {predictor.performance_metrics}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        claim_data = json.loads(sys.argv[1])
        predictor = OpenAIDenialPredictor()
        result = predictor.get_denial_prediction_json(claim_data)
        # Format denial reasons before output
        if result.get('denial_reasons'):
            result['denial_reasons'] = format_denial_reasons(claim_data, result['denial_reasons'])
        print("DEBUG: Prediction JSON to return:", json.dumps(result), file=sys.stderr)
        print(json.dumps(result))
    else:
        # Optionally, you can call your main() for CLI/testing
        main() 