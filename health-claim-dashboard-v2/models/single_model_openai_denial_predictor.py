#!/usr/bin/env python3
"""
OpenAI Healthcare Claim Denial Predictor - Single Model Version

Uses OpenAI GPT model for denial prediction and reasoning.
"""

import json
import os
import sys
from datetime import datetime
import openai
from openai import OpenAI
import time
from dotenv import load_dotenv

# Load environment variables from .env.local in the project root
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env.local'))

class SingleModelOpenAIDenialPredictor:
    def __init__(self, openai_api_key=None):
        """Initialize the OpenAI predictor"""
        # Set up OpenAI
        if openai_api_key:
            self.client = OpenAI(api_key=openai_api_key)
        else:
            # Try to get from environment
            api_key = os.getenv('OPENAI_API_KEY')
            if api_key:
                self.client = OpenAI(api_key=api_key)
            else:
                print("⚠️ Warning: No OpenAI API key provided.", file=sys.stderr)
                self.client = None
        
        # Medical knowledge base
        self.medical_patterns = self._initialize_medical_knowledge()
        
    def _initialize_medical_knowledge(self):
        """Initialize medical knowledge base"""
        return {
            'procedure_categories': {
                'evaluation_management': {
                    'range': '99201-99499',
                    'description': 'Office visits, consultations, and other evaluation services'
                },
                'surgery': {
                    'range': '10004-69990',
                    'description': 'Surgical procedures and interventions'
                },
                'radiology': {
                    'range': '70010-79999',
                    'description': 'Diagnostic imaging and radiation therapy'
                },
                'pathology_lab': {
                    'range': '80047-89398',
                    'description': 'Laboratory tests and pathology services'
                },
                'medicine': {
                    'range': '90281-99607',
                    'description': 'Non-surgical medical services and procedures'
                },
                'anesthesia': {
                    'range': '00100-01999',
                    'description': 'Anesthesia services'
                }
            },
            'diagnosis_categories': {
                'acute_conditions': {
                    'ranges': ['A00-B99', 'J00-J99', 'S00-T88'],
                    'description': 'Infectious diseases, respiratory conditions, and injuries'
                },
                'chronic_conditions': {
                    'ranges': ['E00-E89', 'I00-I99', 'M00-M99'],
                    'description': 'Endocrine, circulatory, and musculoskeletal disorders'
                },
                'mental_health': {
                    'ranges': ['F01-F99'],
                    'description': 'Mental and behavioral disorders'
                },
                'preventive_care': {
                    'ranges': ['Z00-Z99'],
                    'description': 'Preventive care and health status factors'
                },
                'specialty_care': {
                    'ranges': ['C00-D49', 'G00-G99', 'H00-H95', 'K00-K95', 'L00-L99', 'N00-N99'],
                    'description': 'Specialized medical conditions and treatments'
                }
            }
        }

    
    def create_clinical_context(self, claim_data):
        """Create clinical context for the claim"""
        procedure_code = str(claim_data.get('procedure_code', 'N/A'))
        diagnosis_code = str(claim_data.get('diagnosis_code', 'N/A'))
        
        # Get diagnosis category and description
        diagnosis_category, diagnosis_description = self._get_diagnosis_category(diagnosis_code)
        
        context = f"""
        HEALTHCARE CLAIM ANALYSIS
        ========================
        
        PATIENT CLINICAL DATA:
        - Procedure Code: {procedure_code} ({self._get_procedure_description(procedure_code)})
        - Diagnosis Code: {diagnosis_code} ({diagnosis_description})
        - Insurance Type: {claim_data.get('insurance_type', 'N/A')}
        - Billed Amount: ${claim_data.get('billed_amount', 0):,}
        
        CLINICAL ASSESSMENT:
        - Procedure Category: {self._classify_procedure_type(procedure_code)}
        - Diagnosis Category: {diagnosis_category}
        - Cost Appropriateness: {self._assess_cost_appropriateness(claim_data)}
        - Insurance Coverage Pattern: {self._assess_insurance_pattern(claim_data)}
        """
        return context
    
    def _get_procedure_description(self, procedure_code):
        """Get procedure description from medical knowledge base"""
        try:
            # Handle N/A or empty procedure codes
            if not procedure_code or procedure_code == 'N/A':
                return "Unknown procedure"
                
            # Try to convert to integer and get description
            procedure_code_int = int(procedure_code)
            for category, info in self.medical_patterns['procedure_categories'].items():
                start, end = map(int, info['range'].split('-'))
                if start <= procedure_code_int <= end:
                    return f"{info['description']} ({category.replace('_', ' ').title()})"
            return "Unknown procedure"
        except (ValueError, TypeError):
            return "Unknown procedure"
    
    def _classify_procedure_type(self, procedure_code):
        """Classify procedure type using enhanced medical knowledge"""
        try:
            # Handle N/A or empty procedure codes
            if not procedure_code or procedure_code == 'N/A':
                return "UNKNOWN_PROCEDURE"
                
            # Try to convert to integer and classify
            procedure_code_int = int(procedure_code)
            for category, info in self.medical_patterns['procedure_categories'].items():
                start, end = map(int, info['range'].split('-'))
                if start <= procedure_code_int <= end:
                    return category.upper()
            return "UNKNOWN_PROCEDURE"
        except (ValueError, TypeError):
            return "UNKNOWN_PROCEDURE"
    
    def _get_diagnosis_category(self, diagnosis_code):
        """Get the category of a diagnosis code"""
        for category, info in self.medical_patterns['diagnosis_categories'].items():
            for code_range in info['ranges']:
                start, end = code_range.split('-')
                if start <= diagnosis_code <= end:
                    return category, info['description']
        return "UNKNOWN", "Unknown diagnosis category"
    
    def _assess_cost_appropriateness(self, claim_data):
        """Assess if billing amount is appropriate"""
        billed = float(claim_data.get('billed_amount', 0))
        allowed = float(claim_data.get('allowed_amount', 0))
        
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
        insurance = claim_data.get('insurance_type', '')
        if insurance == 'Medicare':
            return "GOVERNMENT_COVERAGE"
        elif insurance == 'Medicaid':
            return "GOVERNMENT_ASSISTANCE"
        elif insurance == 'Commercial':
            return "PRIVATE_COVERAGE"
        else:
            return "OTHER_COVERAGE"
    
    def get_prediction(self, claim_data, max_retries=3):
        """Get prediction from OpenAI"""
        if not self.client:
            return None
            
        context = self.create_clinical_context(claim_data)
        
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
            "prediction": "APPROVED|DENIED",
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
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a board-certified physician specializing in healthcare administration, insurance policy, and evidence-based medicine. You provide thorough clinical assessments with detailed reasoning."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=800
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
                prediction = result.get('prediction', 'DENIED').upper()
                confidence = float(result.get('confidence', 0.5))
                confidence = max(0.0, min(1.0, confidence))  # Clamp to [0,1]
                
                # Only allow APPROVED or DENIED
                if prediction not in ['APPROVED', 'DENIED']:
                    if confidence >= 0.5:
                        prediction = 'DENIED'
                    else:
                        prediction = 'APPROVED'
                
                return {
                    'prediction': prediction,
                    'confidence': confidence,
                    'medical_necessity_score': float(result.get('medical_necessity_score', 0.5)),
                    'cost_appropriateness_score': float(result.get('cost_appropriateness_score', 0.5)),
                    'risk_factors': result.get('risk_factors', []),
                    'clinical_reasoning': result.get('clinical_reasoning', []),
                    'denial_risk_assessment': result.get('denial_risk_assessment', 'MODERATE'),
                    'patient_harm_risk': result.get('patient_harm_risk', 'MINIMAL'),
                    'quality_metrics_impact': result.get('quality_metrics_impact', 'NEUTRAL'),
                    'appeal_likelihood': result.get('appeal_likelihood', 'MODERATE')
                }
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error (attempt {attempt + 1}): {e}", file=sys.stderr)
                if attempt == max_retries - 1:
                    return None
                    
            except Exception as e:
                print(f"OpenAI API error (attempt {attempt + 1}): {e}", file=sys.stderr)
                if attempt == max_retries - 1:
                    return None
                time.sleep(2)  # Wait before retry
        
        return None
    
    def get_denial_prediction_json(self, claim_data):
        """Get a clean JSON prediction output"""
        try:
            # Get the prediction
            prediction_result = self.get_prediction(claim_data)
            
            if not prediction_result:
                return {
                    "likelihood_percent": 50.0,
                    "prediction": "error",
                    "denial_reasons": ["Error processing claim - unable to get prediction"],
                    "next_steps": ["Manual review required due to processing error"],
                    "confidence_percent": 0.0,
                    "analysis_details": {
                        "error": "No prediction available"
                    }
                }
            
            # Extract key information
            prediction = prediction_result['prediction']
            confidence = prediction_result['confidence']
            clinical_reasoning = prediction_result['clinical_reasoning']
            risk_factors = prediction_result['risk_factors']
            
            # Convert prediction format and set display fields
            if prediction == 'APPROVED':
                prediction_label = 'accepted'
                confidence_percent = round(confidence * 100, 1)
                likelihood_percent = None
            elif prediction == 'DENIED':
                prediction_label = 'denied'
                confidence_percent = None
                likelihood_percent = round(confidence * 100, 1)
            else:
                prediction_label = 'error'
                confidence_percent = None
                likelihood_percent = None
            
            # Generate denial reasons if prediction is denied
            denial_reasons = []
            accepted_reasons = []
            if prediction == 'DENIED':
                if clinical_reasoning:
                    denial_reasons.extend([reason for reason in clinical_reasoning if reason])
                if risk_factors:
                    denial_reasons.extend([f"Risk factor: {factor}" for factor in risk_factors if factor])
                # Add specific denial reasons based on clinical analysis
                medical_necessity = prediction_result.get('medical_necessity_score', 0.5)
                cost_appropriateness = prediction_result.get('cost_appropriateness_score', 0.5)
                if medical_necessity < 0.6:
                    denial_reasons.append("Low medical necessity score - procedure may not be clinically required")
                if cost_appropriateness < 0.6:
                    denial_reasons.append("Cost concerns - billed amount exceeds typical allowable limits")
                if not denial_reasons:
                    denial_reasons = ["Requires additional review based on claim characteristics"]
            elif prediction == 'APPROVED':
 
                # Add positive reasons based on valid codes and normal bill amount
                procedure_code = claim_data.get('procedure_code', '')
                diagnosis_code = claim_data.get('diagnosis_code', '')
                billed_amount = float(claim_data.get('billed_amount', 0))
                allowed_amount = float(claim_data.get('allowed_amount', 0))
                # Check for valid codes (simple check: non-empty and numeric for procedure, non-empty for diagnosis)
                if procedure_code and procedure_code.isdigit():
                    accepted_reasons.append(f"Procedure code {procedure_code} is valid and recognized.")
                if diagnosis_code and isinstance(diagnosis_code, str) and diagnosis_code.strip():
                    accepted_reasons.append(f"Diagnosis code {diagnosis_code} is valid and recognized.")
                # Check for normal billed amount (within 10% of allowed amount)
                if allowed_amount > 0:
                    ratio = billed_amount / allowed_amount
                    if 0.9 <= ratio <= 1.1:
                        accepted_reasons.append("Billed amount is within the typical range for this procedure.")
                if not accepted_reasons:
                    accepted_reasons = ["Claim meets standard criteria for approval."]
            
            # Generate next steps based on prediction and clinical analysis
            next_steps = []
            if prediction_label == 'accepted':
                base_steps = [
                    "Claim approved for processing",
                    "Verify beneficiary eligibility",
                    "Process payment according to contract terms"
                ]
                # Add procedure-specific steps
                procedure_type = self._classify_procedure_type(claim_data.get('procedure_code', ''))
                if procedure_type == "EMERGENCY_CARE":
                    base_steps.append("Expedite payment processing for emergency care")
                elif procedure_type == "PREVENTIVE_CARE":
                    base_steps.append("Update preventive care tracking metrics")
                next_steps = base_steps
            elif prediction_label == 'denied':
                base_steps = [
                    "Notify provider of denial decision",
                    "Send denial letter with specific reasons"
                ]
                # Add appeal-related steps based on appeal likelihood
                appeal_likelihood = prediction_result.get('appeal_likelihood', 'MODERATE')
                if appeal_likelihood in ['HIGH', 'MODERATE']:
                    base_steps.append("Prepare for potential appeal - gather supporting documentation")
                    base_steps.append("Review clinical guidelines and policy requirements")
                # Add patient harm risk considerations
                patient_harm_risk = prediction_result.get('patient_harm_risk', 'MINIMAL')
                if patient_harm_risk in ['SIGNIFICANT', 'MODERATE']:
                    base_steps.append("Schedule expedited clinical review")
                    base_steps.append("Consider alternative treatment options")
                base_steps.append("Provider may submit additional documentation for reconsideration")
                base_steps.append("Patient may appeal decision within 60 days")
                next_steps = base_steps
            
            # Clean up denial reasons
            denial_reasons = list(set([reason.strip() for reason in denial_reasons if reason and reason.strip()]))
            
            # Create the final JSON response
            json_response = {
                "prediction": prediction_label,
                "denial_reasons": denial_reasons if denial_reasons else None,
                "accepted_reasons": accepted_reasons if accepted_reasons else None,
                "next_steps": next_steps,
                "analysis_details": {
                    "procedure_code": claim_data.get('procedure_code', 'N/A'),
                    "diagnosis_code": claim_data.get('diagnosis_code', 'N/A'),
                    "billed_amount": claim_data.get('billed_amount', 0),
                    "insurance_type": claim_data.get('insurance_type', 'N/A')
                }
            }
            if confidence_percent is not None:
                json_response["confidence_percent"] = confidence_percent
            if likelihood_percent is not None:
                json_response["likelihood_percent"] = likelihood_percent
            return json_response
        except Exception as e:
            # Fallback response in case of errors
            return {
                "likelihood_percent": 50.0,
                "prediction": "error",
                "denial_reasons": [f"Error processing claim: {str(e)}"],
                "next_steps": ["Manual review required due to processing error"],
                "confidence_percent": 0.0,
                "analysis_details": {
                    "error": str(e)
                }
            }

def main():
    """Main function to process a single claim"""
    if len(sys.argv) < 2:
        print("Usage: python single_model_openai_denial_predictor.py '{\"procedure_code\": \"99213\", \"diagnosis_code\": \"I10\", ...}'")
        sys.exit(1)
        
    try:
        # Parse the claim data from command line argument
        claim_data = json.loads(sys.argv[1])
        
        # Initialize predictor
        predictor = SingleModelOpenAIDenialPredictor()
        
        # Get prediction
        json_result = predictor.get_denial_prediction_json(claim_data)
        
        # Output the result as JSON
        print(json.dumps(json_result))
        
    except json.JSONDecodeError:
        print("Error: Invalid JSON input")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()