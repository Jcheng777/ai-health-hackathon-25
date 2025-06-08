#!/usr/bin/env python3
"""
Enhanced Synthetic Healthcare Claims Data Generator

Generates realistic healthcare claims data with various scenarios
for training denial prediction models.
"""

import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
import string

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

class HealthcareDataGenerator:
    def __init__(self):
        # Define realistic healthcare data patterns
        self.procedure_codes = {
            # Office visits
            '99213': {'name': 'Office visit - established patient', 'avg_cost': 150, 'denial_rate': 0.15},
            '99214': {'name': 'Office visit - detailed exam', 'avg_cost': 200, 'denial_rate': 0.20},
            '99215': {'name': 'Office visit - comprehensive', 'avg_cost': 300, 'denial_rate': 0.25},
            
            # Hospital visits
            '99221': {'name': 'Initial hospital care', 'avg_cost': 400, 'denial_rate': 0.30},
            '99222': {'name': 'Initial hospital care - detailed', 'avg_cost': 500, 'denial_rate': 0.35},
            '99223': {'name': 'Initial hospital care - comprehensive', 'avg_cost': 600, 'denial_rate': 0.40},
            
            # Subsequent hospital care
            '99231': {'name': 'Subsequent hospital care', 'avg_cost': 300, 'denial_rate': 0.20},
            '99232': {'name': 'Subsequent hospital care - detailed', 'avg_cost': 400, 'denial_rate': 0.25},
            '99233': {'name': 'Subsequent hospital care - comprehensive', 'avg_cost': 500, 'denial_rate': 0.30},
            
            # Discharge
            '99238': {'name': 'Hospital discharge', 'avg_cost': 250, 'denial_rate': 0.25},
            
            # Emergency
            '99281': {'name': 'Emergency dept visit - straightforward', 'avg_cost': 200, 'denial_rate': 0.10},
            '99282': {'name': 'Emergency dept visit - low complexity', 'avg_cost': 300, 'denial_rate': 0.15},
            '99283': {'name': 'Emergency dept visit - moderate complexity', 'avg_cost': 450, 'denial_rate': 0.20},
            '99284': {'name': 'Emergency dept visit - high complexity', 'avg_cost': 600, 'denial_rate': 0.25},
            '99285': {'name': 'Emergency dept visit - critical', 'avg_cost': 800, 'denial_rate': 0.30},
        }
        
        self.diagnosis_codes = {
            # Common conditions with varying denial rates
            'A00.1': {'name': 'Typhoid fever', 'denial_rate': 0.15},
            'A01.9': {'name': 'Paratyphoid fever', 'denial_rate': 0.18},
            'A02.1': {'name': 'Salmonella sepsis', 'denial_rate': 0.22},
            'A03.3': {'name': 'Shigellosis', 'denial_rate': 0.16},
            'A04.0': {'name': 'Enteropathogenic E. coli infection', 'denial_rate': 0.14},
            'A04.1': {'name': 'Enterotoxigenic E. coli infection', 'denial_rate': 0.12},
            'A04.2': {'name': 'Enteroinvasive E. coli infection', 'denial_rate': 0.13},
            'A05.0': {'name': 'Foodborne staphylococcal intoxication', 'denial_rate': 0.10},
            'A06.0': {'name': 'Acute amebic dysentery', 'denial_rate': 0.25},
            'A07.8': {'name': 'Other specified protozoal intestinal diseases', 'denial_rate': 0.28},
            'A08.1': {'name': 'Acute gastroenteropathy due to Norwalk agent', 'denial_rate': 0.11},
            'A09.9': {'name': 'Gastroenteritis, unspecified', 'denial_rate': 0.08},
            
            # Respiratory conditions
            'J44.1': {'name': 'COPD with acute exacerbation', 'denial_rate': 0.20},
            'J45.9': {'name': 'Asthma, unspecified', 'denial_rate': 0.12},
            'J18.9': {'name': 'Pneumonia, unspecified organism', 'denial_rate': 0.18},
            
            # Cardiovascular
            'I10': {'name': 'Essential hypertension', 'denial_rate': 0.08},
            'I25.10': {'name': 'Atherosclerotic heart disease', 'denial_rate': 0.15},
            'I50.9': {'name': 'Heart failure, unspecified', 'denial_rate': 0.22},
            
            # Diabetes
            'E11.9': {'name': 'Type 2 diabetes without complications', 'denial_rate': 0.10},
            'E11.65': {'name': 'Type 2 diabetes with hyperglycemia', 'denial_rate': 0.18},
            
            # Mental health (higher denial rates)
            'F32.9': {'name': 'Major depressive disorder', 'denial_rate': 0.35},
            'F41.1': {'name': 'Generalized anxiety disorder', 'denial_rate': 0.32},
            'F43.10': {'name': 'Post-traumatic stress disorder', 'denial_rate': 0.40},
        }
        
        self.insurance_types = {
            'Medicare': {'denial_rate': 0.12, 'weight': 0.30},
            'Medicaid': {'denial_rate': 0.18, 'weight': 0.25},
            'Commercial': {'denial_rate': 0.15, 'weight': 0.35},
            'Self-Pay': {'denial_rate': 0.45, 'weight': 0.10}
        }
        
        self.reason_codes = [
            'Authorization not obtained',
            'Pre-existing condition',
            'Lack of medical necessity',
            'Duplicate claim',
            'Incorrect billing information',
            'Patient eligibility issues',
            'Service not covered',
            'Missing documentation',
            'Experimental treatment',
            'Out of network provider'
        ]
        
        self.ar_statuses = ['Open', 'Closed', 'Pending', 'On Hold', 'Denied', 'Partially Paid']
        self.outcomes = ['Paid', 'Denied', 'Partially Paid']
        
    def generate_realistic_amount(self, procedure_code, insurance_type):
        """Generate realistic billing amounts based on procedure and insurance"""
        base_cost = self.procedure_codes[procedure_code]['avg_cost']
        
        # Add random variation (±30%)
        variation = np.random.uniform(0.7, 1.3)
        billed_amount = int(base_cost * variation)
        
        # Insurance-specific allowed amounts
        if insurance_type == 'Medicare':
            allowed_rate = np.random.uniform(0.6, 0.8)  # Medicare typically pays less
        elif insurance_type == 'Medicaid':
            allowed_rate = np.random.uniform(0.55, 0.75)  # Medicaid pays least
        elif insurance_type == 'Commercial':
            allowed_rate = np.random.uniform(0.75, 0.95)  # Commercial pays more
        else:  # Self-Pay
            allowed_rate = np.random.uniform(0.4, 0.7)  # Self-pay often negotiated down
            
        allowed_amount = int(billed_amount * allowed_rate)
        
        # Paid amount (if claim is approved)
        paid_rate = np.random.uniform(0.85, 1.0)  # Usually pay most of allowed
        paid_amount = int(allowed_amount * paid_rate)
        
        return billed_amount, allowed_amount, paid_amount
    
    def determine_claim_outcome(self, procedure_code, diagnosis_code, insurance_type, billed_amount):
        """Determine claim outcome based on multiple factors"""
        
        # Base denial rates
        proc_denial_rate = self.procedure_codes[procedure_code]['denial_rate']
        diag_denial_rate = self.diagnosis_codes[diagnosis_code]['denial_rate']
        ins_denial_rate = self.insurance_types[insurance_type]['denial_rate']
        
        # Combined risk score
        risk_score = (proc_denial_rate + diag_denial_rate + ins_denial_rate) / 3
        
        # Adjust for high amounts (>$500 more likely to be denied)
        if billed_amount > 500:
            risk_score *= 1.3
        elif billed_amount > 1000:
            risk_score *= 1.5
            
        # Add some randomness
        risk_score += np.random.uniform(-0.1, 0.1)
        risk_score = np.clip(risk_score, 0, 1)
        
        # Determine outcome
        rand = np.random.random()
        if rand < risk_score * 0.7:  # Denied
            return 'Denied'
        elif rand < risk_score:  # Under Review (uncertain cases)
            return 'Under Review'
        else:  # Paid
            return 'Paid'
    
    def generate_claim_id(self):
        """Generate realistic claim ID"""
        return ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    
    def generate_provider_id(self):
        """Generate realistic provider ID"""
        return ''.join(random.choices(string.digits, k=10))
    
    def generate_patient_id(self):
        """Generate realistic patient ID"""
        return ''.join(random.choices(string.digits, k=10))
    
    def generate_date_of_service(self, start_date, end_date):
        """Generate random date within range"""
        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)
        return start_date + timedelta(days=random_days)
    
    def generate_dataset(self, num_claims=5000):
        """Generate comprehensive synthetic dataset"""
        
        print(f"Generating {num_claims} synthetic healthcare claims...")
        
        data = []
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 12, 31)
        
        for i in range(num_claims):
            if i % 1000 == 0:
                print(f"Generated {i} claims...")
                
            # Select procedure and diagnosis based on realistic distributions
            procedure_code = np.random.choice(list(self.procedure_codes.keys()))
            diagnosis_code = np.random.choice(list(self.diagnosis_codes.keys()))
            
            # Select insurance type based on weights
            insurance_weights = [self.insurance_types[ins]['weight'] for ins in self.insurance_types.keys()]
            insurance_type = np.random.choice(list(self.insurance_types.keys()), p=insurance_weights)
            
            # Generate amounts
            billed_amount, allowed_amount, paid_amount = self.generate_realistic_amount(
                procedure_code, insurance_type
            )
            
            # Determine claim outcome
            claim_status = self.determine_claim_outcome(
                procedure_code, diagnosis_code, insurance_type, billed_amount
            )
            
            # Adjust paid amount based on outcome
            if claim_status == 'Denied':
                paid_amount = 0
                outcome = 'Denied'
            elif claim_status == 'Under Review':
                # Under review claims might be partially paid
                if np.random.random() < 0.3:
                    paid_amount = int(paid_amount * np.random.uniform(0.3, 0.8))
                    outcome = 'Partially Paid'
                else:
                    outcome = np.random.choice(['Paid', 'Denied', 'Partially Paid'])
            else:  # Paid
                outcome = np.random.choice(['Paid', 'Partially Paid'], p=[0.8, 0.2])
                if outcome == 'Partially Paid':
                    paid_amount = int(paid_amount * np.random.uniform(0.6, 0.9))
            
            # Generate other fields
            claim_data = {
                'Claim ID': self.generate_claim_id(),
                'Provider ID': self.generate_provider_id(),
                'Patient ID': self.generate_patient_id(),
                'Date of Service': self.generate_date_of_service(start_date, end_date).strftime('%m/%d/%Y'),
                'Billed Amount': billed_amount,
                'Procedure Code': procedure_code,
                'Diagnosis Code': diagnosis_code,
                'Allowed Amount': allowed_amount,
                'Paid Amount': paid_amount,
                'Insurance Type': insurance_type,
                'Claim Status': claim_status,
                'Reason Code': np.random.choice(self.reason_codes),
                'Follow-up Required': np.random.choice(['Yes', 'No'], p=[0.3, 0.7]),
                'AR Status': np.random.choice(self.ar_statuses),
                'Outcome': outcome
            }
            
            data.append(claim_data)
        
        df = pd.DataFrame(data)
        
        # Add some data quality insights
        print(f"\n📊 Dataset Summary:")
        print(f"Total claims: {len(df)}")
        print(f"\nClaim Status Distribution:")
        print(df['Claim Status'].value_counts())
        print(f"\nOutcome Distribution:")
        print(df['Outcome'].value_counts())
        print(f"\nInsurance Type Distribution:")
        print(df['Insurance Type'].value_counts())
        print(f"\nAverage Billed Amount: ${df['Billed Amount'].mean():.2f}")
        print(f"Denial Rate: {(df['Outcome'] == 'Denied').mean():.2%}")
        
        return df

def main():
    """Generate and save synthetic healthcare claims data"""
    generator = HealthcareDataGenerator()
    
    # Generate dataset
    df = generator.generate_dataset(num_claims=5000)  # Generate 5000 claims for robust training
    
    # Save to CSV
    output_file = 'enhanced_claim_data.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✅ Dataset saved to {output_file}")
    
    # Create a smaller test set
    test_df = df.sample(n=100, random_state=42)
    test_df.to_csv('test_claims.csv', index=False)
    print(f"✅ Test set (100 claims) saved to test_claims.csv")
    
    return df

if __name__ == "__main__":
    df = main() 