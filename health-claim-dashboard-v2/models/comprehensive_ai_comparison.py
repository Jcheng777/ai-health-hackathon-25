#!/usr/bin/env python3
"""
Comprehensive AI Model Comparison for Healthcare Claim Denial Prediction

Compares all developed models:
1. Traditional ML (Random Forest baseline)
2. OpenAI Enhanced (ML + GPT-4)
3. Reinforcement Learning (Q-Learning)
4. ClinicalBERT (MIMIC-III trained)
5. ClinicalBERT + OpenAI Hybrid (Advanced Clinical AI)

Provides detailed analysis, performance metrics, and visualizations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import time
from typing import Dict, List, Any
import os
import sys

# Add models directory to path
sys.path.append('.')

# Import all models
try:
    from openai_denial_predictor import OpenAIEnhancedPredictor
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from improved_rl_comparison import ImprovedQLearningPredictor
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False

try:
    from clinical_bert_predictor import ClinicalBERTPredictor
    CLINICAL_BERT_AVAILABLE = True
except ImportError:
    CLINICAL_BERT_AVAILABLE = False

try:
    from clinical_bert_openai_predictor import ClinicalBERTOpenAIPredictor
    CLINICAL_BERT_OPENAI_AVAILABLE = True
except ImportError:
    CLINICAL_BERT_OPENAI_AVAILABLE = False

warnings.filterwarnings('ignore')

class ComprehensiveAIComparison:
    """Comprehensive comparison of all AI models for healthcare claim prediction"""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.training_data = None
        self.test_data = None
        
    def load_data(self):
        """Load training and test data"""
        try:
            self.training_data = pd.read_csv('../data/enhanced_claim_data.csv')
            self.test_data = pd.read_csv('../data/test_claims.csv')
            print(f"✅ Data loaded: {len(self.training_data)} training, {len(self.test_data)} test claims")
            return True
        except FileNotFoundError:
            print("❌ Data files not found. Please ensure data files are in ../data/ directory")
            return False
    
    def initialize_models(self):
        """Initialize all available AI models"""
        print("🚀 Initializing AI Models...")
        
        # Traditional ML + OpenAI Enhanced
        if OPENAI_AVAILABLE:
            try:
                self.models['OpenAI Enhanced'] = OpenAIEnhancedPredictor()
                print("✅ OpenAI Enhanced Model initialized")
            except Exception as e:
                print(f"⚠️ OpenAI Enhanced Model failed: {e}")
        
        # Reinforcement Learning
        if RL_AVAILABLE:
            try:
                self.models['Reinforcement Learning'] = ImprovedQLearningPredictor()
                print("✅ Reinforcement Learning Model initialized")
            except Exception as e:
                print(f"⚠️ RL Model failed: {e}")
        
        # ClinicalBERT
        if CLINICAL_BERT_AVAILABLE:
            try:
                self.models['ClinicalBERT'] = ClinicalBERTPredictor()
                print("✅ ClinicalBERT Model initialized")
            except Exception as e:
                print(f"⚠️ ClinicalBERT Model failed: {e}")
        
        # ClinicalBERT + OpenAI Hybrid
        if CLINICAL_BERT_OPENAI_AVAILABLE:
            try:
                self.models['ClinicalBERT + OpenAI'] = ClinicalBERTOpenAIPredictor()
                print("✅ ClinicalBERT + OpenAI Hybrid Model initialized")
            except Exception as e:
                print(f"⚠️ ClinicalBERT + OpenAI Model failed: {e}")
        
        print(f"📊 Total models initialized: {len(self.models)}")
        return len(self.models) > 0
    
    def train_all_models(self):
        """Train/calibrate all models"""
        print("\n🎓 Training All AI Models...")
        
        for model_name, model in self.models.items():
            print(f"\n--- Training {model_name} ---")
            try:
                if hasattr(model, 'train_on_claims_data'):
                    model.train_on_claims_data(self.training_data)
                elif hasattr(model, 'train'):
                    model.train(self.training_data)
                else:
                    print(f"⚠️ No training method found for {model_name}")
                
                print(f"✅ {model_name} training completed")
                
            except Exception as e:
                print(f"❌ {model_name} training failed: {e}")
    
    def evaluate_all_models(self, num_test_samples: int = 25):
        """Evaluate all models on test data"""
        print(f"\n🔍 Evaluating All Models (Test Sample: {num_test_samples})...")
        
        test_sample = self.test_data.sample(n=min(num_test_samples, len(self.test_data)), 
                                          random_state=42)
        
        for model_name, model in self.models.items():
            print(f"\n--- Evaluating {model_name} ---")
            start_time = time.time()
            
            try:
                # Get model predictions
                predictions = []
                confidences = []
                additional_metrics = []
                
                for _, claim in test_sample.iterrows():
                    claim_dict = claim.to_dict()
                    
                    # Get prediction based on model type
                    if hasattr(model, 'predict_with_clinical_ai'):
                        # ClinicalBERT + OpenAI Hybrid
                        result = model.predict_with_clinical_ai(claim_dict)
                        prediction = result['prediction']
                        confidence = result['confidence']
                        additional_metrics.append({
                            'medical_necessity': result.get('medical_necessity', 'Medium'),
                            'clinical_reasoning': result.get('clinical_reasoning', []),
                            'ai_models_used': result.get('ai_models_used', [])
                        })
                    
                    elif hasattr(model, 'predict_with_openai'):
                        # OpenAI Enhanced
                        result = model.predict_with_openai(claim_dict)
                        prediction = result['prediction']
                        confidence = result['confidence']
                        additional_metrics.append({
                            'ml_confidence': result.get('ml_confidence', 0.5),
                            'openai_confidence': result.get('openai_confidence', 0.5),
                            'reasoning': result.get('reasoning', '')
                        })
                    
                    elif hasattr(model, 'predict_with_clinical_bert'):
                        # ClinicalBERT
                        result = model.predict_with_clinical_bert(claim_dict)
                        prediction = result['prediction']
                        confidence = result['confidence']
                        additional_metrics.append({
                            'medical_concepts': result.get('medical_concepts', []),
                            'clinical_reasoning': result.get('clinical_reasoning', '')
                        })
                    
                    elif hasattr(model, 'predict_claim'):
                        # Reinforcement Learning
                        result = model.predict_claim(claim_dict)
                        prediction = result['prediction']
                        confidence = result['confidence']
                        additional_metrics.append({
                            'state': result.get('state', 'unknown'),
                            'q_value': result.get('q_value', 0.0)
                        })
                    
                    else:
                        # Generic prediction method
                        result = model.predict(claim_dict)
                        if isinstance(result, dict):
                            prediction = result.get('prediction', 'APPROVED')
                            confidence = result.get('confidence', 0.5)
                        else:
                            prediction = 'APPROVED' if result > 0.5 else 'DENIED'
                            confidence = result if result > 0.5 else 1.0 - result
                        additional_metrics.append({})
                    
                    predictions.append(prediction)
                    confidences.append(confidence)
                
                # Calculate metrics
                actual_outcomes = ['DENIED' if outcome == 'Denied' else 'APPROVED' 
                                 for outcome in test_sample['Outcome']]
                
                correct_predictions = sum(p == a for p, a in zip(predictions, actual_outcomes))
                accuracy = correct_predictions / len(predictions)
                avg_confidence = np.mean(confidences)
                
                # Calculate confusion matrix
                tp = sum(1 for p, a in zip(predictions, actual_outcomes) if p == 'APPROVED' and a == 'APPROVED')
                tn = sum(1 for p, a in zip(predictions, actual_outcomes) if p == 'DENIED' and a == 'DENIED')
                fp = sum(1 for p, a in zip(predictions, actual_outcomes) if p == 'APPROVED' and a == 'DENIED')
                fn = sum(1 for p, a in zip(predictions, actual_outcomes) if p == 'DENIED' and a == 'APPROVED')
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                evaluation_time = time.time() - start_time
                
                # Store results
                self.results[model_name] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'correct_predictions': correct_predictions,
                    'total_predictions': len(predictions),
                    'evaluation_time': evaluation_time,
                    'predictions': predictions,
                    'confidences': confidences,
                    'additional_metrics': additional_metrics,
                    'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
                }
                
                print(f"✅ {model_name} Evaluation Complete:")
                print(f"   Accuracy: {accuracy:.3f}")
                print(f"   Confidence: {avg_confidence:.3f}")
                print(f"   F1-Score: {f1_score:.3f}")
                print(f"   Time: {evaluation_time:.2f}s")
                
            except Exception as e:
                print(f"❌ {model_name} evaluation failed: {e}")
                self.results[model_name] = {
                    'accuracy': 0.0,
                    'error': str(e)
                }
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        print("\n" + "="*80)
        print("🏥 COMPREHENSIVE AI MODEL COMPARISON REPORT")
        print("Healthcare Claim Denial Prediction Analysis")
        print("="*80)
        
        if not self.results:
            print("❌ No results available. Please run evaluation first.")
            return
        
        # Performance Summary Table
        print("\n📊 PERFORMANCE SUMMARY")
        print("-" * 80)
        print(f"{'Model':<25} {'Accuracy':<10} {'Confidence':<12} {'F1-Score':<10} {'Time (s)':<10}")
        print("-" * 80)
        
        sorted_models = sorted(self.results.items(), 
                             key=lambda x: x[1].get('accuracy', 0), reverse=True)
        
        for model_name, metrics in sorted_models:
            if 'error' not in metrics:
                print(f"{model_name:<25} {metrics['accuracy']:<10.3f} "
                      f"{metrics['avg_confidence']:<12.3f} {metrics['f1_score']:<10.3f} "
                      f"{metrics['evaluation_time']:<10.2f}")
            else:
                print(f"{model_name:<25} {'ERROR':<10} {'N/A':<12} {'N/A':<10} {'N/A':<10}")
        
        # Best performing model
        if sorted_models and 'error' not in sorted_models[0][1]:
            best_model, best_metrics = sorted_models[0]
            print(f"\n🏆 BEST PERFORMING MODEL: {best_model}")
            print(f"   Accuracy: {best_metrics['accuracy']:.3f}")
            print(f"   Confidence: {best_metrics['avg_confidence']:.3f}")
            print(f"   F1-Score: {best_metrics['f1_score']:.3f}")
        
        # Model-specific insights
        print(f"\n🔍 MODEL-SPECIFIC INSIGHTS")
        print("-" * 50)
        
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                print(f"\n{model_name}:")
                
                # Confusion matrix
                cm = metrics['confusion_matrix']
                print(f"   True Positives: {cm['tp']}, True Negatives: {cm['tn']}")
                print(f"   False Positives: {cm['fp']}, False Negatives: {cm['fn']}")
                
                # Additional model-specific metrics
                if model_name == 'ClinicalBERT + OpenAI':
                    medical_necessities = [m.get('medical_necessity', 'Medium') 
                                         for m in metrics['additional_metrics']]
                    necessity_dist = {n: medical_necessities.count(n) for n in ['High', 'Medium', 'Low']}
                    print(f"   Medical Necessity Distribution: {necessity_dist}")
                
                elif model_name == 'OpenAI Enhanced':
                    avg_ml_conf = np.mean([m.get('ml_confidence', 0.5) 
                                         for m in metrics['additional_metrics']])
                    avg_openai_conf = np.mean([m.get('openai_confidence', 0.5) 
                                             for m in metrics['additional_metrics']])
                    print(f"   Avg ML Confidence: {avg_ml_conf:.3f}")
                    print(f"   Avg OpenAI Confidence: {avg_openai_conf:.3f}")
    
    def create_comparison_visualizations(self):
        """Create comparison visualizations"""
        print("\n📈 Generating Comparison Visualizations...")
        
        if not self.results:
            print("❌ No results to visualize")
            return
        
        # Prepare data
        models = []
        accuracies = []
        confidences = []
        f1_scores = []
        
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                models.append(model_name)
                accuracies.append(metrics['accuracy'])
                confidences.append(metrics['avg_confidence'])
                f1_scores.append(metrics['f1_score'])
        
        if not models:
            print("❌ No valid results to visualize")
            return
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comprehensive AI Model Comparison\nHealthcare Claim Denial Prediction', 
                    fontsize=16, fontweight='bold')
        
        # 1. Accuracy Comparison
        bars1 = ax1.bar(models, accuracies, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9370DB'])
        ax1.set_title('Model Accuracy Comparison', fontweight='bold')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 2. Confidence Comparison
        bars2 = ax2.bar(models, confidences, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9370DB'])
        ax2.set_title('Model Confidence Comparison', fontweight='bold')
        ax2.set_ylabel('Average Confidence')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, conf in zip(bars2, confidences):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{conf:.3f}', ha='center', va='bottom')
        
        # 3. F1-Score Comparison
        bars3 = ax3.bar(models, f1_scores, color=['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9370DB'])
        ax3.set_title('Model F1-Score Comparison', fontweight='bold')
        ax3.set_ylabel('F1-Score')
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, f1 in zip(bars3, f1_scores):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom')
        
        # 4. Radar Chart for Overall Performance
        categories = ['Accuracy', 'Confidence', 'F1-Score']
        
        # Normalize all metrics to 0-1 scale for radar chart
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        ax4 = plt.subplot(2, 2, 4, projection='polar')
        
        colors = ['#2E8B57', '#4169E1', '#DC143C', '#FF8C00', '#9370DB']
        
        for i, model in enumerate(models[:3]):  # Limit to top 3 models for clarity
            metrics_values = [accuracies[i], confidences[i], f1_scores[i]]
            metrics_values += metrics_values[:1]  # Complete the circle
            
            ax4.plot(angles, metrics_values, 'o-', linewidth=2, 
                    label=model, color=colors[i])
            ax4.fill(angles, metrics_values, alpha=0.25, color=colors[i])
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(categories)
        ax4.set_ylim(0, 1)
        ax4.set_title('Top 3 Models - Overall Performance', fontweight='bold', pad=20)
        ax4.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        
        plt.tight_layout()
        
        # Save visualization
        plt.savefig('../visualizations/comprehensive_ai_comparison.png', 
                   dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: ../visualizations/comprehensive_ai_comparison.png")
        
        plt.show()
    
    def export_detailed_results(self):
        """Export detailed results to CSV"""
        print("\n💾 Exporting Detailed Results...")
        
        if not self.results:
            print("❌ No results to export")
            return
        
        # Create summary DataFrame
        summary_data = []
        for model_name, metrics in self.results.items():
            if 'error' not in metrics:
                summary_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'],
                    'Average_Confidence': metrics['avg_confidence'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1_Score': metrics['f1_score'],
                    'Evaluation_Time_Seconds': metrics['evaluation_time'],
                    'Correct_Predictions': metrics['correct_predictions'],
                    'Total_Predictions': metrics['total_predictions']
                })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv('../analysis/comprehensive_ai_results.csv', index=False)
        print("✅ Results exported: ../analysis/comprehensive_ai_results.csv")
    
    def run_comprehensive_comparison(self, num_test_samples: int = 25):
        """Run complete comprehensive comparison"""
        print("🚀 STARTING COMPREHENSIVE AI MODEL COMPARISON")
        print("=" * 65)
        
        # Load data
        if not self.load_data():
            return False
        
        # Initialize models
        if not self.initialize_models():
            print("❌ No models available for comparison")
            return False
        
        # Train models
        self.train_all_models()
        
        # Evaluate models
        self.evaluate_all_models(num_test_samples)
        
        # Generate report
        self.generate_comparison_report()
        
        # Create visualizations
        self.create_comparison_visualizations()
        
        # Export results
        self.export_detailed_results()
        
        print(f"\n🎉 Comprehensive AI Comparison Complete!")
        print(f"🏥 All models evaluated on {num_test_samples} test claims")
        print(f"📊 Results include {len(self.results)} AI models")
        
        return True

def main():
    """Run comprehensive AI model comparison"""
    comparison = ComprehensiveAIComparison()
    
    # Run full comparison with 25 test samples
    success = comparison.run_comprehensive_comparison(num_test_samples=25)
    
    if success:
        print("\n🌟 Healthcare AI Model Comparison Successfully Completed!")
        print("📈 Check visualizations/ and analysis/ directories for detailed results")
    else:
        print("\n❌ Comparison failed. Please check error messages above.")

if __name__ == "__main__":
    main() 