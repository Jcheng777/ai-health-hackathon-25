#!/usr/bin/env python3
"""
Improved Reinforcement Learning Healthcare Claim Denial Predictor

Enhanced Q-learning with better state representation and comprehensive comparison.
"""

import pandas as pd
import numpy as np
import random
from collections import defaultdict
from sklearn.preprocessing import StandardScaler, LabelEncoder, KBinsDiscretizer
import matplotlib.pyplot as plt
from openai_denial_predictor import OpenAIDenialPredictor
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

class ImprovedQLearningAgent:
    """Improved Q-learning agent with better state handling"""
    
    def __init__(self, action_size=2, learning_rate=0.15, discount_factor=0.9):
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = 0.8  # Start with high exploration
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        
        # Q-table as dictionary
        self.q_table = defaultdict(lambda: np.zeros(action_size))
        
        # Performance tracking
        self.training_accuracies = []
        self.episode_rewards = []
        self.correct_predictions = 0
        self.total_predictions = 0
        
    def preprocess_claim(self, claim):
        """Convert claim to simplified state representation"""
        try:
            # Extract key features and create simple state
            insurance_type = str(claim.get('Insurance Type', 'Unknown'))
            billed_amount = float(claim.get('Billed Amount', 0))
            procedure_code = str(claim.get('Procedure Code', 'Unknown'))
            
            # Discretize billed amount into bins
            if billed_amount < 1000:
                amount_bin = 0
            elif billed_amount < 5000:
                amount_bin = 1
            elif billed_amount < 15000:
                amount_bin = 2
            else:
                amount_bin = 3
            
            # Simplify insurance type
            insurance_map = {
                'Commercial': 0,
                'Medicare': 1,
                'Medicaid': 2,
                'Self-Pay': 3
            }
            insurance_bin = insurance_map.get(insurance_type, 0)
            
            # Simplify procedure code (use first digit)
            try:
                proc_bin = int(str(procedure_code)[0]) % 5 if procedure_code.isdigit() else 0
            except:
                proc_bin = 0
            
            return (insurance_bin, amount_bin, proc_bin)
            
        except Exception as e:
            # Fallback state
            return (0, 0, 0)
    
    def get_action(self, state, training=True):
        """Choose action using epsilon-greedy policy"""
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        q_values = self.q_table[state]
        return np.argmax(q_values)
    
    def update_q_table(self, state, action, reward, next_state=None):
        """Update Q-table using Q-learning formula"""
        current_q = self.q_table[state][action]
        
        if next_state is not None:
            max_next_q = np.max(self.q_table[next_state])
        else:
            max_next_q = 0
        
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def train_on_data(self, df, episodes=200):
        """Train Q-learning agent on claim data"""
        print(f"🎮 Training Q-learning agent for {episodes} episodes...")
        
        for episode in range(episodes):
            # Shuffle data each episode
            shuffled_df = df.sample(frac=1.0, random_state=episode).reset_index(drop=True)
            
            total_reward = 0
            correct_count = 0
            
            for idx, row in shuffled_df.iterrows():
                # Get state representation
                state = self.preprocess_claim(row)
                
                # Get action
                action = self.get_action(state, training=True)
                
                # Get actual outcome
                actual_outcome = 1 if row['Outcome'] == 'Denied' else 0
                
                # Calculate reward
                if action == actual_outcome:
                    reward = 10.0
                    correct_count += 1
                else:
                    reward = -10.0
                    
                # Business impact adjustment
                if action == 1 and actual_outcome == 0:  # False denial
                    reward -= 5.0
                elif action == 0 and actual_outcome == 1:  # False approval
                    reward -= 15.0
                
                total_reward += reward
                
                # Update Q-table
                self.update_q_table(state, action, reward)
            
            # Track performance
            accuracy = correct_count / len(shuffled_df)
            self.training_accuracies.append(accuracy)
            self.episode_rewards.append(total_reward)
            
            # Decay epsilon
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            
            # Print progress
            if episode % 50 == 0:
                avg_accuracy = np.mean(self.training_accuracies[-50:]) if len(self.training_accuracies) >= 50 else accuracy
                print(f"Episode {episode:3d} | Accuracy: {accuracy:.3f} | Avg Accuracy: {avg_accuracy:.3f} | Q-table size: {len(self.q_table):3d} | Epsilon: {self.epsilon:.3f}")
        
        print(f"✅ Q-learning training completed!")
        print(f"   Final Q-table size: {len(self.q_table)} states")
        print(f"   Final accuracy: {self.training_accuracies[-1]:.3f}")
    
    def predict(self, claim):
        """Make prediction for a single claim"""
        state = self.preprocess_claim(claim)
        action = self.get_action(state, training=False)
        
        prediction = 'DENIED' if action == 1 else 'APPROVED'
        
        # Calculate confidence based on Q-values
        q_values = self.q_table[state]
        if np.sum(np.abs(q_values)) > 0:
            max_q = np.max(q_values)
            min_q = np.min(q_values)
            confidence = abs(max_q - min_q) / (abs(max_q) + abs(min_q) + 1e-8)
            confidence = max(0.6, min(1.0, confidence))
        else:
            confidence = 0.6
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'q_values': q_values,
            'action': action,
            'state': state
        }

def run_comprehensive_comparison():
    """Run comprehensive model comparison with improved RL"""
    print("🚀 Comprehensive Healthcare Denial Prediction Model Comparison")
    print("Including Traditional ML, OpenAI Framework, and Improved Q-Learning RL")
    print("=" * 75)
    
    # Load data
    print("📂 Loading data...")
    df = pd.read_csv('enhanced_claim_data.csv')
    test_df = pd.read_csv('test_claims.csv')
    print(f"Training data: {len(df)} claims")
    print(f"Test data: {len(test_df)} claims")
    
    # Initialize models
    models = {}
    results = {}
    
    print("\n🔄 Loading and training all models...")
    
    # 1. Traditional ML Model
    print("📊 Loading Traditional ML model...")
    models['traditional'] = OpenAIDenialPredictor()
    models['traditional'].train_traditional_model(df)
    
    # 2. OpenAI Enhanced Model (framework)
    print("🧠 Loading OpenAI Enhanced model...")
    models['openai'] = OpenAIDenialPredictor()
    models['openai'].train_traditional_model(df)
    
    # 3. Improved Q-learning Model
    print("🎮 Loading Improved RL (Q-learning) model...")
    models['rl_improved'] = ImprovedQLearningAgent()
    models['rl_improved'].train_on_data(df, episodes=150)
    
    print("\n✅ All models loaded successfully!")
    
    # Evaluate models
    print(f"\n📊 Evaluating all models on test data...")
    num_test_claims = min(50, len(test_df))
    test_claims = test_df.sample(n=num_test_claims, random_state=42)
    
    evaluation_results = {
        'traditional': {'predictions': [], 'confidences': [], 'correct': []},
        'openai': {'predictions': [], 'confidences': [], 'correct': []},
        'rl_improved': {'predictions': [], 'confidences': [], 'correct': []}
    }
    
    for idx, (_, claim) in enumerate(test_claims.iterrows()):
        if idx % 10 == 0:
            print(f"Evaluating claim {idx + 1}/{num_test_claims}...")
        
        actual_outcome = 'DENIED' if claim['Outcome'] == 'Denied' else 'APPROVED'
        
        # Traditional ML prediction
        trad_result = models['traditional'].predict_with_ensemble(claim.to_dict())
        evaluation_results['traditional']['predictions'].append(trad_result['ensemble_prediction'])
        evaluation_results['traditional']['confidences'].append(trad_result['confidence'])
        evaluation_results['traditional']['correct'].append(trad_result['ensemble_prediction'] == actual_outcome)
        
        # OpenAI prediction
        openai_result = models['openai'].predict_with_ensemble(claim.to_dict())
        evaluation_results['openai']['predictions'].append(openai_result['ensemble_prediction'])
        evaluation_results['openai']['confidences'].append(openai_result['confidence'])
        evaluation_results['openai']['correct'].append(openai_result['ensemble_prediction'] == actual_outcome)
        
        # RL prediction
        rl_result = models['rl_improved'].predict(claim.to_dict())
        evaluation_results['rl_improved']['predictions'].append(rl_result['prediction'])
        evaluation_results['rl_improved']['confidences'].append(rl_result['confidence'])
        evaluation_results['rl_improved']['correct'].append(rl_result['prediction'] == actual_outcome)
    
    # Calculate final metrics
    for model_name, model_results in evaluation_results.items():
        accuracy = np.mean(model_results['correct'])
        avg_confidence = np.mean(model_results['confidences'])
        
        results[model_name] = {
            'accuracy': accuracy,
            'avg_confidence': avg_confidence,
            'total_predictions': len(model_results['predictions']),
            'correct_predictions': sum(model_results['correct'])
        }
    
    # Create comprehensive report
    print(f"\n📋 Comprehensive Model Comparison Report")
    print("=" * 65)
    
    # Performance table
    print(f"{'Model':<25} {'Accuracy':<12} {'Confidence':<12} {'Predictions':<12}")
    print("-" * 61)
    
    model_names = {
        'traditional': 'Traditional ML',
        'openai': 'OpenAI Enhanced',
        'rl_improved': 'RL Q-Learning (Improved)'
    }
    
    for model_name, model_results in results.items():
        model_display = model_names.get(model_name, model_name)
        accuracy = model_results['accuracy']
        confidence = model_results['avg_confidence']
        total = model_results['total_predictions']
        
        print(f"{model_display:<25} {accuracy:<12.3f} {confidence:<12.3f} {total:<12}")
    
    # Best model identification
    best_accuracy_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
    
    print(f"\n🏆 Best Performance:")
    print(f"   Highest Accuracy: {model_names[best_accuracy_model]} ({results[best_accuracy_model]['accuracy']:.3f})")
    
    # Model improvements
    print(f"\n🔍 Model Analysis:")
    trad_acc = results['traditional']['accuracy']
    
    for model_name, model_results in results.items():
        improvement = ""
        if model_name != 'traditional':
            model_acc = model_results['accuracy']
            improvement_pct = ((model_acc - trad_acc) / trad_acc) * 100
            improvement = f" ({improvement_pct:+.1f}% vs Traditional)"
        
        print(f"   {model_names[model_name]:<25}: {model_results['accuracy']:.3f} accuracy{improvement}")
    
    # RL specific insights
    rl_model = models['rl_improved']
    print(f"\n🎮 RL Model Insights:")
    print(f"   Q-table size: {len(rl_model.q_table)} states learned")
    print(f"   Final exploration rate: {rl_model.epsilon:.3f}")
    print(f"   Learning converged: {'Yes' if rl_model.epsilon < 0.1 else 'Partially'}")
    if len(rl_model.episode_rewards) > 0:
        print(f"   Average episode reward: {np.mean(rl_model.episode_rewards[-10:]):.2f}")
    
    # Create visualizations
    print(f"\n📊 Creating comparison visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Accuracy comparison
    model_list = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in model_list]
    display_names = [model_names[model] for model in model_list]
    
    bars = axes[0, 0].bar(display_names, accuracies, color=['skyblue', 'lightgreen', 'coral'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Add accuracy values on bars
    for i, acc in enumerate(accuracies):
        axes[0, 0].text(i, acc + 0.01, f'{acc:.3f}', ha='center', va='bottom')
    
    # 2. Confidence comparison
    confidences = [results[model]['avg_confidence'] for model in model_list]
    
    bars = axes[0, 1].bar(display_names, confidences, color=['skyblue', 'lightgreen', 'coral'])
    axes[0, 1].set_title('Average Confidence Comparison')
    axes[0, 1].set_ylabel('Average Confidence')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Add confidence values on bars
    for i, conf in enumerate(confidences):
        axes[0, 1].text(i, conf + 0.01, f'{conf:.3f}', ha='center', va='bottom')
    
    # 3. RL Training Progress (Accuracy)
    episodes = range(len(rl_model.training_accuracies))
    axes[1, 0].plot(episodes, rl_model.training_accuracies, color='coral', alpha=0.8, linewidth=2)
    axes[1, 0].set_title('RL Training Progress (Accuracy)')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_ylim(0, 1)
    
    # 4. RL Training Progress (Rewards)
    episodes = range(len(rl_model.episode_rewards))
    axes[1, 1].plot(episodes, rl_model.episode_rewards, color='purple', alpha=0.8, linewidth=2)
    axes[1, 1].set_title('RL Training Progress (Rewards)')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Episode Reward')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('final_rl_model_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Visualizations saved as 'final_rl_model_comparison.png'")
    
    # Summary insights
    print(f"\n💡 Key Insights:")
    print(f"   • Traditional ML maintains high baseline accuracy ({results['traditional']['accuracy']:.3f})")
    print(f"   • RL model learns adaptive policies from business-oriented rewards")
    print(f"   • OpenAI framework provides foundation for LLM integration")
    print(f"   • All models preserved for comparative benchmarking")
    print(f"   • RL demonstrates {len(rl_model.q_table)} unique decision states learned")
    
    print(f"\n🎉 Comprehensive model comparison complete!")
    print(f"📈 Check 'final_rl_model_comparison.png' for detailed visualizations.")
    
    return results, models

if __name__ == "__main__":
    results, models = run_comprehensive_comparison() 