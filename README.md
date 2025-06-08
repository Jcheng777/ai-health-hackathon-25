# Healthcare Denial Prediction - AI Health Hackathon 2025

## 🏥 Project Overview

A comprehensive machine learning system for predicting healthcare claim denials, featuring three distinct approaches:
- **Traditional ML**: Random Forest baseline (96.0% accuracy)
- **OpenAI Enhanced**: LLM-integrated ensemble framework  
- **Reinforcement Learning**: Q-learning with business-cost optimization (82.0% accuracy)

## 📁 Project Structure

```
ai-health-hackathon-25/
├── models/                          # Core ML Models
│   ├── openai_denial_predictor.py   # Traditional ML + OpenAI framework
│   └── improved_rl_comparison.py    # Q-learning RL implementation
├── data/                            # Datasets
│   ├── enhanced_claim_data.csv      # 5,000 synthetic training claims
│   ├── test_claims.csv              # 100 test claims
│   └── claim_data.csv               # Original 1,000 claims
├── analysis/                        # Documentation & Reports
│   ├── COMPREHENSIVE_MODEL_ANALYSIS.md  # Complete analysis
│   ├── MODEL_PERFORMANCE_SUMMARY.md     # Traditional ML analysis
│   └── ML_INTEGRATION_README.md         # Setup guide
├── evaluation/                      # Testing & Evaluation
│   ├── evaluate_model_performance.py    # Comprehensive evaluation
│   ├── test_openai_integration.py       # OpenAI testing
│   └── generate_synthetic_data.py       # Data generation
├── visualizations/                  # Charts & Plots
│   ├── final_rl_model_comparison.png    # Multi-model comparison
│   └── model_performance_analysis.png   # Traditional ML analysis
├── setup/                          # Environment Setup
│   ├── requirements.txt            # Python dependencies
│   └── setup_ml_environment.sh     # Environment setup script
└── legacy/                         # Original Files
    ├── Denial_Prediction_Model.ipynb    # Original notebook
    ├── HealthHackathon (1).ipynb        # Original notebook
    └── Sample1500_DME.pdf               # Sample form
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Navigate to setup directory
cd setup/

# Run setup script
chmod +x setup_ml_environment.sh
./setup_ml_environment.sh

# Or manual setup
python -m venv ../ml_env
source ../ml_env/bin/activate  # On Windows: ..\ml_env\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Model Comparison
```bash
# Navigate to models directory
cd models/

# Run comprehensive comparison (all three models)
python improved_rl_comparison.py
```

### 3. Evaluate Individual Models
```bash
# Navigate to evaluation directory  
cd evaluation/

# Evaluate traditional ML performance
python evaluate_model_performance.py

# Test OpenAI integration (requires API key)
python test_openai_integration.py
```

## 📊 Model Performance Summary

| Model | Accuracy | Confidence | Key Features |
|-------|----------|------------|--------------|
| Traditional ML | 96.0% | 87.8% | High baseline, stable performance |
| OpenAI Enhanced | 96.0% | 87.8% | LLM-ready framework, clinical reasoning |
| RL Q-Learning | 82.0% | 86.4% | Adaptive learning, business-cost aware |

## 🔧 Key Features

### Traditional ML Model
- **Algorithm**: Random Forest with 8 engineered features
- **Performance**: 98.1% training, 96.0% test accuracy
- **Strengths**: Fast, interpretable, production-ready

### OpenAI Enhanced Model  
- **Architecture**: 60% Traditional ML + 40% LLM reasoning
- **Current**: Fallback mode (96.0% accuracy)
- **Potential**: Clinical reasoning with API integration

### Reinforcement Learning Model
- **Algorithm**: Q-learning with epsilon-greedy exploration
- **Innovation**: Business-cost optimized rewards
- **Learning**: 8 decision states, adaptive policies

## 🎯 Business Impact

### Cost-Aware Decision Making
- **False Approvals**: -25 points (financial loss)
- **False Denials**: -15 points (customer impact)  
- **Correct Decisions**: +10 points (operational efficiency)

### Real-World Applications
- **Traditional ML**: Immediate production deployment
- **RL Model**: Dynamic adaptation to changing patterns
- **OpenAI**: Complex case reasoning and explanations

## 📈 Usage Examples

### Traditional ML Prediction
```python
from models.openai_denial_predictor import OpenAIDenialPredictor
import pandas as pd

# Load model and data
predictor = OpenAIDenialPredictor()
df = pd.read_csv('data/enhanced_claim_data.csv')
predictor.train_traditional_model(df)

# Make prediction
claim = {
    'Procedure Code': '99213',
    'Diagnosis Code': 'M54.2', 
    'Insurance Type': 'Commercial',
    'Billed Amount': 350.0,
    'Allowed Amount': 280.0,
    'Paid Amount': 280.0
}

result = predictor.predict_with_ensemble(claim)
print(f"Prediction: {result['ensemble_prediction']}")
print(f"Confidence: {result['confidence']:.3f}")
```

### RL Model Training & Prediction
```python
from models.improved_rl_comparison import run_comprehensive_comparison

# Run complete comparison (trains all models)
results, models = run_comprehensive_comparison()

# Access trained RL model
rl_model = models['rl_improved']
prediction = rl_model.predict(claim)
print(f"RL Prediction: {prediction['prediction']}")
print(f"Q-values: {prediction['q_values']}")
```

## 🔍 Analysis & Documentation

- **[Comprehensive Analysis](analysis/COMPREHENSIVE_MODEL_ANALYSIS.md)**: Complete technical analysis
- **[Performance Summary](analysis/MODEL_PERFORMANCE_SUMMARY.md)**: Traditional ML deep dive  
- **[Integration Guide](analysis/ML_INTEGRATION_README.md)**: Setup and deployment guide

## 📊 Visualizations

- **Multi-model Comparison**: `visualizations/final_rl_model_comparison.png`
- **Traditional ML Analysis**: `visualizations/model_performance_analysis.png`

## 🛠️ Development

### Adding New Models
1. Create model file in `models/` directory
2. Follow existing interface patterns
3. Add evaluation in `evaluation/` directory
4. Update comparison scripts

### Data Requirements
- **Training**: 5,000+ claims with outcomes
- **Features**: Procedure codes, diagnosis codes, amounts, insurance types
- **Format**: CSV with standard healthcare claim fields

### Testing
```bash
cd evaluation/
python -m pytest test_*.py  # Run all tests
```

## 🔮 Future Enhancements

### Phase 1: Production Deployment
- [ ] API endpoint development
- [ ] Real-time prediction service
- [ ] Performance monitoring dashboard
- [ ] A/B testing framework

### Phase 2: Advanced RL
- [ ] Deep Q-Networks (DQN)
- [ ] Multi-agent systems  
- [ ] Online learning from feedback
- [ ] Policy gradient methods

### Phase 3: OpenAI Integration
- [ ] API key management
- [ ] Clinical reasoning explanations
- [ ] Fine-tuned medical models
- [ ] Regulatory compliance features

## 📋 Requirements

### Core Dependencies
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
openai>=1.0.0 (optional)
```

### System Requirements
- Python 3.8+
- 4GB+ RAM for training
- ~100MB disk space for data

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is part of the AI Health Hackathon 2025. See individual files for specific licensing.

## 📞 Support

For questions or issues:
- Create GitHub Issue
- Check documentation in `analysis/` directory
- Review code examples in `evaluation/` directory

---

**AI Health Hackathon 2025** - Advancing Healthcare Through Machine Learning