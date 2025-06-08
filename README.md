# 🏥 Healthcare AI - Claim Denial Prediction

**AI Health Hackathon 2025** | Advanced Clinical AI for Healthcare Claims

## 🚀 What This Does

Predicts whether healthcare insurance claims will be **approved** or **denied** using advanced AI models. Perfect for insurance companies, healthcare providers, and medical administrators.

## 🎯 Models & Performance

| Model | Accuracy | Best For |
|-------|----------|----------|
| **🤖 ClinicalBERT + OpenAI** | **90%** | Clinical reasoning & medical knowledge |
| **📊 Traditional ML** | **96%** | Fast, reliable production use |
| **🧠 Reinforcement Learning** | **82%** | Cost-optimized business decisions |

## ⚡ Quick Start

### 1. Install
```bash
pip install -r setup/requirements.txt
```

### 2. Run Clinical AI Demo
```bash
cd models
python clinical_ai_demo.py
```

### 3. See Visualizations
```bash
python create_visualizations.py
```

## 🏥 Clinical AI Features

Our **ClinicalBERT + OpenAI** model provides:

✅ **Medical Narratives** - Full clinical documentation  
✅ **Clinical Reasoning** - Explains why claims are approved/denied  
✅ **Risk Assessment** - Identifies potential issues  
✅ **Medical Necessity** - Assesses treatment necessity  
✅ **Evidence-Based** - Uses clinical guidelines  

### Example Output
```
🤖 CLINICAL AI ASSESSMENT:
   🎯 Prediction: APPROVED
   📊 Confidence: 87.5%
   🏥 Medical Necessity: Medium

🧠 CLINICAL REASONING:
   1. Emergency/urgent care indication
   2. Evidence-based clinical guidelines met
   3. Standard procedure for diagnosis
```

## 📁 Key Files

```
📂 models/
  └── clinical_bert_openai_predictor.py    # 🎯 Main Clinical AI
  └── clinical_ai_demo.py                  # 🎮 Interactive demo

📂 visualizations/
  └── clinical_ai_performance_comparison.png    # 📊 Performance charts
  └── medical_analysis_dashboard.png            # 🏥 Medical insights

📂 data/
  └── enhanced_claim_data.csv             # 📋 5,000 training claims
  └── test_claims.csv                     # 🧪 100 test claims
```

## 🔧 Usage

### Predict Single Claim
```python
from models.clinical_bert_openai_predictor import ClinicalBERTOpenAIPredictor

# Initialize AI
clinical_ai = ClinicalBERTOpenAIPredictor()

# Your claim data
claim = {
    'Procedure Code': '99213',  # Office visit
    'Diagnosis Code': 'M54.2',  # Back pain
    'Insurance Type': 'Commercial',
    'Billed Amount': 350.0
}

# Get AI prediction
result = clinical_ai.predict_with_clinical_ai(claim)

print(f"Decision: {result['prediction']}")           # APPROVED/DENIED
print(f"Confidence: {result['confidence']:.1%}")     # 87.5%
print(f"Reasoning: {result['clinical_reasoning']}")  # Why this decision
```

## 📊 What You Get

### 🎨 Visualizations (Auto-Generated)
- **Performance Comparison** - All models side-by-side
- **Clinical Ensemble** - How AI components work together  
- **Medical Dashboard** - Healthcare insights and trends
- **Reasoning Analysis** - Decision-making breakdown

### 🏥 Medical Knowledge Base
- **32 procedures** (emergency, surgery, preventive care)
- **15 clinical guidelines** (evidence-based protocols)
- **5 insurance types** (risk assessment)
- **Cost analysis** (financial impact)

## 🎯 Real-World Use Cases

**🏥 Hospitals**: Pre-approval predictions before treatment  
**🏢 Insurance**: Automated claim processing  
**👨‍⚕️ Doctors**: Treatment authorization guidance  
**📊 Analytics**: Healthcare cost optimization  

## 🛠️ Advanced Features

### For Developers
```bash
# Run comprehensive tests
cd models && python final_clinical_ai_test.py

# Generate all visualizations  
python create_visualizations.py

# Compare all models
python comprehensive_ai_comparison.py
```

### For Data Scientists
- **Feature Engineering**: 8 optimized features
- **Ensemble Learning**: Multi-modal AI architecture
- **Clinical Knowledge**: MIMIC-III medical database
- **Evaluation Metrics**: Accuracy, F1-score, clinical relevance

## 🚀 Next Steps

1. **🎮 Try the Demo**: `python clinical_ai_demo.py`
2. **📊 See Charts**: Check `visualizations/` folder
3. **🔧 Customize**: Modify models for your specific needs
4. **🚀 Deploy**: Production-ready API integration

## 📞 Need Help?

- 📖 **Full docs**: Check `analysis/` folder
- 🐛 **Issues**: Create GitHub issue  
- 💡 **Ideas**: Open discussion

---