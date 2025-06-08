#!/bin/bash

# Healthcare Claim Denial Prediction - ML Environment Setup
echo "Setting up ML environment for Healthcare Claim Denial Prediction..."

# Create Python virtual environment if it doesn't exist
if [ ! -d "ml_env" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv ml_env
fi

# Activate virtual environment
echo "Activating virtual environment..."
source ml_env/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install required packages
echo "Installing required Python packages..."
pip install pandas==2.1.4
pip install numpy==1.24.4
pip install scikit-learn==1.3.2
pip install matplotlib==3.8.2
pip install seaborn==0.13.0

# Optional: Install additional packages for advanced features
echo "Installing optional packages..."
pip install joblib==1.3.2  # For model serialization
pip install tqdm==4.66.1   # For progress bars

# Create requirements.txt for future reference
echo "Creating requirements.txt..."
pip freeze > requirements.txt

# Create the scripts directory if it doesn't exist
echo "Creating scripts directory..."
mkdir -p scripts

# Make the prediction script executable
if [ -f "scripts/predict_denial.py" ]; then
    chmod +x scripts/predict_denial.py
    echo "Made predict_denial.py executable"
fi

echo "✅ ML environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source ml_env/bin/activate"
echo ""
echo "To test the prediction system, run:"
echo "cd scripts && python predict_denial.py '{\"procedureCode\":\"99213\",\"diagnosisCode\":\"A16.5\",\"insuranceType\":\"Medicare\",\"billedAmount\":348}'" 