#!/bin/bash
cd "$(dirname "$0")"
export PATH="$PATH:/Users/krish_01/Library/Python/3.9/bin"
echo "Starting FraudShield at http://localhost:8501 ..."
streamlit run app.py --server.port 8501
