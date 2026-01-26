#!/bin/bash

# BERT Sentiment Analysis Service Startup Script

set -e

echo "Starting BERT Sentiment Analysis Service..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "Error: pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set environment variables
export BERT_MODEL=${BERT_MODEL:-"nlptown/bert-base-multilingual-uncased-sentiment"}
export PORT=${PORT:-8080}
export DEBUG=${DEBUG:-false}

echo "Model: $BERT_MODEL"
echo "Port: $PORT"
echo "Debug: $DEBUG"

# Check if model directory exists and download if needed
echo "Checking model availability..."
python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification
print('Downloading/checking model: $BERT_MODEL')
tokenizer = AutoTokenizer.from_pretrained('$BERT_MODEL')
model = AutoModelForSequenceClassification.from_pretrained('$BERT_MODEL')
print('Model ready!')
"

# Start the service
echo "Starting BERT sentiment analysis service on port $PORT..."
if [ "$DEBUG" = "true" ]; then
    python3 bert_sentiment_service.py
else
    gunicorn -w 2 -b 0.0.0.0:$PORT --timeout 120 bert_sentiment_service:app
fi