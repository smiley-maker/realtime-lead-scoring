# Real-Time Lead Scoring Pipeline

> *Connecting homebuyers with the right agents using ML-powered streaming analytics*

This project implements a Real-Time Lead Prioritization System using a Spark Structured Streaming pipeline to immediately score new user activity with 90% test accuracy. It utilizes a SparkML Random Forest Classifier trained on historical data to predict the probability of lead conversion (contacting an agent), mimicking a critical system for Zillow's Connections AI team. 



## Setup
```bash
# Create environment
conda create -n lead-scoring python=3.10 -y
conda activate lead-scoring

# Install dependencies
pip install -r requirements.txt

# If running into java errors:
conda install -c conda-forge openjdk=17
```