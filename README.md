# Speed Dating Match Prediction Agent 

This repository contains a Jupyter Notebook project that builds an autonomous AI Agent to predict match probabilities from speed dating data. The agent is powered by **LangGraph**, **LangChain**, and **OpenAI**, and automates a complete Machine Learning pipeline using a ReAct architecture.

## Project Overview

Traditional Machine Learning pipelines require manual execution of steps like data cleaning, feature selection, and model training. This project flips that paradigm by introducing an **Agentic Workflow**.

Using LangGraph, we provide an AI agent with a set of "tools" and a high-level objective in natural language:
> *"Clean 'speeddating.csv', select top 10 features, and predict match probability."*

The agent autonomously reasons through the problem, deciding which tools to call, in what order, and how to pass state between them to arrive at a final evaluating Score (ROC AUC) for the trained XGBoost model.

## Key Features

1. **Exploratory Data Analysis (EDA)**: Initial programmatic look into the dataset shape, missing arrays, byte string prevalence, and target (match) distribution.
2. **Autonomous Data Cleaning Tool**: Automatically drops data-leakage columns (`decision`, `decision_o`), fixes byte-string encoding issues, and imputes missing numeric data. 
3. **Automated Feature Selection Tool**: Uses Recursive Feature Elimination (RFE) paired with a Random Forest Classifier to dynamically identify the top most predictive features.
4. **Machine Learning Model Tool**: Trains an `XGBoost` classifier to predict the probability of a match rather than a binary yes/no, evaluating reliability via the ROC AUC metric.
5. **LangGraph State Management**: Employs a cyclical graph architecture that allows the LLM to recursively think, act, and observe until the pipeline is complete.

## Technologies Used

* **[LangGraph](https://python.langchain.com/docs/langgraph/) / LangChain**: For building the stateful, multi-actor agent workflow.
* **[OpenAI API](https://openai.com/)**: `gpt-4o-mini` serves as the core reasoning engine.
* **Pandas & NumPy**: For data manipulation and EDA.
* **Scikit-Learn**: For preprocessing (LabelEncoder, SimpleImputer) and Feature Selection (RFE).
* **XGBoost**: Fast, gradient-boosted decision trees for the final predictive model.

## Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/Speed-Dating-Match-Prediction-Agent.git
   cd Speed-Dating-Match-Prediction-Agent
   ```

2. **Install the required packages**:
   You can install the dependencies used in the notebook:
   ```bash
   pip install langchain langchain-openai langgraph openai numpy pandas scikit-learn scipy xgboost tqdm joblib requests PyYAML
   ```

3. **Set your OpenAI API Key**:
   You must set your OpenAI API key in your environment variables for the agent to function.
   
   **Windows (PowerShell):**
   ```powershell
   $env:OPENAI_API_KEY="sk-your-api-key-here"
   ```
   **Mac/Linux:**
   ```bash
   export OPENAI_API_KEY="sk-your-api-key-here"
   ```
   *(Alternatively, you can place the API key directly into the notebook script where it initializes `ChatOpenAI`, but environment variables are recommended for security).*

### Running the Notebook

Open `Couple Matching Probability.ipynb` using Jupyter Notebook, JupyterLab, or VS Code, and run the cells sequentially from top to bottom. The notebook will automatically download the dataset from an IBM Cloud object storage bucket if it isn't present locally.

## Dataset Information

The data comes from the well-known [Speed Dating Dataset](https://www.kaggle.com/datasets/ulrikthygepedersen/speed-dating/data) by Ulrik Thyge Pedersen. It represents 4-minute speed dates where participants rated each other on various attributes (attractiveness, sincerity, intelligence, fun, ambition, shared interests).

## Future Extensions

This proof-of-concept demonstrates how single-instruction AI workflows can manage ML pipelines. The same structure can be adapted for:
* Hyperparameter tuning automation
* Multi-model benchmarking (letting the agent pick the best model)
* Full-scale automated feature engineering
* End-to-end report generation for stakeholders
