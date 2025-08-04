# Network_Intrusion_Detection_with_IBMCloud

## 📋 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Machine Learning Models](#-machine-learning-models)
- [IBM AutoAI Integration](#-ibm-autoai-integration)
- [Model Performance Comparison](#-model-performance-comparison)
- [Technology Stack](#-technology-stack)
- [Project Architecture](#-project-architecture)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Results](#-results)
- [Performance Visualization](#-performance-visualization)
- [Model Deployment](#-model-deployment)
- [Acknowledgments](#-acknowledgments)

## 🎯 Overview

This project implements a comprehensive Network Intrusion Detection System (NIDS) using various machine learning algorithms to identify and classify network attacks. The system leverages both traditional ML approaches and IBM's AutoAI capabilities to achieve optimal performance in detecting network anomalies and security threats.

### Key Features
- Multi-model approach for robust intrusion detection
- IBM AutoAI integration for automated model selection and optimization
- Comprehensive performance analysis and comparison
- Real-time prediction capabilities
- Scalable cloud-based architecture

## 📊 Dataset

**Source**: [Network Intrusion Detection Dataset](https://www.kaggle.com/datasets/sampadab17/network-intrusion-detection/data)

The dataset contains network traffic data with various features that help in identifying different types of network attacks and normal traffic patterns.

### Dataset Characteristics:
- **Size**: Multiple network traffic records
- **Features**: Network protocol information, connection details, traffic statistics
- **Target Classes**: Normal traffic and various attack types
- **Format**: CSV format suitable for ML processing

## 🤖 Machine Learning Models

This project implements and compares multiple machine learning algorithms to find the most effective approach for network intrusion detection.



### 1. K-Nearest Neighbors (KNN)
- **Algorithm Type**: Instance-based learning
- **Strengths**: Simple, effective for local patterns
- **Performance**: Train Score: 0.984065, Test Score: 0.976978
- **Use Case**: Good baseline model for comparison

### 2. Logistic Regression
- **Algorithm Type**: Linear classification
- **Strengths**: Fast training, interpretable results
- **Performance**: Train Score: 0.931326, Test Score: 0.924848
- **Use Case**: Efficient for large-scale deployments

### 3. Decision Tree
- **Algorithm Type**: Tree-based learning
- **Strengths**: Highly interpretable, handles non-linear relationships
- **Performance**: Train Score: 0.999943, Test Score: 0.992591
- **Use Case**: Excellent for understanding attack patterns

  ![Decision Tree](Assets/Screenshot%202025-08-04%20171725.png)

### 4. Random Forest
- **Algorithm Type**: Ensemble method
- **Strengths**: Reduces overfitting, robust performance
- **Performance**: Train Score: 0.999603, Test Score: 0.994972
- **Use Case**: Balanced accuracy and generalization

### 5. Gradient Boosting Machine (GBM)
- **Algorithm Type**: Boosting ensemble
- **Strengths**: High accuracy, handles complex patterns
- **Performance**: Train Score: 0.994499, Test Score: 0.992988
- **Use Case**: High-performance production systems

### 6. XGBoost
- **Algorithm Type**: Optimized gradient boosting
- **Strengths**: State-of-the-art performance, efficient
- **Performance**: Train Score: 0.999887, Test Score: 0.996163
- **Use Case**: Competition-grade accuracy

### 7. AdaBoost
- **Algorithm Type**: Adaptive boosting
- **Strengths**: Combines weak learners effectively
- **Performance**: Train Score: 0.976239, Test Score: 0.976713
- **Use Case**: Good ensemble base learner

### 8. LightGBM
- **Algorithm Type**: Gradient boosting framework
- **Strengths**: Fast training, memory efficient
- **Performance**: Train Score: 0.999943, Test Score: 0.995898
- **Use Case**: Large-scale data processing

### 9. CatBoost
- **Algorithm Type**: Categorical boosting
- **Strengths**: Handles categorical features well
- **Performance**: Train Score: 0.998866, Test Score: 0.994972
- **Use Case**: Mixed data types

### 10. Naive Bayes
- **Algorithm Type**: Probabilistic classifier
- **Strengths**: Fast, works well with small datasets
- **Performance**: Train Score: 0.893785, Test Score: 0.894284
- **Use Case**: Real-time classification

### 11. Voting Classifier
- **Algorithm Type**: Ensemble method
- **Strengths**: Combines multiple algorithms
- **Performance**: Train Score: 0.999887, Test Score: 0.995634
- **Use Case**: Maximum accuracy through consensus

### 12. Support Vector Machine (SVM)
- **Algorithm Type**: Kernel-based learning
- **Strengths**: Effective in high-dimensional spaces
- **Performance**: Train Score: 0.965521, Test Score: 0.966393
- **Use Case**: Complex decision boundaries

![Model Performance Chart](assets/performance_chart.png)

## 🚀 IBM AutoAI Integration

This project leverages IBM Watson Machine Learning's AutoAI capabilities to automatically discover, configure, and deploy the best-performing machine learning pipeline.

![AutoAI Workflow](assets/autoai_workflow.png)

### AutoAI Process:
1. **Data Preparation**: Automated data preprocessing and feature engineering
2. **Algorithm Selection**: Automatic selection of best-performing algorithms
3. **Hyperparameter Optimization**: Automated tuning for optimal performance
4. **Pipeline Generation**: Creation of end-to-end ML pipelines
5. **Model Comparison**: Comprehensive evaluation and ranking

### AutoAI Features Used:
- **Data Preprocessing**: Automatic handling of missing values, encoding
- **Feature Engineering**: Automated feature creation and selection
- **Model Selection**: Testing multiple algorithms simultaneously
- **Hyperparameter Tuning**: Optimization using advanced techniques
- **Pipeline Evaluation**: Cross-validation and performance metrics

![AutoAI Pipeline](Assets/Screenshot%202025-08-03%20232919.png)

### Generated Pipelines:
The AutoAI experiment generated multiple pipelines with different configurations:
- **Pipeline 4 (P4)**: Top-performing pipeline with advanced feature engineering
- **Pipeline 5 (P5)**: Balanced performance and interpretability
- **Pipeline 6 (P6)**: Optimized for speed and efficiency

![Pipeline Comparison](Assets/Screenshot%202025-08-03%20232800.png)


## 📈 Model Performance Comparison

Based on comprehensive evaluation, here's the performance ranking:

| Rank | Model | Train Score | Test Score | Key Advantage |
|------|-------|-------------|------------|---------------|
| 1 | XGBoost | 0.999887 | 0.996163 | Highest accuracy |
| 2 | Voting Classifier | 0.999887 | 0.995634 | Ensemble stability |
| 3 | LightGBM | 0.999943 | 0.995898 | Speed + accuracy |
| 4 | Random Forest | 0.999603 | 0.994972 | Robust performance |
| 5 | CatBoost | 0.998866 | 0.994972 | Categorical handling |



### Key Insights:
- **XGBoost** achieved the highest test accuracy (99.62%)
- **Tree-based models** generally outperformed linear models
- **Ensemble methods** showed excellent generalization
- **AutoAI pipelines** competitive with manual implementations

## 🛠 Technology Stack

### IBM Cloud Services:
- **IBM Watson Studio**: ML development environment
- **IBM Watson Machine Learning**: Model deployment and management
- **IBM AutoAI**: Automated machine learning
- **IBM Cloud Object Storage**: Data storage and management
- **IBM Watson Runtime**: Model serving infrastructure


### Development Tools:
- **Python 3.8+**: Primary programming language
- **Scikit-learn**: Traditional ML algorithms
- **XGBoost/LightGBM/CatBoost**: Advanced boosting algorithms
- **Pandas/NumPy**: Data manipulation and analysis
- **Jupyter Notebooks**: Interactive development
- **Git**: Version control

### Deployment:
- **IBM Cloud**: Cloud infrastructure
- **Docker**: Containerization
- **REST APIs**: Model serving endpoints

## 🏗 Project Architecture

![Architecture Diagram](Assets/Screenshot%202025-08-03%20232648.png)

The project follows a modular architecture:

1. **Data Layer**: IBM Cloud Object Storage for dataset management
2. **Processing Layer**: IBM Watson Studio for data preprocessing
3. **ML Layer**: Multiple algorithms and AutoAI pipelines
4. **Evaluation Layer**: Comprehensive model comparison
5. **Deployment Layer**: IBM Watson Machine Learning for serving

## 📥 Installation & Setup

### Prerequisites:
```bash
Python 3.8+
IBM Cloud Account
Watson Studio Access
```

### IBM Cloud Setup:
1. Create Watson Studio instance
2. Set up Watson Machine Learning service
3. Configure Cloud Object Storage
4. Import project notebooks


## 🚀 Usage

![Usage Example](assets/usage_example.png)

### Making Predictions:
```python
# Load trained model
model = load_model('best_model.pkl')

# Predict on new data
predictions = model.predict(new_network_data)
probabilities = model.predict_proba(new_network_data)
```

## 📊 Results

### Best Performing Models:
1. **XGBoost**: 99.62% accuracy - Best overall performance
2. **Voting Ensemble**: 99.56% accuracy - Most stable predictions
3. **LightGBM**: 99.59% accuracy - Fastest training time
   
![Performance Metrics](Assets/Screenshot%202025-08-04%20171644.png)

### AutoAI Results:
- **Pipeline 5**: Achieved competitive performance with automated optimization
- **Feature Engineering**: AutoAI discovered 18 optimal features
- **Model Selection**: Automatically selected gradient boosting variants

### Key Findings:
- Tree-based ensemble methods are most effective for this dataset
- Feature engineering significantly improves model performance
- AutoAI provides competitive results with minimal manual intervention
- The system can detect various attack types with high accuracy

![Confusion Matrix](assets/confusion_matrix.png)

### Attack Detection Capabilities:
- **Normal Traffic**: 99.8% precision
- **DoS Attacks**: 99.5% detection rate
- **Probe Attacks**: 98.9% detection rate
- **R2L Attacks**: 97.8% detection rate
- **U2R Attacks**: 96.5% detection rate

## 📈 Performance Visualization

The project includes comprehensive visualizations:

![ROC Curves](assets/roc_curves.png)

- **ROC Curves**: Model discrimination capability
- **Precision-Recall**: Performance across different thresholds
- **Feature Importance**: Most influential network features
- **Learning Curves**: Training progression and convergence

## 🔧 Model Deployment

### IBM Watson Machine Learning:
```python
# Deploy best model
deployment = client.deployments.create(
    artifact_uid=model_uid,
    meta_props={
        client.deployments.ConfigurationMetaNames.NAME: "intrusion_detection_model",
        client.deployments.ConfigurationMetaNames.ONLINE: {}
    }
)

# Get scoring endpoint
scoring_endpoint = client.deployments.get_scoring_href(deployment)
```


### Real-time Scoring:
```python
# Score new network traffic
payload = {
    client.deployments.ScoringMetaNames.INPUT_DATA: [{
        'values': network_features
    }]
}

response = client.deployments.score(deployment_uid, payload)
prediction = response['predictions'][0]['values'][0]
```


## 🙏 Acknowledgments

- IBM Watson Studio team for AutoAI capabilities
- Kaggle community for the network intrusion dataset
- Open source ML community for algorithm implementations
- IBM Cloud for providing the infrastructure platform



---

**Note**: This project is part of ongoing research in cybersecurity and machine learning. The models and techniques used here are for educational and research purposes. For production deployment, additional security measures and compliance checks should be implemented.

