🧠 Breast Cancer Classification using ANN (TensorFlow)
📌 Project Description
Binary classification of breast cancer tumors using a regularized Artificial Neural Network (ANN) built with TensorFlow/Keras.
This project focuses on building a well-generalized neural network, applying regularization techniques, and evaluating performance using professional classification metrics.

📊 Dataset
Name: Breast Cancer Wisconsin Dataset
Source: UCI Machine Learning Repository (loaded via sklearn)
Samples: 569
Features: 30 numerical features
Target Classes:
0 → Benign

1 → Malignant

🧠 Model Architecture
Input Layer (30 features)
↓
Dense (128 units, ReLU)
↓
Dropout (0.3)
↓
Dense (64 units, ReLU)
↓
Dropout (0.3)
↓
Dense (1 unit, Sigmoid)

Design Choices
ReLU activation for non-linearity
Dropout layers to reduce overfitting
Sigmoid activation for binary classification

⚙️ Training Configuration
Framework: TensorFlow / Keras
Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy
Batch Size: 16
Epochs: Up to 100
Early Stopping
Early stopping is used to monitor validation loss and automatically stop training when no improvement is observed.
This helps:
Prevent overfitting
Select the best-performing model
Reduce unnecessary computation

📈 Evaluation Metrics
Model performance is evaluated using:
Accuracy
Precision, Recall, F1-score
Confusion Matrix
ROC-AUC Score
These metrics provide a more reliable evaluation than accuracy alone, especially for medical classification tasks.

📊 Results
High classification accuracy (typically above 95%)
Strong ROC-AUC score
Stable training and validation loss curves
Note: Results may slightly vary due to random initialization and train-test splits.

🛠 Technologies Used
Python
TensorFlow / Keras
NumPy
Scikit-learn
Matplotlib

🎯Key Learning Outcomes
Implementing an ANN from scratch using TensorFlow
Understanding binary classification with neural networks
Applying regularization techniques (Dropout)
Using Early Stopping to improve generalization
Evaluating classification models with professional metrics

📌Author
This project was developed as a learning-focused yet professional implementation of Artificial Neural Networks, following real-world machine learning practices.
