ML Chat Workspace

ML Chat Workspace is an interactive Streamlit-based machine learning application that helps users go through a complete ML workflow step by step.
The app uses a chatbot-style command panel on the left and a browser-tab-style output workspace on the right.

Users can upload a CSV dataset, explore the data, understand encoding, choose the problem type, select target/features, preprocess data, train ML models, view results, and generate multiple visualizations.

Features
Upload CSV datasets
Step-by-step machine learning workflow
Command-based navigation
Automatic page unlocking through chatbot commands
Full dataset encoding for categorical columns
Separate encoding explanation page
Problem type selection:
Classification
Regression
Clustering
Target and feature selection
Preprocessing options:
StandardScaler
MinMaxScaler
PCA
Model training for multiple algorithms
Results page with evaluation plots
Manual prediction input section
Visualization page with multiple chart types
Browser-tab-style step navigation
Dataset reset when a new file is uploaded
Workflow Steps

The application is divided into the following steps:

Upload
Preview
Encoding
Problem Type
Target & Features
Preprocess
Train
Results
Visualization

Each step is unlocked through chatbot commands.

Supported Commands
upload
preview
encoding
problem type
target
preprocess
train
results
visualization
Technologies Used
Python
Streamlit
Pandas
NumPy
Matplotlib
Scikit-learn
Algorithms Included
Classification
Logistic Regression
Decision Tree Classifier
Random Forest Classifier
Bagging Classifier
AdaBoost Classifier
Gradient Boosting Classifier
SVM Classifier
KNN Classifier
Regression
Linear Regression
Decision Tree Regressor
Random Forest Regressor
Bagging Regressor
AdaBoost Regressor
Gradient Boosting Regressor
SVM Regressor
KNN Regressor
Clustering
K-Means
DBSCAN
Agglomerative Clustering
Encoding Concept

This project includes a dedicated Encoding page.

The encoding step converts categorical text columns into numeric values so that machine learning models can process the dataset effectively.

The app shows:

Original dataset
Fully encoded dataset
Encoding mappings for categorical columns
Visualization Features

The Visualization page allows the user to manually choose X and Y columns for different plots.

Included visualization types:

Line Chart
Bar Chart
Scatter Plot
Box Plot
Violin Plot
Area Chart
Installation

Clone the repository:

git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name

Install dependencies:

pip install -r requirements.txt

Run the Streamlit app:

streamlit run app.py
Project Structure
ML_Chat_Workspace/
│
├── app.py
├── requirements.txt
├── README.md
└── assets/
How to Use
Run the Streamlit application
Upload a CSV dataset
Use commands in the command panel
Unlock workflow steps one by one
Explore encoding and preprocessing options
Train a machine learning model
View results and visualizations
Example Use Cases
Student machine learning practice
Dataset exploration
Teaching ML workflow concepts
Preprocessing and encoding demonstration
Basic model training and result analysis
Future Improvements
Add more visualization types
Add deep learning models
Add downloadable reports
Add model comparison page
Add automatic feature importance analysis
Add export option for encoded datasets
Author

Anbarasu

License

This project is for educational and learning purposes.
