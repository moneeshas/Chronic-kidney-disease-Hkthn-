"""
Created on Mon Apr 27 09:05:39 2020

@author: MANIDEEP
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from flask import Flask, render_template, url_for, request, jsonify
import numpy as np
import pandas as pd

# Load Dataset
dataset = pd.read_csv('kidney_disease.csv')
dataset.drop('id', axis=1, inplace=True)

# Fill Missing Values with Mode or Mean
categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
numerical_columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']

for col in categorical_columns:
    dataset[col] = dataset[col].fillna(dataset[col].mode().iloc[0])

for col in numerical_columns:
    dataset[col] = dataset[col].fillna(dataset[col].mean())

# Fix inconsistent values
dataset.wc = dataset.wc.replace("\t6200", 6200).replace("\t8400", 8400).replace("\t?", 9800)
dataset.pcv = dataset.pcv.replace("\t43", 43).replace("\t?", 41)
dataset.rc = dataset.rc.replace("\t?", 5.2)

# Convert Data Types
dataset.pcv = dataset.pcv.astype(int)
dataset.wc = dataset.wc.astype(int)
dataset.rc = dataset.rc.astype(float)

# Encode Categorical Features
dataset.classification = dataset.classification.replace('ckd\t', 'ckd')
dataset.classification = [1 if each == "ckd" else 0 for each in dataset.classification]
dataset.rbc = [1 if each == "abnormal" else 0 for each in dataset.rbc]
dataset.pc = [1 if each == "abnormal" else 0 for each in dataset.pc]
dataset.pcc = [1 if each == "present" else 0 for each in dataset.pcc]
dataset.ba = [1 if each == "present" else 0 for each in dataset.ba]
dataset.htn = [1 if each == "yes" else 0 for each in dataset.htn]
dataset.dm = [1 if each == "yes" else 0 for each in dataset.dm]
dataset.cad = [1 if each == "yes" else 0 for each in dataset.cad]
dataset.appet = [1 if each == "good" else 0 for each in dataset.appet]
dataset.pe = [1 if each == "yes" else 0 for each in dataset.pe]
dataset.ane = [1 if each == "yes" else 0 for each in dataset.ane]

# Split Features and Target
x = dataset.iloc[:, :-1]  # All columns except the last one (classification)
y = dataset.iloc[:, -1]   # Last column (classification)

# Split Data into Training and Testing Sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

# Train Decision Tree Model
dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(x_train, y_train)

# Flask App Setup
app = Flask(__name__, static_folder='./static/')


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/model', methods=['POST'])
def runModel():
    data = request.get_json()['fields']

    # Define feature column names
    feature_columns = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba',
                       'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                       'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    # Convert input data to DataFrame with correct feature names
    input_df = pd.DataFrame([data], columns=feature_columns)

    # Predict the result
    result = dt.predict(input_df)

    print(f"Prediction: {result[0]}")  # Debugging Output
    return jsonify({'prediction': int(result[0])})


if __name__ == "__main__":
    app.run(debug=True)
