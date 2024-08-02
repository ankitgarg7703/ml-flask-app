import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, send_file
import io

# Load the Iris dataset
def load_data():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = pd.read_csv(url, header=None, names=column_names)
    return df

# Preprocess the data
def preprocess_data(df):
    X = df.drop("class", axis=1)
    y = df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

# Exploratory Data Analysis
def eda(df):
    sns.pairplot(df, hue="class")
    plt.savefig("eda.png")
    plt.close()

# Train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    with open("evaluation.txt", "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Classification Report:\n{report}\n")
    return accuracy, report

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction[0]})

@app.route('/eda', methods=['GET'])
def get_eda():
    return send_file("eda.png", mimetype='image/png')

@app.route('/evaluation', methods=['GET'])
def get_evaluation():
    with open("evaluation.txt", "r") as f:
        evaluation_results = f.read()
    return jsonify({'evaluation': evaluation_results})

if __name__ == '__main__':
    df = load_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    eda(df)
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    
    app.run(debug=True)
