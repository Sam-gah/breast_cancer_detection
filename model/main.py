# import library
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle as pickle
import numpy as np

df = pd.read_csv(r"D:\Minor_python\data.csv")

def get_clean_data():
  data = pd.read_csv("D:\Minor_python\data.csv")
  
  data = data.drop(['Unnamed: 32', 'id'], axis=1)
  
  data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
  
  return data

def train_models(data):
    X = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']
 
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
  
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
  
    # train the models
    lr_model = LogisticRegression()
    lr_model.fit(X_train, y_train)
  
    svm_model = SVC()
    svm_model.fit(X_train, y_train)
  
    knn_model = KNeighborsClassifier()
    knn_model.fit(X_train, y_train)
  
    rf_model = RandomForestClassifier()
    rf_model.fit(X_train, y_train)
  
    # test the models
    lr_y_pred = lr_model.predict(X_test)
    svm_y_pred = svm_model.predict(X_test)
    knn_y_pred = knn_model.predict(X_test)
    rf_y_pred = rf_model.predict(X_test)
  
    print('Accuracy of Logistic Regression model: ', accuracy_score(y_test, lr_y_pred))
    print("Classification report for Logistic Regression model: \n", classification_report(y_test, lr_y_pred))
  
    print('Accuracy of SVM model: ', accuracy_score(y_test, svm_y_pred))
    print("Classification report for SVM model: \n", classification_report(y_test, svm_y_pred))
  
    print('Accuracy of KNN model: ', accuracy_score(y_test, knn_y_pred))
    print("Classification report for KNN model: \n", classification_report(y_test, knn_y_pred))
  
    print('Accuracy of Random Forest model: ', accuracy_score(y_test, rf_y_pred))
    print("Classification report for Random Forest model: \n", classification_report(y_test, rf_y_pred))
  
    return lr_model, svm_model, knn_model, rf_model, scaler

def ensemble_models(models, scaler, data, weights=None):
    X_test = scaler.transform(data.drop(['diagnosis'], axis=1))
    y_test = data['diagnosis']
  
    # predict using each model
    predictions = []
    for model in models:
        predictions.append(model.predict(X_test))
  
    # apply weights if provided
    if weights is not None:
        predictions = [p * w for p, w in zip(predictions, weights)]
  
    # calculate weighted average
    weighted_average = np.average(predictions, axis=0, weights=weights)
  
    # calculate accuracy and classification report
    accuracy = accuracy_score(y_test, weighted_average.round())
    report = classification_report(y_test, weighted_average.round())
  
    print('Accuracy of ensemble model: ', accuracy)
    print("Classification report for ensemble model: \n", report)
   
    return weighted_average

def main():
    data = get_clean_data(),ensemble_models
    

    models, scaler = train_models(data)
   


    with open('model/lr_model.pkl', 'wb') as f:
        pickle.dump(models[0], f)
    
    with open('model/svm_model.pkl', 'wb') as f:
        pickle.dump(models[1], f)
    
    with open('model/knn_model.pkl', 'wb') as f:
        pickle.dump(models[2], f)
    
    with open('model/rf_model.pkl', 'wb') as f:
        pickle.dump(models[3], f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    

if __name__ == '__main__':
    main()
