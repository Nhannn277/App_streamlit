import os
import io
import pickle
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.tree import plot_tree # type: ignore
from streamlit_option_menu import option_menu # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore
from sklearn.linear_model import LinearRegression, LogisticRegression # type: ignore
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder, LabelEncoder # type: ignore

def trainModelInterface(df, ml_algorithm, dependent_var, independent_vars):
        
    if ml_algorithm == "Linear Regression":  
        trainModel(ml_algorithm, df, independent_vars, dependent_var, model_selected= LinearRegression())

    elif ml_algorithm == "Logistic Regression":
        trainModel(ml_algorithm, df, independent_vars, dependent_var, model_selected= LogisticRegression())
    
    elif ml_algorithm == "KNN":
        trainModel(ml_algorithm, df, independent_vars, dependent_var, model_selected= KNeighborsClassifier())
        
    elif ml_algorithm == "Decision Tree":
        trainModel(ml_algorithm, df, independent_vars, dependent_var, model_selected= DecisionTreeClassifier())
    
    


def trainModel(ml_algorithm, df, independent_vars, dependent_var, model_selected):
    
    X = df[independent_vars]
    y = df[dependent_var]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = model_selected
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    resultForecast(y_test, y_pred)
    
    if ml_algorithm == 'Linear Regression':
        create_chart_scatterplot(y_test, y_pred)
    elif ml_algorithm == 'Logistic Regression':
        create_chart_heatmap(y_test, y_pred)
    elif ml_algorithm == 'KNN':
        create_chart_heatmap(y_test, y_pred)
    elif ml_algorithm == 'Decision Tree':
        create_chart_tree(model, X, y)
    

def resultForecast(y_test, y_pred):
    st.subheader("Kết quả dự đoán")
    result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
    st.write(result_df)
        
        
def create_chart_scatterplot(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    sns.lineplot(x=y_test, y=y_test, color='red', label='Linear line')
    plt.xlabel("Thực tế")
    plt.ylabel("Dự đoán")
    plt.title("Biểu đồ dự đoán vs Thực tế")
    plt.legend()
    st.pyplot(plt)

def create_chart_heatmap(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d')
    plt.title("Ma trận Confusion")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    st.pyplot(plt)

def create_chart_tree(model, X, y):
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=y.unique().tolist())
    st.pyplot(plt) 