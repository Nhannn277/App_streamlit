import os
import io
import pickle
import base64
# import graphviz
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

def check(df):
    ml_algorithm = st.sidebar.selectbox("Chọn thuật toán", 
                                            [
                                                "Linear Regression", 
                                                "Logistic Regression", 
                                                "KNN", 
                                                "Decision Tree"
                                             ])
    dependent_var = ''
    max_depth = 1
    

    if ml_algorithm == 'Linear Regression':
        dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc", df.columns)
        if  df[dependent_var].dtype == 'object':
            st.warning('Biến phụ thuộc phải có kiểu số')          
    elif ml_algorithm == 'KNN':
        binary_columns = df.columns[df.nunique() == 2]
        if not binary_columns.empty:
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc", binary_columns)     
        else:
            st.warning('Không có cột phù hợp để làm biến phụ thuộc trong mô hình KNN')
            dependent_var = None   
    elif ml_algorithm == "Decision Tree":
        # chọn các cột không phải kiểu object
        object_columns =  df.select_dtypes(exclude=['object']).columns
        # nếu len(object_columns) == 0, thì các cột trong object_columns điều thuộc kiểu object 
        if  len(object_columns) == 0:
             dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc", object_columns)
        else: 
            st.warning('Không có cột phù hợp để làm biến phụ thuộc trong mô hình Decision Tree') 
            dependent_var = None 
        
    else:
        dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc", df.columns)
    
    if dependent_var:
        independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))
        null_columns_independent_vars = [col for col in independent_vars if df[col].isnull().any()]
    else:
        independent_vars = []
        null_columns_independent_vars = []
        st.warning('Chọn mô hình khác')
        
    if len(independent_vars) == 0:
        pass
    elif df[dependent_var].isnull().any():
        st.warning('Biến phụ thuộc không được chứa giá trị null')   
    elif null_columns_independent_vars:
        st.warning(f"cột {', '.join(null_columns_independent_vars)} có chứa giá trị null không thể thực hiện huấn luyện")
        st.warning('Vui lòng làm sạch dữ liệu trước khi huấn luyện')
    elif any(df[independent_vars].dtypes == 'object'):
        st.warning('biến độc lập phải có kiểu dữ liệu số')    
    else:
        if ml_algorithm == 'Decision Tree':
            max_depth = st.sidebar.slider('Độ sâu tối đa của cây quyết định', min_value=1, max_value=10, value=3, step=1)     
        if st.sidebar.button("Dự đoán"):
            if not dependent_var or not independent_vars:
                st.warning('Không để trống các ô')
            else:
                trainModel(ml_algorithm, df, independent_vars, dependent_var, max_depth)  

def trainModel(ml_algorithm, df, independent_vars, dependent_var, max_depth):
    
    X = df[independent_vars]
    y = df[dependent_var]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if ml_algorithm == "Linear Regression":  
        model = LinearRegression()
    elif ml_algorithm == "Logistic Regression":
        model = LogisticRegression()
    elif ml_algorithm == "KNN":
        model = KNeighborsClassifier()
    elif ml_algorithm == "Decision Tree":
        model = DecisionTreeClassifier()
    
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    
    resultForecast(y_test, y_pred)
    
    if ml_algorithm == 'Linear Regression':
        create_chart_scatterplot(y_test, y_pred)
    elif ml_algorithm == 'Logistic Regression':
        create_chart_scatterplot(y_test, y_pred)
    elif ml_algorithm == 'KNN':
        create_chart_confusion(y_test, y_pred)
    elif ml_algorithm == 'Decision Tree':
        create_chart_tree(model, X, y, max_depth)
    

def resultForecast(y_test, y_pred):
    st.subheader("Kết quả dự đoán")
    result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred}).T
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

def create_chart_confusion(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d')
    plt.title("Ma trận Confusion")
    plt.xlabel("Dự đoán")
    plt.ylabel("Thực tế")
    st.pyplot(plt)

def create_chart_tree(model, X, y, max_depth):
    plt.figure(figsize=(20, 10))
    plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=y.unique().tolist(), max_depth= max_depth)
    st.pyplot(plt.gcf()) 
    