import os
import io
import pickle
import base64
import numpy as np
import pandas as pd
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns # type: ignore
from sklearn.tree import plot_tree # type: ignore
from sklearn.tree import DecisionTreeClassifier # type: ignore
from sklearn.neighbors import KNeighborsClassifier # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay # type: ignore
from sklearn.linear_model import LinearRegression, LogisticRegression # type: ignore
from sklearn.preprocessing import  OneHotEncoder, OrdinalEncoder, LabelEncoder # type: ignore

def remove_col(my_df, unwanted_col):
    my_df = my_df.drop(columns=unwanted_col, errors='ignore')
    return my_df

# Hàm điền giá trị null
def fill_null_values(my_df, selected_columns):
    for col in selected_columns:
        if my_df[col].dtype == "object":
            mode_val = my_df[col].mode()[0]
            my_df[col].fillna(mode_val, inplace=True)
        if my_df[col].dtype == "int64" or my_df[col].dtype == "float64":  # Nếu cột là số
            unique_values = my_df[col].dropna().unique()
            if len(unique_values) == 2 and set(unique_values) == {0, 1}:  # Nếu chỉ có 2 giá trị và là 0 hoặc 1
                mode_val = my_df[col].mode()[0]  # Lấy mode (giá trị xuất hiện nhiều nhất)
                my_df[col].fillna(mode_val, inplace=True)
        if my_df[col].dtype == "int64" or my_df[col].dtype == "int32":  # Nếu cột là số nguyên
            if my_df[col].nunique() > 2:  # Nếu có nhiều hơn 2 giá trị khác nhau
                mean_val = my_df[col].mean().astype(int)  # Lấy giá trị trung bình kiểu int
                # Thay thế các giá trị 0 bằng giá trị trung bình
                my_df[col] = my_df[col].replace(0, mean_val)
                my_df[col].fillna(mean_val, inplace=True)  # Điền giá trị null bằng giá trị trung bình
        if my_df[col].dtype == "float64" or my_df[col].dtype == "float32":  # Nếu cột là số thực
            if my_df[col].nunique() > 2:  # Nếu có nhiều hơn 2 giá trị khác nhau
                mean_val = my_df[col].mean().astype(float)  # Lấy giá trị trung bình kiểu float
                # Thay thế các giá trị 0 bằng giá trị trung bình
                my_df[col] = my_df[col].replace(0, mean_val)
                my_df[col].fillna(mean_val, inplace=True)  # Điền giá trị null bằng giá trị trung bình
    return my_df

# Hàm ép kiểu
def convert_column_dtype(my_df, column, new_dtype):
    try:
        if new_dtype == "int32":
            # Fill NaN and inf with a placeholder value (here we use -1)
            my_df[column].fillna(0, inplace=True)
            my_df[column].replace([np.inf, -np.inf], 0, inplace=True)
            my_df[column] = my_df[column].astype(np.float32).astype(np.int32)
        elif new_dtype == "int64":
            my_df[column].fillna(0, inplace=True)
            my_df[column].replace([np.inf, -np.inf], 0, inplace=True)
            my_df[column] = my_df[column].astype(np.float64).astype(np.int64)
        elif new_dtype == "float32":
            my_df[column].replace([np.inf, -np.inf], np.nan, inplace=True)
            my_df[column] = my_df[column].astype(np.float32)
        elif new_dtype == "float64":
            my_df[column].replace([np.inf, -np.inf], np.nan, inplace=True)
            my_df[column] = my_df[column].astype(np.float64)
        elif new_dtype == "object":
            my_df[column] = my_df[column].astype(str)
    except Exception as e:
        st.error(f"Error converting column {column} to {new_dtype}: {e}")
    return my_df

# Hàm check outliers
def check_outliers_plot(my_df, selected_column):
    if my_df[selected_column].dtype == "int64" or my_df[selected_column].dtype == "float64" or my_df[selected_column].dtype == "float32" or my_df[selected_column].dtype == "int32":
        # Tính giá trị Q1, Q3 và IQR
        Q1 = my_df[selected_column].quantile(0.25)
        Q3 = my_df[selected_column].quantile(0.75)
        IQR = Q3 - Q1
        
        # Tìm giá trị ngoại lệ dưới và trên
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Tạo DataFrame chứa thông tin về outlier
        outliers = my_df[(my_df[selected_column] < lower_bound) | (my_df[selected_column] > upper_bound)]
        
        if outliers.empty:
            st.write("No outliers found.")
        else:
            # Vẽ biểu đồ box plot
            fig = px.box(my_df, y=selected_column, title=f'Box plot of {selected_column}')
            st.plotly_chart(fig)
        
def remove_outliers(my_df, selected_column):
    # Tính giá trị Q1, Q3 và IQR
    Q1 = my_df[selected_column].quantile(0.25)
    Q3 = my_df[selected_column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Tìm giá trị ngoại lệ dưới và trên
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Loại bỏ các ngoại lệ khỏi dữ liệu
    my_df = my_df[(my_df[selected_column] >= lower_bound) & (my_df[selected_column] <= upper_bound)]
    
    return my_df
            
# Hàm lưu dataset
def save_dataset(my_df, filename):
    my_df.to_csv(filename, index=False)
    st.success(f"Dataset saved as {filename}")

# Hàm download dataset
def get_download_link(my_df, filename, text):
    csv = my_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encode to base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

# Hàm xử lý duplicate
def handle_duplicates(my_df):
    
    # Drop duplicates
    my_df.drop_duplicates(keep= 'first', inplace=True)
    
    return my_df  
            
# Hàm mã hóa biến phân loại bằng phương pháp One-Hot Encoding
def one_hot_encode(my_df, column):
    encoder = OneHotEncoder()
    encoded = encoder.fit_transform(my_df[[column]]).toarray()
    df_encoded = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
    my_df = pd.concat([my_df, df_encoded], axis=1)
    my_df.drop(columns=[column], inplace=True)
    return my_df

# Hàm mã hóa biến phân loại bằng phương pháp Ordinal Encoding
def ordinal_encode(my_df, column):
    encoder = OrdinalEncoder()
    my_df[[column]] = encoder.fit_transform(my_df[[column]])
    return my_df

# Hàm mã hóa biến phân loại bằng phương pháp Label Encoding
def label_encode(my_df, column):
    encoder = LabelEncoder()
    my_df[column] = encoder.fit_transform(my_df[column])
    return my_df    

