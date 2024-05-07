import os
import pandas as pd
import pickle
import numpy as np
import streamlit as st
import seaborn as sns
import plotly.express as px
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt

import base64
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, LabelEncoder



from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import io
from sklearn.tree import plot_tree



st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")



working_dir = os.path.dirname(os.path.abspath(__file__))

heart_disease_model = pickle.load(open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb'))

with st.sidebar:
    selected = option_menu('Menu',
                           ['Upload CSV',
                            'Heart Disease Prediction',
                            'Clean Data'],
                           menu_icon='hospital-fill',
                           icons=['cloud-upload', 'heart','data'],
                           default_index=0)


#----------------------------------------------------------------------------------------------------------    

if selected == 'Upload CSV':
    
    st.title('Upload CSV')

    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            st.write(uploaded_file.name)
            st.write(df.head(1))
            col1, col2 = st.columns(2)
  
        ml_algorithm = st.sidebar.selectbox("Chọn thuật toán", ["Linear Regression", "Logistic Regression", "KNN", "Decision Tree"])

        if ml_algorithm == "Linear Regression":
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc", df.columns)
            independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))

            if st.sidebar.button("Dự đoán"):
                X = df[independent_vars]
                y = df[dependent_var]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                st.subheader("Kết quả dự đoán vs Giá trị thực tế")
                result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                st.write(result_df)

                plt.figure(figsize=(10, 6))
                sns.scatterplot(x=y_test, y=y_pred)
                sns.lineplot(x=y_test, y=y_test, color='red', label='Linear line')
                plt.xlabel("Thực tế")
                plt.ylabel("Dự đoán")
                plt.title("Biểu đồ dự đoán vs Thực tế")
                plt.legend()
                st.pyplot(plt)

        elif ml_algorithm == "Logistic Regression":
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
            independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))

            if st.sidebar.button("Dự đoán"):
                X = df[independent_vars]
                y = df[dependent_var]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = LogisticRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                st.subheader("Kết quả dự đoán")
                result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                st.write(result_df)

                plt.figure(figsize=(10, 6))
                sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d')
                plt.title("Ma trận Confusion")
                plt.xlabel("Dự đoán")
                plt.ylabel("Thực tế")
                st.pyplot(plt)

        elif ml_algorithm == "KNN":
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
            independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))

            if st.sidebar.button("Dự đoán"):
                X = df[independent_vars]
                y = df[dependent_var]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = KNeighborsClassifier()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                st.subheader("Kết quả dự đoán")
                result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                st.write(result_df)
                plt.figure(figsize=(10, 6))
                sns.heatmap(pd.crosstab(y_test, y_pred), annot=True, fmt='d')
                plt.title("Ma trận Confusion")
                plt.xlabel("Dự đoán")
                plt.ylabel("Thực tế")
                st.pyplot(plt)
        elif ml_algorithm == "Decision Tree":
            dependent_var = st.sidebar.selectbox("Chọn biến phụ thuộc (Chỉ phân loại)", df.columns)
            independent_vars = st.sidebar.multiselect("Chọn biến độc lập", df.columns.drop(dependent_var))

            if st.sidebar.button("Dự đoán"):
                X = df[independent_vars]
                y = df[dependent_var]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                model = DecisionTreeClassifier()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                st.subheader("Kết quả dự đoán")
                result_df = pd.DataFrame({"Thực tế": y_test, "Dự đoán": y_pred})
                st.write(result_df)

                # In ra cây quyết định
                plt.figure(figsize=(20, 10))
                plot_tree(model, filled=True, feature_names=X.columns.tolist(), class_names=y.unique().tolist())
                st.pyplot(plt)
    
#---------------
if selected == 'Heart Disease Prediction':

    # page title#
    st.title('Heart Disease Prediction using ML')

#     col1, col2, col3 = st.columns(3)

#     with col1:
#         age = st.text_input('Tuổi')

#     with col2:
#         sex = st.text_input('Giới tính (1 = Nam, 0 = Nữ)')

#     with col3:
#         cp = st.text_input('Loại đau ngực')

#     with col1:
#         trestbps = st.text_input('Huyết áp lúc nghỉ (tính bằng mm Hg)')

#     with col2:
#         chol = st.text_input('Cholestoral mg/dl')

#     with col3:
#         fbs = st.text_input('Đường trong máu > 120 mg/dl (1 = true; 0 = false)')

#     with col1:
#         restecg = st.text_input('Kết quả điện tâm đồ lúc nghỉ ngơi')

#     with col2:
#         thalach = st.text_input('Nhịp tim tối đa đạt được')

#     with col3:
#         exang = st.text_input('Tập thể dục có gây đau tắc ngực không (1 = Có; 0 = Không)')

#     with col1:
#         oldpeak = st.text_input('Chênh lệch đoan ST trong khi tập thể dục so với lúc nghỉ')

#     with col2:
#         slope = st.text_input('Độ dốc tại đỉnh của đoạn ST khi tập thể dục')

#     with col3:
#         ca = st.text_input('Số lượng đoạn mạch chính')

#     with col1:
#         thal = st.text_input('1 = bình thường, 2 = lỗi cố định, 3 = khiếm khuyết có thể đảo ngược')

#     # code for Prediction
#     heart_diagnosis = ''

#     # creating a button for Prediction

#     if st.button('Dự đoán bệnh tim'):

#         user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

#         user_input = [float(x) for x in user_input]

#         heart_prediction = heart_disease_model.predict([user_input])

#         if heart_prediction[0] == 1:
#             heart_diagnosis = 'Người này có mắc bệnh tim'
#         else:
#             heart_diagnosis = 'Người này không mắc bệnh tim'

#     st.success(heart_diagnosis)

#----------------------------------------------------------------------------------------------------------    

if selected == 'Clean Data':
    
    st.title('Clean Data')

    
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    

    if uploaded_files:
        
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            st.write(uploaded_file.name)
            st.write("Hiển thị 5 hàng đầu tiên của dataset")
            st.write(df.head(5))
            st.write("Số hàng và số cột trong dataset")
            st.write(df.shape)
            st.write("Kiểu dữ liệu của các cột trong dataset")
            st.write(df.dtypes)
            st.write("Kiểm tra missing values")
            st.write(df.isnull().sum())
            st.write("Mô tả dữ liệu")
            st.write(df.describe())
           
            
            
            
            # Hàm xóa cột
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
            
            # Kiểm tra nếu session state chưa tồn tại, khởi tạo mới
            if 'my_df' not in st.session_state:
                st.session_state.my_df = pd.DataFrame()
                st.session_state.my_df = df.copy()
            if 'deleted_columns' not in st.session_state:
                st.session_state.deleted_columns = []    
            
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(["Remove Columns", 
                                                                      "Fill Null Values", 
                                                                      "Handle duplicates", 
                                                                      "Remove Rows with Null", 
                                                                      "Change Data Types", 
                                                                      "Check Outliers", 
                                                                      "Encode Categorical Variables", 
                                                                      "Save dataset"])
        
            with tab1:
                st.write("Remove Columns")
                unwanted_col = st.multiselect("Remove column", st.session_state.my_df.columns, key="deleted_columns")
                if st.button('Remove'):
                    st.session_state.my_df = remove_col(st.session_state.my_df, unwanted_col)
                    st.session_state.deleted_columns.extend(unwanted_col)
                    st.write(st.session_state.my_df.head(5))

            with tab2:
                st.header("Fill Null Values")
                st.write(st.session_state.my_df.head(5))
                st.write("Choose columns to fill null values")
                selected_columns = st.multiselect("Columns", st.session_state.my_df.columns, key="fill_null_values")
                if st.button('Fill Null Values'):
                    # Áp dụng hàm fill_null_values cho các cột đã chọn
                    filled_df = fill_null_values(st.session_state.my_df, selected_columns)
                    # Cập nhật lại DataFrame
                    st.session_state.my_df = filled_df
                    st.write("Null values filled for selected columns")
                    st.write(st.session_state.my_df.head(5))

            
            with tab3:
                st.header("Handle Duplicates")

                if st.button("Handle Duplicates"):
                    
                    duplicate = st.session_state.my_df[st.session_state.my_df.duplicated(keep=False)]
                    
                    if duplicate.empty:
                        st.write("Don't have duplicate")
                    else:
                        st.write('row sum: {}'.format(len(st.session_state.my_df)))
                        #st.write('Have {} duplicates'.format(df.duplicated().sum()))  
                        
                        st.session_state.my_df = handle_duplicates(st.session_state.my_df)
                        
                        st.write('number of goods remaining after processing', len(st.session_state.my_df))
                                    


            with tab4:
                st.header("Remove Rows with Null")
                
                col1 , col2 = st.columns(2)
    
                with col1:
                    st.write("Kiểm tra missing values")
                    st.write(st.session_state.my_df.isnull().sum())

                with col2:
                    selected_columns = st.multiselect("Select columns to remove rows with null values:", st.session_state.my_df.columns, key="RemoveRowsNull")
                    
                    # Lấy mask cho các hàng có giá trị null trong các cột đã chọn
                    mask = st.session_state.my_df[selected_columns].isnull().any(axis=1)
                    
                    # Lấy DataFrame chứa các hàng có giá trị null
                    rows_with_null = st.session_state.my_df[mask]
                    
                    if st.button("Remove Rows with Null"):
                        # Xóa các hàng có giá trị null
                        st.session_state.my_df = st.session_state.my_df.drop(rows_with_null.index)
                        st.write("Rows with null values removed successfully.")
            
            with tab5:
                st.header("Change Data Types")
                st.write("Choose column and new data type:")

                # Hiển thị danh sách các cột và kiểu dữ liệu hiện tại
                st.write("Current data types:")
                st.write(st.session_state.my_df.dtypes)

                selected_column = st.selectbox("Column to convert", st.session_state.my_df.columns, key="convert_column")
                new_dtype = st.selectbox("New data type", ["int32", "int64", "float32", "float64", "object"], key="new_dtype")

                if st.button("Convert"):
                    # Áp dụng hàm convert_column_dtype cho cột được chọn
                    st.session_state.my_df = convert_column_dtype(st.session_state.my_df, selected_column, new_dtype)
                    st.write(f"Converted column '{selected_column}' to {new_dtype}")
                    st.write(st.session_state.my_df.dtypes)
                
            with tab6:
                st.header("Check Outliers")
                st.write("Choose column to check outliers")
                
                selected_column = st.selectbox("Column", st.session_state.my_df.columns, key="outlier_select")
                
                # tạo 1 container chứa biểu đồ trong trường hợp có outlier
                container_diagram =  st.empty()
                
                with container_diagram.container():
                    check_outliers_plot(st.session_state.my_df, selected_column)
                    
                if st.button('Handle Outliers'):
                    st.session_state.my_df = remove_outliers(df, selected_column)
                    st.write('remove successfully')
                    # sau khi remove successfully thì container sẽ được làm rỗng
                    container_diagram.empty()
            with tab7:
                st.header("Encode Categorical Variables")
                
                # Lựa chọn phương pháp mã hóa từ người dùng
                encode_method = st.selectbox("Select encoding method:", ["One-Hot Encoding", "Ordinal Encoding", "Label Encoding"])

                # Mã hóa dữ liệu theo phương pháp được chọn
                if encode_method == "One-Hot Encoding":
                    column = st.selectbox("Select column to encode:", st.session_state.my_df.columns)
                    df_encoded = one_hot_encode(st.session_state.my_df, column)
                elif encode_method == "Ordinal Encoding":
                    column = st.selectbox("Select column to encode:", st.session_state.my_df.columns)
                    df_encoded = ordinal_encode(st.session_state.my_df, column)
                else:  # Label Encoding
                    column = st.selectbox("Select column to encode:", st.session_state.my_df.columns)
                    df_encoded = label_encode(st.session_state.my_df, column)

                # Hiển thị kết quả
                st.write("Encoded DataFrame:")
                st.write(df_encoded)

            with tab8:
                st.header("Save dataset")
    
                # Kiểm tra nếu có DataFrame và đã clean data
                if 'my_df' in st.session_state and st.session_state.my_df is not None:
                    st.write("Your cleaned dataset:")
                    st.write(st.session_state.my_df.head())
                    
                    # Xác định tên file mặc định
                    default_filename = None
                    if uploaded_files:
                        # Nếu có file tải lên, sử dụng tên file đầu tiên kèm theo "_cleaned.csv"
                        default_filename = uploaded_files[0].name.split('.')[0] + "_cleaned.csv"
                    filename = st.text_input("Enter a filename to save as:", default_filename)
                    # Thêm nút để lưu dataset
                    if st.button("Save Cleaned Dataset"):
                        
                        if filename.strip() == "":
                            st.warning("Please enter a valid filename.")
                        else:
                            save_dataset(st.session_state.my_df, filename)
                            
                            # Hiển thị link để tải file về
                            download_link = get_download_link(st.session_state.my_df, filename, "Click here to download the cleaned dataset")
                            st.markdown(download_link, unsafe_allow_html=True)
                else:
                    st.warning("No cleaned dataset available. Please clean your data first.")
            
            
            
            
            
            
            

        
    


    



    
    

    