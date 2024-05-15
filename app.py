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
from controller.cleanDataController import (
    remove_col, 
    label_encode, 
    save_dataset,
    ordinal_encode, 
    one_hot_encode, 
    fill_null_values,
    handle_duplicates, 
    get_download_link, 
    check_outliers_plot, 
    convert_column_dtype 
)

st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")



working_dir = os.path.dirname(os.path.abspath(__file__))

heart_disease_model = pickle.load(
    open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb')
    )

with st.sidebar: selected = option_menu('Menu',
                                        [
                                            'Upload CSV',
                                            'Heart Disease Prediction',
                                            'Clean Data'
                                        ],
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
  
        ml_algorithm = st.sidebar.selectbox("Chọn thuật toán", 
                                            [
                                                "Linear Regression", 
                                                "Logistic Regression", 
                                                "KNN", "Decision Tree"
                                             ])

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

            # Kiểm tra nếu session state chưa tồn tại, khởi tạo mới
            if 'my_df' not in st.session_state:

                st.session_state.my_df = df.copy()

            if 'deleted_columns' not in st.session_state:
                st.session_state.deleted_columns = []
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.header(f"{df.shape[1]} hàng đầu")
                st.write(df.head(df.shape[1]))
                st.write(f" ( {df.shape[0]} Hàng, {df.shape[1]} Cột )") 
            with col3: 
                st.header("Kiểu dữ liệu")
                st.write(df.dtypes)
           
            col1, col2, col3 = st.columns(3)
            with col1:
                st.header("Mô tả dữ liệu:")
                st.write(df.describe())
            with col2:
                st.write()
            with col3: 
                st.header("Kiểm tra missing values:")
                missing_values = df.isnull().sum(axis=0)
                missing_values = missing_values[missing_values > 0].to_frame().T

                if missing_values.empty:
                    st.write("Không có missing values trong dataset.")
                else:
                    st.write(missing_values)
                    
                    
            # Main content
            st.markdown("***")

            # Create an anchor point
            st.markdown('<div id="main-content"></div>', unsafe_allow_html=True)

            # JavaScript to scroll to the anchor point
            scroll_script = """
            <script>
                document.querySelectorAll('.option-menu div[role="radiogroup"] > label').forEach(function(label) {
                    label.addEventListener('click', function() {
                        document.getElementById('main-content').scrollIntoView({ behavior: 'smooth' });
                    });
                });
            </script>
            """
            st.markdown(scroll_script, unsafe_allow_html=True)
                    
            with st.sidebar: option = option_menu('Select an option',
                       ["Remove Columns", 
                        "Fill Null Values",
                        "Handle Duplicates", 
                        "Remove Rows with Null",
                        "Change Data Types",
                        "Check Outliers", 
                        "Encode Categorical Variables",
                        "Save Dataset"],
                       menu_icon='gear',
                       icons=['columns-gap',  # Remove Columns
                              'file-earmark-excel',  # Fill Null Values
                              'files',  # Handle Duplicates
                              'trash',  # Remove Rows with Null
                              'clipboard-data',  # Change Data Types
                              'exclamation-triangle',  # Check Outliers
                              'tags',  # Encode Categorical Variables
                              'save'],  # Save Dataset
                       default_index=0) 
           
            if option == "Remove Columns":
                st.header("Remove Columns")
                st.session_state.my_df = st.session_state.my_df
                unwanted_col = st.multiselect("Select columns", st.session_state.my_df.columns, key="deleted_columns")
                if st.button('Remove'):
                    st.session_state.my_df = remove_col(st.session_state.my_df, unwanted_col)
                    st.session_state.deleted_columns.extend(unwanted_col)
                    st.write(st.session_state.my_df.head(5))
                    
            elif option == "Fill Null Values":
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
                    
            elif option == "Handle Duplicates":
                st.header("Handle Duplicates")

                duplicate = st.session_state.my_df[st.session_state.my_df.duplicated(keep=False)]
                
                if duplicate.empty:
                    st.markdown(" ''' :green[ Don't have duplicate] ''' ")
                else:
                    st.markdown('''total duplicate rows: :red[{}] '''.format(len(st.session_state.my_df)))
                    
                    if st.button("Handle Duplicates"):
                        st.session_state.my_df = handle_duplicates(st.session_state.my_df)
                        st.write('number of goods remaining after processing', len(st.session_state.my_df))
                                        
            elif option == "Remove Rows with Null":
                st.header("Remove Rows with Null")
                
                col1 , col2 = st.columns(2)

                with col1:
                    missing_values_2 = st.session_state.my_df.isnull().sum(axis=0)
                    missing_values_2 = missing_values_2[missing_values_2 > 0].to_frame().T

                    if missing_values_2.empty:
                        st.write("Không có missing values trong dataset.")
                    else:
                        st.write(missing_values_2)

                with col2:

                    selected_columns = st.multiselect("Select columns to remove rows with null values:", missing_values_2.columns, key="RemoveRowsNull")
                    
                    # Lấy mask cho các hàng có giá trị null trong các cột đã chọn
                    mask = st.session_state.my_df[selected_columns].isnull().any(axis=1)
                    
                    # Lấy DataFrame chứa các hàng có giá trị null
                    rows_with_null = st.session_state.my_df[mask]
                    
                    if st.button("Remove Rows with Null"):
                        # Xóa các hàng có giá trị null
                        st.session_state.my_df = st.session_state.my_df.drop(rows_with_null.index)
                        st.write("Rows with null values removed successfully.")
             
            elif option == "Change Data Types":
                st.header("Change Data Types")
                st.write("Choose column and new data type:")

                # Hiển thị danh sách các cột và kiểu dữ liệu hiện tại
                st.write("Current data types:")
                st.write(st.session_state.my_df.dtypes.to_frame().T)

                selected_column = st.selectbox("Column to convert", st.session_state.my_df.columns, key="convert_column")
                new_dtype = st.selectbox("New data type", ["int32", "int64", "float32", "float64", "object"], key="new_dtype")

                if st.button("Convert"):
                    # Áp dụng hàm convert_column_dtype cho cột được chọn
                    st.session_state.my_df = convert_column_dtype(st.session_state.my_df, selected_column, new_dtype)
                    st.write(f"Converted column '{selected_column}' to {new_dtype}")
                    st.write(st.session_state.my_df.dtypes)
                
            elif option == "Check Outliers":
                st.header("Check Outliers")

                col1, col2 = st.columns(2)

                with col1: 
                    selected_column = st.selectbox("Choose column to check outliers", st.session_state.my_df.columns, key="outlier_select")

                with col2: 
                   check_outliers_plot(st.session_state.my_df, selected_column)
                    
            elif option == "Encode Categorical Variables":
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

                if st.button('Save'):
                    st.session_state.my_df = df_encoded
            else:
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
                    if st.session_state.my_df is None:
                        st.warning("No cleaned dataset available. Please clean your data first.")
            
            
            
            

        
    


    



    
    

    
