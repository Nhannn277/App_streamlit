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
from matplotlib_venn import venn2, venn3
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
from controller.trainModelController import check

st.set_page_config(page_title="Health Assistant", layout="wide", page_icon="🧑‍⚕️")



working_dir = os.path.dirname(os.path.abspath(__file__))

heart_disease_model = pickle.load(
    open(f'{working_dir}/saved_models/heart_disease_model.sav', 'rb')
    )

with st.sidebar:
    selected = option_menu('Menu',
                           ['Upload CSV', 
                            'Data Analysis',
                            'Heart Disease Prediction',
                            'Clean Data',
                            'Merge Data'],
                           menu_icon='hospital-fill',
                           icons=['cloud-upload', 'bar-chart-fill', 'heart', 'trash-fill', 'arrows-angle-contract'],
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
            check(df)
#---------------------------------------------------------------------------------------------------------- 


if selected == 'Data Analysis':
    st.title('Data Analysis')
    
    uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            df = pd.read_csv(uploaded_file)
            st.write(uploaded_file.name)
            st.write(df.head(1))
            
       
        ml_algorithm = st.selectbox("Chọn biểu đồ", ["Bar chart", 
                                                 "Line chart", 
                                                 "Pie chart", 
                                                 "Area chart", 
                                                 "Scatter plot", 
                                                 "Histogram", 
                                                 "Box-and-whisker plot",
                                                 "Venn diagram",])

        selected_variables = st.multiselect("Chọn biến", df.columns)
       
        # Vẽ biểu đồ dựa trên lựa chọn từ người dùng
        if ml_algorithm == "Bar chart":
            if len(selected_variables) == 1:
                plt.xlabel(selected_variables)
                plt.ylabel("Frequency")
                st.bar_chart(df[selected_variables])
            elif len(selected_variables) >= 2:
                # Tạo biểu đồ Bar Chart
                st.subheader("Biểu đồ Bar Chart")
                plt.xlabel(selected_variables[0])
                plt.ylabel("Values")
                sns.barplot(x=selected_variables[0], y=selected_variables[1], data=df)
                st.pyplot()
                

        elif ml_algorithm == "Line chart":
            if len(selected_variables) == 1:
                plt.xlabel(selected_variables)
                plt.ylabel("Value")
                st.line_chart(df[selected_variables])
            elif len(selected_variables) >= 2:
                # Tạo biểu đồ Line Chart
                st.subheader("Biểu đồ Line Chart")
                plt.xlabel(selected_variables[0])
                plt.ylabel("Values")
                for variable in selected_variables[1:]:
                    sns.lineplot(x=selected_variables[0], y=variable, data=df, label=variable)
                plt.legend()
                st.pyplot()
                

        elif ml_algorithm == "Pie chart":
                for var in selected_variables:
                    if df[var].dtype == 'object':  # Kiểm tra kiểu dữ liệu của biến
                        counts = df[var].value_counts()
                        fig, ax = plt.subplots()
                        ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                        st.subheader(f"Biểu đồ Pie Chart cho biến {var}")
                        st.pyplot(fig)
                    else:
                        fig, ax = plt.subplots()
                        ax.pie(df[var], labels=df.index, autopct='%1.1f%%', startangle=140)
                        ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                        st.subheader(f"Biểu đồ Pie Chart cho biến {var}")
                        st.pyplot(fig)
                

        elif ml_algorithm == "Area chart":
            if len(selected_variables) == 1:
                plt.figure(figsize=(10, 6))
                df[selected_variables].value_counts().sort_index().plot.area()
                plt.xlabel(selected_variables)
                plt.ylabel('Count')
                plt.title(f'Area Chart for {selected_variables}')
                st.pyplot()
            elif len(selected_variables) == 2:
                plt.figure(figsize=(10, 6))
                df.groupby(selected_variables[0])[selected_variables[1]].value_counts().unstack().plot.area()
                plt.xlabel(selected_variables[0])
                plt.ylabel('Count')
                plt.title(f'Area Chart for {selected_variables[0]} and {selected_variables[1]}')
                st.pyplot()


        elif ml_algorithm == "Scatter plot":
            if len(selected_variables) == 1:
                plt.figure(figsize=(10, 6))
                df[selected_variables].value_counts().sort_index().plot(marker='o', linestyle='None', markersize=8)
                plt.xlabel(selected_variables)
                plt.ylabel('Count')
                plt.title(f'Scatter Plot for {selected_variables}')
                st.pyplot()

            elif len(selected_variables) >= 2:
                plt.figure(figsize=(10, 6))
                plt.scatter(df[selected_variables[0]], df[selected_variables[1]])
                plt.xlabel(selected_variables[0])
                plt.ylabel(selected_variables[1])
                plt.title(f'Scatter Plot for {selected_variables[0]} and {selected_variables[1]}')
                st.pyplot()
                
                
        elif ml_algorithm == "Histogram":
            if len(selected_variables) == 1:
                plt.figure(figsize=(10, 6))
                df[selected_variables].plot.hist(bins=20, edgecolor='black')
                plt.xlabel(selected_variables)
                plt.ylabel('Frequency')
                plt.title(f'Histogram for {selected_variables}')
                st.pyplot()
            elif len(selected_variables) == 2:
                plt.figure(figsize=(10, 6))
                plt.hist2d(df[selected_variables[0]], df[selected_variables[1]], bins=20, cmap='Blues')
                plt.colorbar(label='Frequency')
                plt.xlabel(selected_variables[0])
                plt.ylabel(selected_variables[1])
                plt.title(f'Histogram for {selected_variables[0]} and {selected_variables[1]}')
                st.pyplot()


        elif ml_algorithm == "Box-and-whisker plot":
            if all(df[var].dtype in ['int64', 'float64', 'int32', 'float32'] for var in selected_variables):
                if len(selected_variables) == 1:
                    plt.figure(figsize=(10, 6))
                    df.boxplot(column=selected_variables)
                    plt.ylabel(selected_variables)
                    plt.title(f'Box-and-Whisker Plot for {selected_variables}')
                    st.pyplot()
                elif len(selected_variables) == 2:
                    plt.figure(figsize=(10, 6))
                    df.boxplot(column=selected_variables[1], by=selected_variables[0])
                    plt.xlabel(selected_variables[0])
                    plt.ylabel(selected_variables[1])
                    plt.title(f'Box-and-Whisker Plot for {selected_variables[0]} and {selected_variables[1]}')
                    st.pyplot()
            else:
                st.write('Select columns with int or float data')
                            
                            
        elif ml_algorithm == "Venn diagram":
            if len(selected_variables) == 2:
                set1 = set(df[selected_variables[0]])
                set2 = set(df[selected_variables[1]])
                plt.figure(figsize=(8, 6))
                venn2([set1, set2], (selected_variables[0], selected_variables[1]))
                plt.title('Venn Diagram for 2 Sets')
                st.pyplot()
            elif len(selected_variables) == 3:
                set1 = set(df[selected_variables[0]])
                set2 = set(df[selected_variables[1]])
                set3 = set(df[selected_variables[2]])
                plt.figure(figsize=(8, 6))
                venn3([set1, set2, set3], (selected_variables[0], selected_variables[1], selected_variables[2]))
                plt.title('Venn Diagram for 3 Sets')
                st.pyplot()
                        
    
#----------------------------------------------------------------------------------------------------------    
if selected == 'Heart Disease Prediction':

    # page title#
    st.title('Heart Disease Prediction using ML')

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.text_input('Tuổi')

    with col2:
        sex = st.text_input('Giới tính (1 = Nam, 0 = Nữ)')

    with col3:
        cp = st.text_input('Loại đau ngực')

    with col1:
        trestbps = st.text_input('Huyết áp lúc nghỉ (tính bằng mm Hg)')

    with col2:
        chol = st.text_input('Cholestoral mg/dl')

    with col3:
        fbs = st.text_input('Đường trong máu > 120 mg/dl (1 = true; 0 = false)')

    with col1:
        restecg = st.text_input('Kết quả điện tâm đồ lúc nghỉ ngơi')

    with col2:
        thalach = st.text_input('Nhịp tim tối đa đạt được')

    with col3:
        exang = st.text_input('Tập thể dục có gây đau tắc ngực không (1 = Có; 0 = Không)')

    with col1:
        oldpeak = st.text_input('Chênh lệch đoan ST trong khi tập thể dục so với lúc nghỉ')

    with col2:
        slope = st.text_input('Độ dốc tại đỉnh của đoạn ST khi tập thể dục')

    with col3:
        ca = st.text_input('Số lượng đoạn mạch chính')

    with col1:
        thal = st.text_input('1 = bình thường, 2 = lỗi cố định, 3 = khiếm khuyết có thể đảo ngược')

    # code for Prediction
    heart_diagnosis = ''

    # creating a button for Prediction

    if st.button('Dự đoán bệnh tim'):

        user_input = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]

        user_input = [float(x) for x in user_input]

        heart_prediction = heart_disease_model.predict([user_input])

        if heart_prediction[0] == 1:
            heart_diagnosis = 'Người này có mắc bệnh tim'
        else:
            heart_diagnosis = 'Người này không mắc bệnh tim'

    st.success(heart_diagnosis)

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
            
            
#----------------------------------------------------------------------------------------------------------                

if selected == 'Merge Data':
    st.header("Merge Data")

    st.header("Upload the first CSV file")
    uploaded_file1 = st.file_uploader('',type=["csv"])
    st.header("Upload the second CSV file")
    uploaded_file2 = st.file_uploader(' ', type=["csv"])

    if uploaded_file1 is not None and uploaded_file2 is not None:
        df1 = pd.read_csv(uploaded_file1)
        df2 = pd.read_csv(uploaded_file2)

        st.subheader("First CSV")
        st.write(df1)

        st.subheader("Second CSV")
        st.write(df2)

        st.subheader("Select columns to merge on")

        merge_col1 = st.selectbox("Select merge column from first CSV", df1.columns)
        merge_col2 = st.selectbox("Select merge column from second CSV", df2.columns, index=df2.columns.get_loc(merge_col1) if merge_col1 in df2.columns else 0)

        how_option = st.selectbox("How to merge", ["inner", "outer", "left", "right"])




        def save_dataset(merged_df, filename):
                merged_df.to_csv(filename, index=False)
                st.success(f"Dataset saved as {filename}")


        if st.button("Merge"):
            merged_df = pd.merge(df1, df2, left_on=merge_col1, right_on=merge_col2, how=how_option)
            st.subheader("Merged Data")
            st.write(merged_df)
            
            default_filename = "Merged_Data.csv"
            filename = st.text_input("Enter a filename to save as:", default_filename)
                # Thêm nút để lưu dataset
            if st.button("Save Cleaned Dataset"):
                        
                if filename.strip() == "":
                    st.warning("Please enter a valid filename.")
                else:
                    save_dataset(merged_df, filename)
                            
                    # Hiển thị link để tải file về
                    download_link = get_download_link(merged_df, filename, "Click here to download the cleaned dataset")
                    st.markdown(download_link, unsafe_allow_html=True)                 
            
                     

        
    


    



    
    

    
