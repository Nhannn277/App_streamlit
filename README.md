## Setup Instructions

1. **Clone the repository:**
    ```sh
    git clone <repository-url>
    cd <repository-directory>
    ```

2. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

3. **Run the Streamlit app:**
    ```sh
    streamlit run app.py
    ```

## Project Files

- **app.py**: The main Streamlit app code.
- **requirements.txt**: Lists the required Python packages for the Streamlit app.
- **colab_files_to_train_models/Multiple disease prediction system - heart.ipynb**: Jupyter notebook for training the heart disease prediction model.
- **dataset/heart.csv**: Dataset used for training the heart disease prediction model.
- **saved_models/heart_disease_model.sav**: Pre-trained heart disease prediction model.

## Usage

### Heart Disease Prediction

1. Open the Streamlit app.
2. Select "Heart Disease Prediction" from the sidebar menu.
3. Enter the required input values:
    - Age
    - Sex (1 = Male, 0 = Female)
    - Chest Pain Type
    - Resting Blood Pressure (in mm Hg)
    - Cholesterol (mg/dl)
    - Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)
    - Resting Electrocardiographic Results
    - Maximum Heart Rate Achieved
    - Exercise Induced Angina (1 = Yes, 0 = No)
    - ST Depression Induced by Exercise Relative to Rest
    - Slope of the Peak Exercise ST Segment
    - Number of Major Vessels Colored by Fluoroscopy
    - Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)
4. Click the "Dự đoán bệnh tim" button to get the prediction.

## Additional Information

- **.vscode/jsconfig.json**: Configuration file for JavaScript in Visual Studio Code.
- **.vscode/main.js**: JavaScript file for Visual Studio Code extension.
- **.gitignore**: Specifies files and directories to be ignored by Git.

## License

This project is licensed under the MIT License.
