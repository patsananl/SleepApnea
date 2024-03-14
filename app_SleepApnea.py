
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load model and encoders
with open('SleepApnea.pkl', 'rb') as file:
    model = pickle.load(file)

# Load your DataFrame
# Replace 'your_data.csv' with the actual file name or URL
df = pd.read_csv('Sleep_health_and_lifestyle_dataset.csv')
df = df.drop('Person ID', axis=1)

# Streamlit App
st.title('Obstructive Sleep Apnea Prediction')

# Define a session state to remember tab selections
if 'tab_selected' not in st.session_state:
    st.session_state.tab_selected = 0

# Create tabs for prediction and visualization
tabs = ['Sleep Apnea Prediction', 'Visualize Data', 'Predict from CSV']
selected_tab = st.radio('Select Tab:', tabs, index=st.session_state.tab_selected)

# Tab selection logic
if selected_tab != st.session_state.tab_selected:
    st.session_state.tab_selected = tabs.index(selected_tab)

# Tab 1: Obstructive Sleep Apnea Prediction
if st.session_state.tab_selected == 0:
    st.header('Obstructive Sleep Apnea Prediction')


    # User Input Form
    Gender_input = st.radio('Gender', ["Male", "Female"])
    Age_input = st.number_input('Age', value = None, placeholder="Input your age (e.g. 30)")
    Occupation_input = st.selectbox('Occupation', ["Nurse", "Doctor", "Engineer", "Lawyer", "Teacher", "Acccountant", "Saleperson"])
    Sleep_Duration_input = st.slider('Sleep Duration', 5, 9, 7)
    Quality_of_Sleep_input = st.slider('Quality of Sleep', 4, 9, 7)
    Physical_Activity_Level_input = st.slider('Physical Activity Level', 30, 100, 50)
    Stress_Level_input = st.slider('Stress Level', 3, 8, 7)
    BMI_Category_input = st.selectbox('BMI Category', ("Normal", "Overweight", "Obese"), index=None, placeholder="Select BMI Category...")
    Blood_Pressure_input = st.text_input('Blood Pressure', placeholder="Input Your Blood Pressure (e.g. 120/80)")
    Heart_Rate_input = st.number_input('Heart Rate', value = None, placeholder="Input your Heart Rate")
    Daily_Steps_input = st.number_input('Daily Steps', value=None, placeholder="Input your Daily Step")

    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
        'Gender_input': [Gender_input],
        'Age_input': [Age_input],
        'Occupation_input': [Occupation_input],
        'Sleep_Duration_input': [Sleep_Duration_input],
        'Quality_of_Sleep_input': [Quality_of_Sleep_input],
        'Physical_Activity_Level_input': [Physical_Activity_Level_input],
        'Stress_Level_input': [Stress_Level_input],
        'BMI_Category_input': [BMI_Category_input],
        'Blood_Pressure_input': [Blood_Pressure_input],
        'Heart_Rate_input': [Heart_Rate_input],
        'Daily_Steps_input': [Daily_Steps_input],
        'Systolic' : [0],
        'Diastolic' :[0],
        'Blood_Pressure_Category':[0]
    })

    # Categorical Data Encoding
    # user_input['Systolic'] = user_input['Blood_Pressure_input'].str.split('/').str[0].astype(int)
    # user_input['Diastolic'] = user_input['Blood_Pressure_input'].str.split('/').str[1].astype(int)

    # user_input.drop(['Blood_Pressure_input'], axis=1, inplace=True)

    # blood_class_conditions = [
    # (user_input['Systolic'] < 120) & (user_input['Diastolic'] < 80),
    # (user_input['Systolic'].between(120, 140)) & (user_input['Diastolic'] < 90),
    # (user_input['Systolic'] >= 140) & (user_input['Diastolic'] >= 90) | (user_input['Diastolic'] >= 80)
    # ]

    # labels = ['Optimal', 'Normal', 'Hypertension']

    # user_input['Blood_Pressure_Category'] = np.select(blood_class_conditions, labels, default='Undefined')

    # user_input["Gender_input"] = user_input["Gender_input"].astype("category").cat.codes
    # user_input["Occupation_input"] = user_input["Occupation_input"].astype("category").cat.codes
    # user_input["BMI_Category_input"] = user_input["BMI_Category_input"].astype("category").cat.codes
    # user_input["Blood_Pressure_Category_input"] = user_input["Blood_Pressure_Category_input"].astype("category").cat.codes

    # Predicting
    prediction = model.predict(user_input)

    # Display Result
    st.subheader('Obstructive Sleep Apnea Prediction')
    st.write('Sleep Disorder:', prediction[0])

# Tab 2: Visualize Data
elif st.session_state.tab_selected == 1:
    st.header('Visualize Data')

    # Select condition feature
    condition_feature = st.selectbox('Select Condition Feature:', df.columns)

    # Set default condition values
    default_condition_values = ['Select All'] + df[condition_feature].unique().tolist()

    # Select condition values
    condition_values = st.multiselect('Select Condition Values:', default_condition_values)

    # Handle 'Select All' choice
    if 'Select All' in condition_values:
        condition_values = df[condition_feature].unique().tolist()

    if len(condition_values) > 0:
        # Filter DataFrame based on selected condition
        filtered_df = df[df[condition_feature].isin(condition_values)]

        # Plot the number of Focus Group
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=condition_feature, hue='sleep_disorder', data=filtered_df, palette='viridis')
        plt.title('Number of Focus Group for Predicting Sleep Health')
        plt.xlabel(condition_feature)
        plt.ylabel('Number of Focus Group')
        st.pyplot(fig)

# Tab 3: Predict from CSV
elif st.session_state.tab_selected == 2:
    st.header('Predict from CSV')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    # uploaded_file

    if uploaded_file is not None:
        # Read CSV file
        csv_df_org = pd.read_csv(uploaded_file)
        csv_df_org = csv_df_org.dropna()
        # csv_df_org.columns

        csv_df = csv_df_org.copy()
        csv_df = csv_df.drop('Person ID',axis=1)



        # Categorical Data Encoding
        csv_df['Systolic'] = csv_df['Blood_Pressure'].str.split('/').str[0].astype(int)
        csv_df['Diastolic'] = csv_df['Blood_Pressure'].str.split('/').str[1].astype(int)
        
        csv_df.drop(['Blood_Pressure'], axis=1, inplace=True)
        
        csv_df["Gender"] = csv_df["Gender"].astype("category").cat.codes
        csv_df["Occupation"] = csv_df["Occupation"].astype("category").cat.codes
        csv_df["BMI_Category"] = csv_df["BMI_Category"].astype("category").cat.codes
        csv_df["Blood_Pressure_Category"] = csv_df["Blood_Pressure_Category"].astype("category").cat.codes


        # Predicting
        predictions = model.predict(csv_df)

        # Add predictions to the DataFrame
        csv_df_org['sleep_disorder'] = predictions

        # Display the DataFrame with predictions
        st.subheader('Predicted Results:')
        st.write(csv_df_org)

        # Visualize predictions based on a selected feature
        st.subheader('Visualize Predictions')

        # Select feature for visualization
        feature_for_visualization = st.selectbox('Select Feature for Visualization:', csv_df_org.columns)

        # Plot the Number of Focus Group for Predicting Sleep Health for the selected feature
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.countplot(x=feature_for_visualization, hue='sleep_disorder', data=csv_df_org, palette='viridis')
        plt.title(f'Number of Focus Group for Predicting Sleep Health - {feature_for_visualization}')
        plt.xlabel(feature_for_visualization)
        plt.ylabel('Number of Focus Group')
        st.pyplot(fig)

