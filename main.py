# -*- coding: utf-8 -*-
"""
Integrated Health Assistant with AI Chatbot and Full Prediction Parameters
"""
import os
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set page configuration
st.set_page_config(page_title="Health Assistant",
                   layout="wide",
                   page_icon="🧑‍⚕️")

# Load working directory of main.py
working_dir = os.path.dirname(os.path.abspath(__file__))

# Load saved models for disease prediction
diabetes_model = pickle.load(open('C:/Users/agraw/Desktop/Main EL/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/agraw/Desktop/Main EL/heart_disease_model.sav', 'rb'))
parkinsons_model = pickle.load(open('C:/Users/agraw/Desktop/Main EL/parkinsons_model.sav', 'rb'))

# Chatbot response dictionary
chatbot_responses = {
    "parkinson": "Parkinson's disease, a neurological disorder causing tremors, stiffness, and movement difficulties.\n",
    "HNR" : "HNR is the ratio of harmonic energy to noise energy in a voice signal, typically expressed in decibels (dB).",
    "hnr" : "HNR is the ratio of harmonic energy to noise energy in a voice signal, typically expressed in decibels (dB).",
    "RPDE" : "RPDE measures the entropy (randomness or disorder) of recurrence periods in a time series. It evaluates how frequently certain patterns recur in the signal.",
    "rpde" : "RPDE measures the entropy (randomness or disorder) of recurrence periods in a time series. It evaluates how frequently certain patterns recur in the signal.",
    "DFA" : "DFA (Detrended Fluctuation Analysis) is a statistical method used to detect long-range correlations and scaling behavior in time-series data. In voice analysis, it is applied to assess the complexity and stability of vocal fold vibrations.",
    "dfa" : "DFA (Detrended Fluctuation Analysis) is a statistical method used to detect long-range correlations and scaling behavior in time-series data. In voice analysis, it is applied to assess the complexity and stability of vocal fold vibrations.",
    "Spread1" : "Spread1 is a parameter used in voice or signal analysis to measure the spectral dispersion of energy within a signal. It quantifies how concentrated or spread out the energy is in the lower frequencies of the spectrum",
    "Spread2" : " Spread2 calculates the dispersion (or spread) of spectral energy around the second spectral moment, emphasizing higher-frequency components of the signal.It complements Spread1 by providing insights into how energy is distributed across the higher frequencies of the spectrum.",
    "D2" : "D2 is a measure in voice or signal analysis that represents the Correlation Dimension, a mathematical parameter used to assess the complexity of a signal. It is particularly useful for analyzing non-linear systems, such as vocal fold vibrations, by estimating the dimensionality of the attractor in a phase space reconstruction of the signal.",
    "d2" : "D2 is a measure in voice or signal analysis that represents the Correlation Dimension, a mathematical parameter used to assess the complexity of a signal. It is particularly useful for analyzing non-linear systems, such as vocal fold vibrations, by estimating the dimensionality of the attractor in a phase space reconstruction of the signal.",
    "PPE" : "PPE (Pitch Period Entropy) is a parameter used in voice and speech analysis to measure the regularity and stability of the pitch (fundamental frequency, F0) across a voice signal. It quantifies the variability in pitch over time, particularly focusing on the degree of randomness or irregularity",
    "ppe" : "PPE (Pitch Period Entropy) is a parameter used in voice and speech analysis to measure the regularity and stability of the pitch (fundamental frequency, F0) across a voice signal. It quantifies the variability in pitch over time, particularly focusing on the degree of randomness or irregularity",
    "preventive measures for parkinson" : "1. Regular physical activity.\n"
                                          "2. A healthy diet rich in antioxidants.\n"
                                          "3. Managing stress through yoga or meditation.\n"
                                          "4. Maintaining social engagement.\n",
    "parkinsons": "Alias for Parkinson's disease.",
    "diabetes": "A metabolic disorder characterized by high blood sugar levels due to insulin issues.\n",
    "preventive measures for diabetes" : "1. Maintaining a healthy weight.\n"
                                              "2. Regular exercise.\n"
                                              "3. Eating a balanced diet rich in fiber.\n"
                                              "4. Monitoring blood sugar levels.\n",
    "sugar": "Alias for diabetes or high blood sugar.",
    "blood sugar": "Alias for diabetes or high blood sugar.",
    "hypertension": "A condition of persistently high blood pressure, increasing risk for heart disease.",
    "high blood pressure": "Alias for hypertension.",
    "bp": "Alias for hypertension or blood pressure.",
    "asthma": "A respiratory condition marked by difficulty breathing, often triggered by allergens or exertion.",
    "breathing difficulty": "Alias for asthma or difficulty breathing.",
    "shortness of breath": "Alias for asthma or difficulty breathing.",
    "heart disease": "A range of cardiovascular issues, including blocked arteries, heart attacks, and arrhythmias.\n",
    "preventive measures for heart disease" : "1. Maintaining a healthy weight.\n"
                                              "2. Regular physical activity.\n"
                                              "3. Managing blood pressure and cholesterol levels.\n"
                                              "4. Quitting smoking and limiting alcohol intake.\n",
    "cardiac disease": "Alias for heart disease or heart problems.",
    "heart problems": "Alias for heart disease or heart problems.",
    "cancer": "A group of diseases involving uncontrolled cell growth and potential spread to other parts of the body.",
    "tumor": "Alias for cancer or tumors.",
    "stroke": "A medical emergency where blood supply to the brain is disrupted, causing cell damage.",
    "brain attack": "Alias for stroke.",
    "pregnancies": "The number of times a person has been pregnant, relevant for gestational health.",
    "pregnancy count": "Alias for the number of pregnancies.",
    "glucose": "The concentration of glucose in plasma, used to measure blood sugar levels.",
    "blood glucose": "Alias for glucose or blood sugar concentration.",
    "sugar level": "Alias for glucose or blood sugar concentration.",
    "bloodpressure": "The force of blood against artery walls, specifically diastolic pressure.",
    "blood pressure": "Alias for bloodpressure or diastolic blood pressure.",
    "pressure": "Alias for bloodpressure or diastolic blood pressure.",
    "skinthickness": "The thickness of the skinfold at the triceps, indicating body fat percentage.",
    "skin thickness": "Alias for skinfold thickness measurement. A caliper is used to measure the thickness of skin after folding.",
    "Skin Thickness Value" : "The thickness of the skinfold at the triceps, indicating body fat percentage. A caliper is used to measure the thickness of skin after folding.",
    "fat layer": "Alias for skinfold thickness measurement.",
    "insulin": "Serum insulin levels, used to assess insulin sensitivity and pancreatic function.",
    "serum insulin": "Alias for insulin levels.",
    "Insulin Level": "Serum insulin levels, used to assess insulin sensitivity and pancreatic function.",
    "bmi": "Body Mass Index, a ratio of weight to height, used to estimate body fat.",
    "body mass index": "Alias for BMI.",
    "BMI value": "Body Mass Index, a ratio of weight to height, used to estimate body fat.",
    "weight to height ratio": "Alias for BMI.",
    "diabetespedigreefunction": "A measure of genetic predisposition to diabetes based on family history.",
    "Diabetes Pedigree Function value": "A measure of genetic predisposition to diabetes based on family history.",
    "genetic predisposition": "Alias for diabetes pedigree function.",
    "age": "The age of the individual in years.",
    "Age of the Person": "The age of the individual in years.",
    "years": "Alias for age.",
    "sex": "Biological sex or gender (1 = male, 0 = female).",
    "gender": "Alias for sex or gender.",
    "cp": "Type of chest pain experienced, categorized as angina or other forms.0: typical angina, 1: atypical angina, 2: non anginal pain, 3: asymptomatic",
    "chest pain": "Alias for type of chest pain.Type of chest pain experienced, categorized as angina or other forms.0: typical angina, 1: atypical angina, 2: non anginal pain, 3: asymptomatic",
    "angina": "Alias for chest pain type.",
    "trestbps": "Resting blood pressure measured while at rest, indicating baseline circulatory health.",
    "resting bp": "Alias for resting blood pressure.",
    "resting blood pressure": "Alias for resting blood pressure.",
    "Resting Blood Pressure": "Resting blood pressure measured while at rest, indicating baseline circulatory health.",
    "chol": "Serum cholesterol levels, used to assess cardiovascular health risks.",
    "cholesterol": "Alias for serum cholesterol.",
    "cholesterol level": "Alias for serum cholesterol.",
    "Cholesterol Level": "Serum cholesterol levels, used to assess cardiovascular health risks.",
    "fbs": "Fasting blood sugar levels (>120 mg/dL indicates diabetes risk).",
    "fasting blood sugar": "Alias for fasting blood sugar levels.",
    "Fasting Blood Sugar": "Alias for fasting blood sugar levels.",
    "fasting sugar": "Alias for fasting blood sugar levels.",
    "restecg": "Electrocardiogram results at rest, measuring heart's electrical activity.",
    "Resting ECG": "Electrocardiogram results at rest, measuring heart's electrical activity.",
    "ecg": "Alias for resting ECG results.",
    "electrocardiogram": "Alias for resting ECG results.",
    "thalach": "Maximum heart rate achieved during stress or exercise testing.",
    "heart rate": "Alias for maximum heart rate.",
    "maximum heart rate": "Alias for maximum heart rate.",
    "Maximum Heart Rate Achieved": "Alias for maximum heart rate.Maximum heart rate achieved during stress or exercise testing.",
    "exang": "Exercise-induced angina, a type of chest discomfort during physical activity.",
    "Exercise Induced Angina": "Exercise-induced angina, a type of chest discomfort during physical activity.",
    "exercise angina": "Alias for exercise-induced angina.",
    "chest discomfort": "Alias for exercise-induced angina.",
    "oldpeak": "ST segment depression relative to resting state, indicating heart stress.",
    "ST Depression": "ST Depression relative to resting state, indicating heart stress.Mild ST Depression: 0.5 to 1 mm (0.05 to 0.1 mV) below the baseline. Moderate ST Depression: 1 to 2 mm (0.1 to 0.2 mV) below the baseline. Severe ST Depression: Greater than 2 mm (0.2 mV) below the baseline",
    "st depression": "ST Depression relative to resting state, indicating heart stress. Mild ST Depression: 0.5 to 1 mm (0.05 to 0.1 mV) below the baseline. Moderate ST Depression: 1 to 2 mm (0.1 to 0.2 mV) below the baseline. Severe ST Depression: Greater than 2 mm (0.2 mV) below the baseline",
    "slope": "The slope of the ST segment during exercise (upsloping, flat, downsloping).",
    "Slope of Peak Exercise ST Segment": "The slope of the ST segment during exercise (upsloping, flat, downsloping).",
    "ca": "The number of major cardiac vessels colored by fluoroscopy (0-3).",
    "Number of Major Vessels": "The number of major cardiac vessels colored by fluoroscopy (0-3).",
    "thal": "Thalassemia, a blood disorder affecting oxygen transport in the body.",
    "Thalassemia": "Thalassemia, a blood disorder affecting oxygen transport in the body.",
    "target": "The presence of heart disease (1 = disease, 0 = no disease).",
    "mdvp:fo(hz)": "The fundamental frequency of the voice, used in voice analysis.",
    "mdvp fo": "The fundamental frequency of the voice, used in voice analysis.",
    "mdvp": "specify fo or fhi",
    "voice pitch": "Alias for fundamental voice frequency.",
    "mdvp:fhi(hz)": "The maximum frequency of the voice, indicating pitch range.",
    "mdvp fhi": "The maximum frequency of the voice, indicating pitch range.",
    "mdvp:fhi": "TThe maximum frequency of the voice, indicating pitch range.",
    "max pitch": "Alias for maximum voice frequency.",
    "mdvp:flo(hz)": "The minimum frequency of the voice, indicating pitch range.",
    "min pitch": "Alias for minimum voice frequency.",
    "jitter": "Variation in voice frequency, used to assess voice quality. Specify percentage or abs",
    "Jitter%": "Variation in voice frequency, used to assess voice quality.Enter value in percentage.",
    "jitter(%)": "Variation in voice frequency, used to assess voice quality.Enter value in percentage.",
    "jitter(abs)": "Jitter (Abs) is the mean absolute difference between the durations of consecutive cycles of a periodic signal, such as vocal fold vibrations. Its measured in microseconds.",
    "shimmer": "Variation in voice amplitude, used to assess voice quality.",
    "Shimmer": "Variation in voice amplitude, used to assess voice quality.",
    "Shimmer(dB)": "Variation in voice amplitude, used to assess voice quality.",
    
}

# Chatbot response function
def chatbot_response(user_input):
    lower_input = user_input.lower()
    if lower_input in chatbot_responses:
        return chatbot_responses[lower_input]
    input_ids = chatbot_tokenizer.encode(user_input, return_tensors="pt")
    response_ids = chatbot_model.generate(
        input_ids,
        max_length=100,
        pad_token_id=chatbot_tokenizer.eos_token_id,
        temperature=0.7,
        top_p=0.9,
    )
    return chatbot_tokenizer.decode(response_ids[0], skip_special_tokens=True)

# Load chatbot model and tokenizer
@st.cache_resource
def load_chatbot_model():
    chatbot_model_path = "C:/Users/agraw/Desktop/Main EL/chatbot"
    model = AutoModelForCausalLM.from_pretrained(chatbot_model_path)
    tokenizer = AutoTokenizer.from_pretrained(chatbot_model_path)
    return model, tokenizer

chatbot_model, chatbot_tokenizer = load_chatbot_model()

# Sidebar for navigation
with st.sidebar:
    selected = option_menu('MedixAI Disease Prediction System',
                           ['Diabetes Prediction',
                            'Heart Disease Prediction',
                            'Parkinsons Prediction',
                            'AI Chatbot'],
                           menu_icon='hospital-fill',
                           icons=['activity', 'heart', 'person', 'chat'],
                           default_index=0)

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction')
    col1, col2, col3 = st.columns(3)
    with col1:
        Pregnancies = st.text_input('Number of Pregnancies')
    with col2:
        Glucose = st.text_input('Glucose Level')
    with col3:
        BloodPressure = st.text_input('Blood Pressure value')
    with col1:
        SkinThickness = st.text_input('Skin Thickness value')
    with col2:
        Insulin = st.text_input('Insulin Level')
    with col3:
        BMI = st.text_input('BMI value')
    with col1:
        DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    with col2:
        Age = st.text_input('Age of the Person')

    diab_diagnosis = ''
    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(Pregnancies), float(Glucose), float(BloodPressure),
                          float(SkinThickness), float(Insulin), float(BMI),
                          float(DiabetesPedigreeFunction), float(Age)]
            diab_prediction = diabetes_model.predict([user_input])
            if diab_prediction[0] == 1:
                diab_diagnosis = 'The person is diabetic'
            else:
                diab_diagnosis = 'The person is not diabetic'
        except ValueError:
            diab_diagnosis = 'Please input valid numerical values.'
    st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction')
    col1, col2, col3 = st.columns(3)
    with col1:
        Age = st.text_input("Age")
    with col2:
        Sex = st.text_input("Sex (1 = Male, 0 = Female)")
    with col3:
        ChestPainType = st.text_input("Chest Pain Type (0-3)")
    with col1:
        RestingBP = st.text_input("Resting Blood Pressure")
    with col2:
        Cholesterol = st.text_input("Cholesterol Level")
    with col3:
        FastingBS = st.text_input("Fasting Blood Sugar (1 if > 120 mg/dL)")
    with col1:
        RestingECG = st.text_input("Resting ECG (0, 1, 2)")
    with col2:
        MaxHR = st.text_input("Maximum Heart Rate Achieved")
    with col3:
        ExerciseAngina = st.text_input("Exercise Induced Angina (1 = Yes, 0 = No)")
    with col1:
        Oldpeak = st.text_input("ST Depression")
    with col2:
        Slope = st.text_input("Slope of Peak Exercise ST Segment (0-2)")
    with col3:
        CA = st.text_input("Number of Major Vessels (0-3)")
    with col1:
        Thal = st.text_input("Thalassemia (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)")

    heart_diagnosis = ''
    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(Age), float(Sex), float(ChestPainType), float(RestingBP),
                          float(Cholesterol), float(FastingBS), float(RestingECG),
                          float(MaxHR), float(ExerciseAngina), float(Oldpeak),
                          float(Slope), float(CA), float(Thal)]
            heart_prediction = heart_disease_model.predict([user_input])
            if heart_prediction[0] == 1:
                heart_diagnosis = 'The person has a risk of heart disease'
            else:
                heart_diagnosis = 'The person does not have a risk of heart disease'
        except ValueError:
            heart_diagnosis = 'Please input valid numerical values.'
    st.success(heart_diagnosis)

# Parkinson's Prediction Page
if selected == 'Parkinsons Prediction':
    st.title("Parkinson's Disease Prediction")
    col1, col2, col3 = st.columns(3)
    with col1:
        Fo = st.text_input("MDVP:Fo(Hz)")
    with col2:
        Fhi = st.text_input("MDVP:Fhi(Hz)")
    with col3:
        Flo = st.text_input("MDVP:Flo(Hz)")
    with col1:
        Jitter = st.text_input("jitter(%)")
    with col2:
        JitterAbs = st.text_input("jitter(abs)")
    with col3:
        Shimmer = st.text_input("Shimmer")
    with col1:
        Shimmer_dB = st.text_input("Shimmer(dB)")
    with col2:
        HNR = st.text_input("hnr")
    with col3:
        RPDE = st.text_input("rpde")
    with col1:
        DFA = st.text_input("dfa")
    with col2:
        Spread1 = st.text_input("Spread1")
    with col3:
        Spread2 = st.text_input("Spread2")
    with col1:
        D2 = st.text_input("d2")
    with col2:
        PPE = st.text_input("ppe")

    parkinsons_diagnosis = ''
    if st.button("Parkinson's Test Result"):
        try:
            user_input = [float(Fo), float(Fhi), float(Flo), float(Jitter),
                          float(JitterAbs), float(Shimmer), float(Shimmer_dB),
                          float(HNR), float(RPDE), float(DFA), float(Spread1),
                          float(Spread2), float(D2), float(PPE)]
            parkinsons_prediction = parkinsons_model.predict([user_input])
            if parkinsons_prediction[0] == 1:
                parkinsons_diagnosis = "The person likely has Parkinson's disease"
            else:
                parkinsons_diagnosis = "The person does not have Parkinson's disease"
        except ValueError:
            parkinsons_diagnosis = 'Please input valid numerical values.'
    st.success(parkinsons_diagnosis)

# AI Chatbot Page
if selected == 'AI Chatbot':
    st.title("AI Chatbot Assistant")
    st.subheader("Chat with your AI Assistant(lowercase prefered)")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_message = st.text_input("You:", key="user_message")
    if st.button("Send"):
        if user_message:
            bot_response = chatbot_response(user_message)
            st.session_state.chat_history.append(("You", user_message))
            st.session_state.chat_history.append(("Bot", bot_response))

    for sender, message in st.session_state.chat_history:
        st.markdown(f"**{sender}:** {message}")

    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        st.experimental_rerun()
