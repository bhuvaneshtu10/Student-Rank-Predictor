import streamlit as st
import pandas as pd

import numpy as np
import joblib

# Load saved model pipeline artifacts

reg_pipe = joblib.load('total_score_regressor.pkl')  # For TotalScore prediction
try:
    clf_pipe = joblib.load("grade_classifier.pkl")  # For Grade classification (optional)
except:
    clf_pipe = None

# Define input options based on dataset attributes and metadata
gender_options = ['Male', 'Female', 'Other']
department_options = ['CS', 'Engineering', 'Business', 'Mathematics', 'Physics', 'Biology', 'Chemistry', 'Others']
education_options = ["No High School", "High School", "Some College", "Associate Degree", "Bachelor's Degree",
                     "Master's Degree", "PhD"]
income_levels = ["Low", "Medium", "High"]
yes_no = ['Yes', 'No']

# Streamlit UI
st.title("🎓 Student Academic Performance Prediction")

st.header("Input student details below:")

gender = st.selectbox("Gender", gender_options)
age = st.number_input("Age", 15, 100, 20)
department = st.selectbox("Department", department_options)
attendance = st.slider("Attendance (%)", 0, 100, 80)
midterm_score = st.slider("Midterm Score", 0.0, 100.0, 70.0)
final_score = st.slider("Final Score", 0.0, 100.0, 75.0)
assignments_avg = st.slider("Assignments Average Score", 0.0, 100.0, 80.0)
quizzes_avg = st.slider("Quizzes Average Score", 0.0, 100.0, 70.0)
participation_score = st.slider("Participation Score (0-10)", 0.0, 10.0, 7.0)
projects_score = st.slider("Project Score", 0.0, 100.0, 85.0)
study_hours_per_week = st.slider("Study Hours Per Week", 0, 100, 15)
extracurriculars = st.selectbox("Extracurricular Activities", yes_no)
internet_access = st.selectbox("Internet Access at Home", yes_no)
parent_education_level = st.selectbox("Parental Education Level", education_options)
family_income_level = st.selectbox("Family Income Level", income_levels)
stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
sleep_hours = st.slider("Sleep Hours Per Night", 1, 24, 7)

# Construct a DataFrame for prediction
input_dict = {
    "Gender": [gender],
    "Age": [age],
    "Department": [department],
    "Attendance": [attendance],
    "MidtermScore": [midterm_score],
    "FinalScore": [final_score],
    "AssignmentsAvg": [assignments_avg],
    "QuizzesAvg": [quizzes_avg],
    "ParticipationScore": [participation_score],
    "ProjectsScore": [projects_score],
    "StudyHoursperWeek": [study_hours_per_week],
    "ExtracurricularActivities": [extracurriculars],
    "InternetAccessatHome": [internet_access],
    "ParentEducationLevel": [parent_education_level],
    "FamilyIncomeLevel": [family_income_level],
    "StressLevel1-10": [stress_level],
    "SleepHoursperNight": [sleep_hours]
}

input_df = pd.DataFrame(input_dict)

if st.button("Predict Performance"):
    # Predict TotalScore (regression)
    total_score_pred = reg_pipe.predict(input_df)[0]
    st.write(f"### Predicted Total Score: {total_score_pred:.2f}")

    # Suggestion based on Total Score
    if total_score_pred < 50:
        st.warning("Suggestion: Additional support in fundamentals recommended.")
    elif total_score_pred < 75:
        st.info("Suggestion: Maintain consistent effort and attendance.")
    else:
        st.success("Suggestion: Excellent performance! Consider advanced coursework.")

    # Optional Grade Classification
    if clf_pipe:
        grade_pred = clf_pipe.predict(input_df)[0]
        st.write(f"### Predicted Grade: {grade_pred}")
        if grade_pred in ['D', 'F']:
            st.warning("Grade indicates risk: recommend tutoring and close monitoring.")
        elif grade_pred == 'C':
            st.info("Grade is average: encourage regular improvements.")
        else:
            st.success("Good grade: encourage continued effort and challenges.")
