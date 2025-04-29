import streamlit as st
import pandas as pd
import pickle

# 1. Load the trained Logistic Regression model
model = pickle.load(open('DecisionTree.pkl', 'rb'))

# 2. Streamlit app title
st.title("🎯 Resume Shortlisting Prediction (Logistic Regression)")

st.write("""
Provide candidate details below to predict if the resume will be shortlisted.
""")

# 3. Input fields for features
years_experience = st.number_input("Enter years of experience:", min_value=0, max_value=50, step=1)
education = st.selectbox("Select education level (encoded):", options=[0, 1, 2, 3])
matching_skills = st.number_input("Enter number of matching skills:", min_value=0, max_value=50, step=1)
projects_completed = st.number_input("Enter number of completed projects:", min_value=0, max_value=50, step=1)
certifications = st.number_input("Enter number of certifications:", min_value=0, max_value=20, step=1)
gpa = st.number_input("Enter GPA:", min_value=0.0, max_value=10.0, step=0.01)

# 4. Predict button
if st.button('Predict'):
    try:
        # Collect inputs into a dataframe
        input_df = pd.DataFrame([[years_experience, education, matching_skills, projects_completed, certifications, gpa]],
                                columns=['years_experience', 'education', 'matching_skills', 'projects_completed', 'certifications', 'gpa'])
        
        # Predict using the model
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # 5. Show result
        if prediction == 1:
            st.success(f"✅ Resume is likely to be shortlisted! (Confidence: {prediction_proba[1]*100:.2f}%)")
        else:
            st.error(f"❌ Resume is unlikely to be shortlisted. (Confidence: {prediction_proba[0]*100:.2f}%)")
    
    except Exception as e:
        st.error(f"Error: {e}")
