from sklearn.base import is_regressor
import streamlit as st
import pickle
import numpy as np
import requests

def load_model():
    try:
        # URL to the raw pickle file
        url = 'https://github.com/Dharshina/Salary-Prediction/raw/main/saved_steps.pkl'
        response = requests.get(url)
        response.raise_for_status()  # Check if the request was successful

        # Load the pickled model
        model_data = pickle.loads(response.content)
        return model_data

    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching model from URL: {e}")
    except pickle.UnpicklingError as e:
        st.error(f"Error loading pickled model: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")

data = load_model()

regressor_loaded = data["model"]
le_country = data["le_country"]
le_education = data["le_education"]

def show_predict_page():
    #We create different widgets
    st.title("Software Developer Salary Prediction")

    st.write(""" ### Please provide some information to predict the salary""")

    countries = (
                "United States of America",                                
                "Other",                                                    
                "Germany",                                                  
                "United Kingdom of Great Britain and Northern Ireland",     
                "Canada",                                                   
                "India",                                                    
                "France",                                                   
                "Netherlands",                                              
                "Australia",                                                
                "Brazil",                                                   
                "Spain",                                                    
                "Sweden",                                                    
                "Italy",                                                     
                "Poland",                                                    
                "Switzerland",                                               
                "Denmark",                                                   
                "Norway",                                                    
                "Israel",                                                    
    )

    education = (
        "Less than a Bachelors",
        "Bachelor's degree",
        "Master's degree",
        "Post grad",
    )

    country = st.selectbox("Country", countries)
    education = st.selectbox("Education Level", education)

    experience = st.slider("Years of Experience", 0, 50, 3)

    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[country, education, experience]])
        X[:, 0] = le_country.transform(X[:,0])
        X[:, 1] = le_education.transform(X[:,1])

        X=X.astype(float)

        salary = regressor_loaded.predict(X)
        salary_float = float(salary[0])
        st.subheader(f"The estimated salary is $ {salary_float:.2f}")
