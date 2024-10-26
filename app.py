import streamlit as st
import pandas as pd
import numpy as np
from statistics import harmonic_mean
from langdetect import detect
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the CSV file
@st.cache_data
def load_data():
    df = pd.read_csv('coursea_data.csv')
    # Drop unnecessary columns
    df.drop(['Unnamed: 0', 'course_organization'], axis=1, inplace=True)
    
    # Filter out rows where course enrollment does not end with 'k'
    df = df[df.course_students_enrolled.str.endswith('k')]

    # Convert course enrollment to integers
    df['course_students_enrolled'] = df['course_students_enrolled'].apply(lambda enrolled: eval(enrolled[:-1]) * 1000)

    # Scale course rating and enrollment
    minmax_scaler = MinMaxScaler()
    scaled_ratings = minmax_scaler.fit_transform(df[['course_rating', 'course_students_enrolled']])
    df['course_rating'] = scaled_ratings[:, 0]
    df['course_students_enrolled'] = scaled_ratings[:, 1]

    # Calculate the overall rating using harmonic mean
    df['overall_rating'] = df[['course_rating', 'course_students_enrolled']].apply(lambda row: harmonic_mean(row), axis=1)

    # Filter for English courses only
    df = df[df.course_title.apply(lambda title: detect(title) == 'en')]

    return df

# Load and preprocess the data
df = load_data()

# Vectorize course titles
@st.cache_resource
def vectorize_titles(df):
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors = vectorizer.fit_transform(df.course_title)
    return vectorizer, vectors

vectorizer, vectors = vectorize_titles(df)

# Define recommendation function
def recommend_by_course_title(title, recomm_count=10):
    title_vector = vectorizer.transform([title])
    cosine_sim = cosine_similarity(vectors, title_vector)
    idx = np.argsort(np.array(cosine_sim[:, 0]))[-recomm_count:]
    recommended_courses = df.iloc[idx].sort_values(by='overall_rating', ascending=False)
    return recommended_courses[['course_title', 'course_rating', 'course_students_enrolled', 'overall_rating']]

# Streamlit App Layout
st.title("Course Recommendation System")
st.write("Enter a course title to get recommendations.")

# User input for course title
course_title = st.text_input("Course Title", "A Crash Course in Data Science")

# Number of recommendations to display
num_recommendations = st.slider("Number of Recommendations", 1, 20, 10)

# Display recommendations
if st.button("Recommend Courses"):
    recommendations = recommend_by_course_title(course_title, recomm_count=num_recommendations)
    st.write("### Recommended Courses")
    st.dataframe(recommendations)
