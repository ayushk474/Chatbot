import pymongo
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ------------------ MongoDB Connection ------------------
client = pymongo.MongoClient("mongodb://localhost:27017/")
db_freelancer = client["Freelancer"]
db_jobs = client["Jobs"]

freelancer_collection = db_freelancer["Freelancer_data"]
job_collection = db_jobs["Jobs_data"]

# ------------------ Fetch Data from MongoDB ------------------
df_freelancers = pd.DataFrame(list(freelancer_collection.find({}, {"_id": 0, "tags": 1, "client_average_rating": 1, "client_review_count": 1})))
df_jobs = pd.DataFrame(list(job_collection.find({}, {"_id": 0, "index": 1, "Key Skills": 1, "Job Title": 1})))

# Handle empty datasets
if df_freelancers.empty or df_jobs.empty:
    st.error("Freelancer or Job database is empty. Please check MongoDB.")
    st.stop()

# ------------------ Data Cleaning ------------------
df_freelancers.columns = df_freelancers.columns.str.strip().str.lower()
df_jobs.columns = df_jobs.columns.str.strip().str.lower()

# Assign default values for missing ratings and reviews
# df_freelancers.dropna(subset=["tags", "client_average_rating", "client_review_count"], inplace=True)

df_freelancers["client_average_rating"].fillna(3.0, inplace=True)  # Default rating = 3
df_freelancers["client_review_count"].fillna(0, inplace=True)  # Default review count = 0

df_jobs.dropna(subset=["index", "key skills", "job title"], inplace=True)

df_freelancers["tags"] = df_freelancers["tags"].str.lower()
df_jobs["key skills"] = df_jobs["key skills"].str.lower()
df_jobs["job title"] = df_jobs["job title"].str.lower()

# ------------------ Helper Function: Fuzzy Job Title Matching ------------------
def get_closest_job_title(input_title, job_titles):
    matches = difflib.get_close_matches(input_title.lower(), job_titles, n=1, cutoff=0.6)
    return matches[0] if matches else None

# ------------------ Recommendation System ------------------
def get_recommendations(job_title, skill_weight=0.6, rating_weight=0.3, review_weight=0.1):
    # Fuzzy match job title
    closest_match = get_closest_job_title(job_title, df_jobs["job title"].tolist())
    
    if not closest_match:
        return pd.DataFrame(), None  # No valid job title found

    job_row = df_jobs[df_jobs["job title"] == closest_match]
    job_id = job_row["index"].values[0]
    job_skills = job_row["key skills"].values[0]

    # ------------------ TF-IDF Vectorization ------------------
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_freelancers["tags"].tolist() + [job_skills])

    freelancer_vectors = tfidf_matrix[:-1]  # All freelancers
    job_vector = tfidf_matrix[-1]  # Selected job

    # Compute cosine similarity
    skill_similarity_scores = cosine_similarity(job_vector, freelancer_vectors).flatten()

    # ------------------ Normalize Rating & Reviews ------------------
    max_reviews = df_freelancers["client_review_count"].max()
    df_freelancers["skill_similarity"] = skill_similarity_scores
    df_freelancers["normalized_rating"] = df_freelancers["client_average_rating"] / 5
    df_freelancers["normalized_reviews"] = df_freelancers["client_review_count"] / max_reviews if max_reviews > 0 else 0

    # ------------------ Compute Final Score ------------------
    # Give a slight boost to new freelancers with high skill similarity
    df_freelancers["final_score"] = (
    (df_freelancers["skill_similarity"] * skill_weight) +
    (df_freelancers["normalized_rating"] * rating_weight) +
    (df_freelancers["normalized_reviews"] * review_weight) +
    (df_freelancers["skill_similarity"] * 0.1)  # Small boost for skill match
    )


    recommended = df_freelancers.sort_values(by="final_score", ascending=False)

    return recommended[["tags", "client_average_rating", "client_review_count", "final_score"]], closest_match

# ------------------ Streamlit UI ------------------
st.title("🔹 Skill-Based Freelancer Recommender")
st.write("Find the best freelancers based on skills, ratings, and past work!")

input_job_title = st.text_input("Enter Job Title", "")

skill_weight = st.slider("Skill Matching Weight", 0.0, 1.0, 0.6, 0.05)
rating_weight = st.slider("Freelancer Rating Weight", 0.0, 1.0, 0.3, 0.05)
review_weight = st.slider("Past Work Done (Review Count) Weight", 0.0, 1.0, 0.1, 0.05)

if st.button("Get Recommendations"):
    st.subheader("🔹 Recommended Freelancers")
    
    recommendations, matched_title = get_recommendations(input_job_title, skill_weight, rating_weight, review_weight)

    if recommendations.empty:
        st.write("No matching job found. Try a different title.")
    else:
        st.write(f"Showing results for **{matched_title}**")
        
        # Highlight Top Freelancer
        top_freelancer = recommendations.iloc[0]
        st.subheader(f"Top Freelancer: {top_freelancer['tags']}")
        st.metric(label="Rating", value=round(top_freelancer["client_average_rating"], 2))
        st.metric(label="Final Score", value=round(top_freelancer["final_score"], 2))
        
        st.table(recommendations)
