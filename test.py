from fastapi import FastAPI
import pymongo
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# ------------------ MongoDB Connection ------------------
client = pymongo.MongoClient("your-mongodb-atlas-connection-string")
db_freelancer = client["Freelancer"]
db_jobs = client["Jobs"]

freelancer_collection = db_freelancer["Freelancer_data"]
job_collection = db_jobs["Jobs_data"]

# ------------------ Helper Functions ------------------
def get_closest_job_title(input_title, job_titles):
    matches = difflib.get_close_matches(input_title.lower(), job_titles, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_recommendations(job_title, skill_weight=0.6, rating_weight=0.3, review_weight=0.1):
    df_freelancers = pd.DataFrame(list(freelancer_collection.find({}, {"_id": 0, "tags": 1, "client_average_rating": 1, "client_review_count": 1})))
    df_jobs = pd.DataFrame(list(job_collection.find({}, {"_id": 0, "index": 1, "Key Skills": 1, "Job Title": 1})))

    if df_freelancers.empty or df_jobs.empty:
        return []

    df_freelancers.fillna({"client_average_rating": 3.0, "client_review_count": 0}, inplace=True)
    df_jobs.dropna(subset=["index", "key skills", "job title"], inplace=True)

    df_freelancers["tags"] = df_freelancers["tags"].str.lower()
    df_jobs["key skills"] = df_jobs["key skills"].str.lower()
    df_jobs["job title"] = df_jobs["job title"].str.lower()

    closest_match = get_closest_job_title(job_title, df_jobs["job title"].tolist())
    if not closest_match:
        return []

    job_row = df_jobs[df_jobs["job title"] == closest_match]
    job_skills = job_row["key skills"].values[0]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df_freelancers["tags"].tolist() + [job_skills])

    freelancer_vectors = tfidf_matrix[:-1]
    job_vector = tfidf_matrix[-1]

    skill_similarity_scores = cosine_similarity(job_vector, freelancer_vectors).flatten()

    df_freelancers["skill_similarity"] = skill_similarity_scores
    df_freelancers["normalized_rating"] = df_freelancers["client_average_rating"] / 5
    df_freelancers["normalized_reviews"] = df_freelancers["client_review_count"] / df_freelancers["client_review_count"].max()

    df_freelancers["final_score"] = (
        (df_freelancers["skill_similarity"] * skill_weight) +
        (df_freelancers["normalized_rating"] * rating_weight) +
        (df_freelancers["normalized_reviews"] * review_weight)
    )

    recommended = df_freelancers.sort_values(by="final_score", ascending=False)

    return recommended[["tags", "client_average_rating", "client_review_count", "final_score"]].to_dict(orient="records")

# ------------------ API Endpoint ------------------
@app.get("/recommend/{job_title}")
def recommend(job_title: str):
    recommendations = get_recommendations(job_title)
    if not recommendations:
        return {"message": "No matching freelancers found"}
    return {"recommendations": recommendations}
