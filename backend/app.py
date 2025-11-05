import streamlit as st
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the trained model
with open("data/reco_model.pkl", "rb") as f:
    df, vectorizer, item_vectors = pickle.load(f)

titles = df["Book-Title"].tolist()

def get_recommendations(query, top_n=5):
    query_vec = vectorizer.transform([query])
    sims = cosine_similarity(query_vec, item_vectors).flatten()
    top_indices = sims.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][["Book-Title"]].values.flatten().tolist()

# Streamlit UI
st.set_page_config(page_title="Book Recommender", page_icon="📚")

st.title("📚 AI Book Recommender Chatbot")
st.write("Type what kind of book you want (e.g. *romantic novel*, *science fiction*, *thriller*)")

user_input = st.text_input("Your query:")

if user_input:
    recs = get_recommendations(user_input)
    st.subheader("Top Recommendations:")
    for i, title in enumerate(recs, 1):
        st.write(f"**{i}. {title}**")
