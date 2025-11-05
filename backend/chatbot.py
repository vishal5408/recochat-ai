import pickle
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# 🔹 Load the trained model
print("📦 Loading recommendation model...")
with open("data/reco_model.pkl", "rb") as f:
    df, vectorizer, item_vectors = pickle.load(f)
print("✅ Model loaded successfully!")

titles = df["Book-Title"].tolist()

# 🔹 Function to get recommendations
def get_recommendations(query, top_n=5):
    # Convert query to vector using the same TF-IDF vectorizer
    query_vec = vectorizer.transform([query])
    
    # Compute cosine similarity between query and all items
    sims = cosine_similarity(query_vec, item_vectors).flatten()
    
    # Get top N recommendations
    top_indices = sims.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][["Book-Title"]].values.flatten().tolist()

# 🔹 Chatbot loop
print("\n🤖 Chatbot ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    
    recs = get_recommendations(user_input)
    print("\n📚 Recommendations:")
    for i, title in enumerate(recs, 1):
        print(f"{i}. {title}")
    print()
