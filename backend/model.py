import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("📂 Loading dataset...")
df = pd.read_csv("data/amazon_products.csv", low_memory=False)
print(f"✅ Loaded: {len(df)} rows")

# Select relevant text column
text_column = "Book-Title"
if text_column not in df.columns:
    raise ValueError(f"Column '{text_column}' not found in CSV!")

# Clean data
df = df.dropna(subset=[text_column]).drop_duplicates(subset=[text_column])
titles = df[text_column].astype(str).sample(5000, random_state=42).tolist()

print(f"✅ Using {len(titles)} unique titles.")

# Convert text into vectors
print("⚙️  Converting titles to TF-IDF vectors...")
vectorizer = TfidfVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(titles)
print("✅ Vectorization complete!")

# Compute similarity
print("🔍 Computing cosine similarity...")
similarity = cosine_similarity(vectors)
print("✅ Similarity matrix created!")

# Save everything for later use by the chatbot
with open("data/reco_model.pkl", "wb") as f:
    pickle.dump((df, vectorizer, similarity), f)

print("💾 Model saved successfully at data/reco_model.pkl")
