import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

print("📂 Loading dataset...")
df = pd.read_csv("data/amazon_products.csv", low_memory=False)
print(f"✅ Loaded: {len(df)} rows")

# Select text column
text_column = "Book-Title"
df = df.dropna(subset=[text_column]).drop_duplicates(subset=[text_column])
titles = df[text_column].astype(str).tolist()

# ✅ Use only the first 50k rows to prevent memory issues
titles = titles[:50000]
df = df.iloc[:50000]

print(f"✅ Using {len(titles)} titles for lightweight model.")

print("⚙️ Vectorizing...")
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
item_vectors = vectorizer.fit_transform(titles)
print("✅ Vectorization done!")

print("💾 Saving model...")
with open("data/reco_model.pkl", "wb") as f:
    pickle.dump((df, vectorizer, item_vectors), f)

print("✅ Saved successfully at data/reco_model.pkl")
