import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import os
from typing import List, Dict

class SimpleVectorStore:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """Initialize with a lightweight embedding model"""
        print("Loading embedding model...")
        self.encoder = SentenceTransformer(model_name)
        self.index = None
        self.texts = []
        self.metadata = []
        print("✅ Embedding model loaded!")
        
    def create_index(self, texts: List[str], metadata: List[Dict] = None):
        """Create FAISS index from texts"""
        print(f"Creating index for {len(texts)} documents...")
        self.texts = texts
        self.metadata = metadata or [{} for _ in texts]
        
        # Generate embeddings
        embeddings = self.encoder.encode(texts, show_progress_bar=True)
        embeddings = np.array(embeddings).astype('float32')
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product = cosine similarity after normalization
        self.index.add(embeddings)
        
        print(f"✅ Index created with {len(texts)} documents!")
        
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None:
            return []
            
        # Encode and normalize query
        query_embedding = self.encoder.encode([query])
        query_embedding = np.array(query_embedding).astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.texts):
                results.append({
                    'text': self.texts[idx],
                    'score': float(score),
                    'metadata': self.metadata[idx] if idx < len(self.metadata) else {},
                    'rank': i + 1
                })
        
        return results
    
    def save(self, path: str = "vector_store"):
        """Save index and data"""
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save texts and metadata
        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump({
                'texts': self.texts,
                'metadata': self.metadata
            }, f)
        print(f"✅ Vector store saved to {path}")
    
    def load(self, path: str = "vector_store"):
        """Load index and data"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Vector store not found at {path}")
            
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Load texts and metadata
        with open(os.path.join(path, "data.pkl"), "rb") as f:
            data = pickle.load(f)
            self.texts = data['texts']
            self.metadata = data['metadata']
        print(f"✅ Vector store loaded from {path}")