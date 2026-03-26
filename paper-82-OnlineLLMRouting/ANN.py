import faiss
import hnswlib
import numpy as np

class ANN():
    def  __init__(self, method="", embed_size=1024, sample_size=26497, top_k=5):
        self.methods = ["diskann", "faiss", "hnsw"]
        self.top_k = top_k
        if method not in self.methods:
            raise KeyError("method not found.")
        
        if method == self.methods[0]:
            raise  Exception("method not implemented")
        elif method == self.methods[1]:
            self.method = faiss.IndexFlatIP(embed_size)
            self.choice = 1
        elif method == self.methods[2]:
            self.method = hnswlib.Index(space='cosine', dim=embed_size)
            self.method.init_index(max_elements=sample_size, ef_construction=200, M=32)
            self.choice = 2

    def add(self, data):
        embeddings = data.astype('float32')
        if self.choice == 1:
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            self.method.add(embeddings)
        elif self.choice == 2:
            self.method.add_items(embeddings)
            self.method.set_ef(50)
        

    def search(self, query):
        query_vector = query.astype('float32')
        if self.choice ==1:
            query_vector = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
            distances, labels = self.method.search(query_vector, self.top_k)  
        elif self.choice == 2:
            labels, distances = self.method.knn_query(query_vector, k=self.top_k)
            distances = 1 - distances

        return labels, distances
