from sentence_transformers import SentenceTransformer
import numpy as np

class EmbeddingRouter:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # Example module embeddings
        self.module_map = {
            "qa_adapter": "question answering",
            "summarization_adapter": "text summarization"
        }

        self.module_embeddings = {
            k: self.model.encode(v)
            for k, v in self.module_map.items()
        }

    def route(self, text, top_k=1):
        query_emb = self.model.encode(text)

        scores = []
        for name, emb in self.module_embeddings.items():
            sim = np.dot(query_emb, emb) / (
                np.linalg.norm(query_emb) * np.linalg.norm(emb)
            )
            scores.append((name, sim))

        scores.sort(key=lambda x: x[1], reverse=True)

        return [name for name, _ in scores[:top_k]]