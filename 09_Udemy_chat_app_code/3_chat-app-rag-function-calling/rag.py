import json
import os


def _cosine_similarity(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class RagProvider:
    def __init__(self, data_dir=None):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(__file__), "data")
        self._data_dir = data_dir

    def fetch_document_data(self, file_name: str):
        file_path = os.path.join(self._data_dir, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def prepare_simple_rag_prompt(self, query: str) -> str:
        kb_data = self.fetch_document_data("knowledgeBase.json")
        context = "\n\n".join(
            f"Q: {item['question']}\nA: {item['answer']}" for item in kb_data
        )
        return f"""You are an AI assistant with access to the following knowledge base:
    {context}
    Based on the above knowledge, answer the following user question:
    User: {query}
    Answer in one short paragraph.
    """

    def prepare_rag_prompt(self, query: str, query_vector: list, faq_vectors: list) -> str:
        ranked = []
        for item in faq_vectors:
            score = _cosine_similarity(query_vector, item["vector"])
            ranked.append({**item, "score": score})
        ranked.sort(key=lambda x: x["score"], reverse=True)
        top2 = ranked[:2]
        context = "\n".join(item["answer"] for item in top2)
        return f"""Use the context below to answer. If the answer isn't there, say "It's not available in the documentation, but I will try to help you as best as I can." and try to help based on your general knowledge.

      Context:
      {context}

      User: {query}""".strip()
