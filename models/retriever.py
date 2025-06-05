from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, es_hosts, es_api_key, index_name="medical_cases"):
        self.es = Elasticsearch(hosts=es_hosts, api_key=es_api_key)
        self.index_name = index_name
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient for indexing

    def index_cases(self, case_list):
        for i, case in enumerate(case_list):
            embedding = self.encoder.encode(case["text"])
            doc = {"text": case["text"], "embedding": embedding.tolist(), "label": case["label"]}
            self.es.index(index=self.index_name, id=i, body=doc)

    def search(self, query, k=5):
        query_vec = self.encoder.encode(query).tolist()
        script_query = {
            "script_score": {
                "query": {"match_all": {}},
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": {"query_vector": query_vec}
                }
            }
        }
        res = self.es.search(index=self.index_name, body={"size": k, "query": script_query})
        return [hit["_source"] for hit in res["hits"]["hits"]]