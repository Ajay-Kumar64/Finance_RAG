from fastapi import FastAPI
from elasticsearch import Elasticsearch

app = FastAPI()

es = Elasticsearch(
    "http://elasticsearch:9200",
    headers={"Accept": "application/json"},
)

@app.get("/")
def health():
    return {
        "status": "ok",
        "elasticsearch": es.info()
    }
