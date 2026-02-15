"""RAG Second Brain — Multi-Source Retrieval Demo Application."""
import os
import time
from contextlib import asynccontextmanager
from typing import Optional
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from seed_data import PAPERS
from retrieval.dense import DenseRetriever
from retrieval.statistical import StatisticalRetriever
from retrieval.kg import KGRetriever
from retrieval.fusion import GatingFusion

# Global state
documents = []
dense_retriever = DenseRetriever()
stat_retriever = StatisticalRetriever()
kg_retriever = KGRetriever()
gating = GatingFusion()

def load_and_index():
    """Load seed data and index across all sources."""
    global documents
    documents = []
    for i, paper in enumerate(PAPERS):
        documents.append({
            "id": i,
            "title": paper["title"],
            "abstract": paper["abstract"],
            "authors": paper.get("authors", "Unknown"),
            "year": paper.get("year", 2024),
        })
    print(f"Indexing {len(documents)} documents...")
    t0 = time.time()
    dense_retriever.index(documents)
    print(f"  Dense index: {time.time()-t0:.1f}s")
    t1 = time.time()
    stat_retriever.index(documents)
    print(f"  Statistical index: {time.time()-t1:.1f}s")
    t2 = time.time()
    kg_retriever.index(documents)
    print(f"  KG index: {time.time()-t2:.1f}s")
    print(f"Total indexing: {time.time()-t0:.1f}s")
    print(f"  Entities: {len(kg_retriever.entities)}")
    print(f"  Relations: {len(kg_retriever.relations)}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_and_index()
    yield

app = FastAPI(title="RAG Second Brain", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ─── Pages ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    stats = {
        "doc_count": len(documents),
        "entity_count": len(kg_retriever.entities),
        "relation_count": len(kg_retriever.relations),
        "ppmi_terms": len(stat_retriever.ppmi_matrix),
    }
    return templates.TemplateResponse("index.html", {
        "request": request, "stats": stats, "documents": documents
    })

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request, q: Optional[str] = None):
    results = None
    gates = None
    expanded_terms = None
    query_entities = None
    individual = None
    if q and q.strip():
        query_entities = kg_retriever.get_query_entities(q)
        gates = gating.compute_gates(q, len(query_entities))
        dense_results = dense_retriever.search(q, top_k=10)
        stat_results = stat_retriever.search(q, top_k=10)
        kg_results = kg_retriever.search(q, top_k=10)
        fused = gating.fuse(
            {"dense": dense_results, "statistical": stat_results, "kg": kg_results},
            gates, top_k=10
        )
        # Enrich with document info
        doc_map = {d["id"]: d for d in documents}
        for item in fused:
            doc = doc_map.get(item["doc_id"], {})
            item["title"] = doc.get("title", "Unknown")
            item["abstract"] = doc.get("abstract", "")[:200] + "..."
            item["authors"] = doc.get("authors", "")
            item["year"] = doc.get("year", "")

        # Individual source results
        def enrich(source_results):
            enriched = []
            for doc_id, score in source_results[:5]:
                doc = doc_map.get(doc_id, {})
                enriched.append({
                    "doc_id": doc_id, "score": round(score, 4),
                    "title": doc.get("title", ""), "authors": doc.get("authors", ""),
                    "year": doc.get("year", ""),
                })
            return enriched

        individual = {
            "dense": enrich(dense_results),
            "statistical": enrich(stat_results),
            "kg": enrich(kg_results),
        }
        results = fused
        expanded_terms = stat_retriever.get_expanded_terms(q)

    example_queries = [
        "What methods improve multi-hop reasoning?",
        "How does retrieval-augmented generation reduce hallucination?",
        "What is the relationship between knowledge graphs and transformers?",
        "Compare dense and sparse retrieval approaches",
        "Which papers cite both BERT and attention mechanisms?",
        "How does PPMI relate to neural word embeddings?",
    ]
    return templates.TemplateResponse("search.html", {
        "request": request, "q": q or "", "results": results, "gates": gates,
        "expanded_terms": expanded_terms, "query_entities": query_entities,
        "individual": individual, "example_queries": example_queries,
    })

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload")
async def upload_document(title: str = Form(...), content: str = Form(...)):
    new_id = len(documents)
    doc = {"id": new_id, "title": title, "abstract": content,
           "authors": "User", "year": 2026}
    documents.append(doc)
    # Re-index all
    dense_retriever.index(documents)
    stat_retriever.index(documents)
    kg_retriever.index(documents)
    return RedirectResponse(url="/?uploaded=1", status_code=303)

@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request):
    return templates.TemplateResponse("graph.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

# ─── API ─────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    return {
        "documents": len(documents),
        "entities": len(kg_retriever.entities),
        "relations": len(kg_retriever.relations),
        "ppmi_terms": len(stat_retriever.ppmi_matrix),
    }

@app.get("/api/graph")
async def api_graph():
    return kg_retriever.get_graph_data()

@app.get("/api/search")
async def api_search(q: str, top_k: int = 10):
    query_entities = kg_retriever.get_query_entities(q)
    gates = gating.compute_gates(q, len(query_entities))
    dense_results = dense_retriever.search(q, top_k=top_k)
    stat_results = stat_retriever.search(q, top_k=top_k)
    kg_results = kg_retriever.search(q, top_k=top_k)
    fused = gating.fuse(
        {"dense": dense_results, "statistical": stat_results, "kg": kg_results},
        gates, top_k=top_k
    )
    doc_map = {d["id"]: d for d in documents}
    for item in fused:
        doc = doc_map.get(item["doc_id"], {})
        item["title"] = doc.get("title", "")
        item["authors"] = doc.get("authors", "")
        item["year"] = doc.get("year", "")
    return {"query": q, "gates": gates, "results": fused,
            "expanded_terms": stat_retriever.get_expanded_terms(q),
            "query_entities": query_entities}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
