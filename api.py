import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
import logging
from typing import List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="QA Local API",
    description="API de Question-Réponse basée sur RAG avec Ollama",
    version="1.0.0"
)

# Configuration CORS pour permettre l'accès depuis un frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
INDEX_DIR = os.environ.get("INDEX_DIR", "index")
COLLECTION = os.environ.get("COLLECTION", "qa_local")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "koesn/mistral-7b-instruct")
TOP_K = int(os.environ.get("TOP_K", "4"))


class Query(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Question à poser")
    top_k: Optional[int] = Field(None, ge=1, le=10, description="Nombre de documents à récupérer")


class Answer(BaseModel):
    answer: str
    sources: List[str]
    confidence: Optional[float] = None


# Initialisation avec gestion d'erreurs
try:
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)
    vs = Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=INDEX_DIR
    )
    llm = ChatOllama(model=LLM_MODEL, temperature=0.2, num_ctx=4096)
    logger.info("Initialisation réussie des modèles")
except Exception as e:
    logger.error(f"Erreur d'initialisation: {e}")
    raise


@app.get("/health")
async def health_check():
    """Point de contrôle de santé de l'API."""
    return {"status": "healthy", "model": LLM_MODEL}


@app.get("/collections")
async def get_collections():
    """Retourne des informations sur la collection."""
    try:
        count = vs._collection.count()
        return {"collection": COLLECTION, "documents": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {e}")


@app.post("/qa", response_model=Answer)
async def qa(q: Query):
    """Point d'entrée principal pour les questions."""
    try:
        # Utiliser top_k personnalisé ou par défaut
        k = q.top_k if q.top_k else TOP_K
        retriever = vs.as_retriever(search_kwargs={"k": k})

        docs = retriever.get_relevant_documents(q.question)

        if not docs:
            return Answer(
                answer="Je n'ai trouvé aucun document pertinent pour répondre à votre question.",
                sources=[]
            )

        # Formatage amélioré du contexte
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "inconnu")
            context_parts.append(f"[Source {i}: {source}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        messages = [
            {
                "role": "system",
                "content": (
                    "Tu es un assistant de question-réponse précis et utile. "
                    "Réponds en français en te basant uniquement sur le contexte fourni. "
                    "Si l'information n'est pas dans le contexte, dis-le clairement. "
                    "Sois concis mais complet."
                )
            },
            {
                "role": "user",
                "content": f"CONTEXTE:\n{context}\n\nQUESTION: {q.question}"
            },
        ]

        resp = llm.invoke(messages)
        sources = [doc.metadata.get("source", "inconnu") for doc in docs]

        # Éliminer les sources dupliquées en préservant l'ordre
        unique_sources = list(dict.fromkeys(sources))

        return Answer(
            answer=resp.content,
            sources=unique_sources
        )

    except Exception as e:
        logger.error(f"Erreur lors de la requête QA: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur interne: {e}")
