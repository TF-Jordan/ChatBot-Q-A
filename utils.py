from pathlib import Path
from typing import List 
import logging
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.schema import Document

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SUPPORTED_EXTS = {".pdf", ".txt", ".md"}

def load_documents(data_dir: str) -> List[Document]:
    """Charge les documents depuis le répertoire spécifié."""
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Le répertoire {data_dir} n'existe pas.")
        return []
    
    docs: List[Document] = []
    
    for path in sorted(data_path.rglob("*")):
        if not path.is_file():
            continue
            
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            continue
            
        try:
            logger.info(f"Chargement de: {path}")
            
            if ext == ".pdf":
                loader = PyPDFLoader(str(path))
            elif ext in [".txt", ".md"]:
                loader = TextLoader(str(path), encoding='utf-8')
            
            loaded_docs = loader.load()
            
            # Ajouter le nom du fichier aux métadonnées
            for doc in loaded_docs:
                doc.metadata["source"] = str(path.name)
                doc.metadata["full_path"] = str(path)
                
            docs.extend(loaded_docs)
            
        except Exception as e:
            logger.error(f"Erreur lors du chargement de {path}: {e}")
            
    return docs

def split_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> List[Document]:
    """Divise les documents en chunks plus petits."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]
    )
    return splitter.split_documents(docs)

def validate_environment_vars():
    """Valide que les variables d'environnement nécessaires sont définies."""
    required_vars = ["EMBED_MODEL", "LLM_MODEL"]
    missing_vars = []
    
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"Variables d'environnement manquantes (valeurs par défaut utilisées): {missing_vars}")
