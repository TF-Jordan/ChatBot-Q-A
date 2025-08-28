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
    """
    Charge et retourne les documents depuis un répertoire donné, en prenant en
    charge certains formats spécifiques (.pdf, .txt, .md).

    Cette fonction parcourt récursivement tous les fichiers du dossier `data_dir`,
    identifie ceux dont l’extension est supportée, puis utilise le loader adapté
    (par exemple `PyPDFLoader` pour les PDF, `TextLoader` pour les fichiers texte).
    Les documents chargés se voient enrichis de métadonnées indiquant le nom du fichier
    source et son chemin complet.

    Args:
        data_dir (str):
            Chemin du répertoire contenant les documents à charger.

    Returns:
        List[Document]:
            Liste d’objets `Document` (LangChain), chacun correspondant au contenu
            d’un fichier ou d’une portion de fichier, avec des métadonnées
            associées (`source`, `full_path`).

    Exemple:
        >>> docs = load_documents("data/")
        >>> len(docs)
        12
        >>> docs[0].metadata
        {'source': 'rapport.pdf', 'full_path': 'data/rapport.pdf'}

    Notes:
        - Les extensions supportées doivent être définies dans `SUPPORTED_EXTS`,
          par ex. :
          ```python
          SUPPORTED_EXTS = [".pdf", ".txt", ".md"]
          ```
        - Les logs permettent de suivre le chargement des fichiers ou d’identifier
          les erreurs (fichiers corrompus, permissions, etc.).
        - Les fichiers ignorés (non supportés ou non lisibles) ne bloquent pas
          l’exécution : ils sont simplement sautés.
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        logger.error(f"Le répertoire {data_dir} n'existe pas.")
        return []

    docs: List[Document] = []

    for path in sorted(data_path.rglob("*")):
        if not path.is_file():
            continue

        ext = path.suffix.lower()  # Récupère l’extension du fichier en minuscule
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
    """
    Divise une liste de documents en sous-documents (chunks) plus petits afin de faciliter
    leur traitement par un modèle de langage ou une base vectorielle.

    Cette fonction utilise `RecursiveCharacterTextSplitter` de LangChain, qui coupe
    les textes de manière récursive en utilisant une liste de séparateurs hiérarchiques
    (par ex. paragraphes, phrases, espaces), jusqu’à obtenir des morceaux de taille
    inférieure ou égale à `chunk_size`.

    Args:
        docs (List[Document]):
            La liste des documents à découper. Chaque `Document` contient
            généralement un champ `page_content` (texte brut) et éventuellement
            des métadonnées.
        chunk_size (int, optionnel):
            Taille maximale (en caractères) d’un chunk.
            Par défaut : 800.
        chunk_overlap (int, optionnel):
            Nombre de caractères qui chevauchent deux chunks consécutifs.
            Cela permet de conserver du contexte quand une phrase ou une idée
            est coupée entre deux morceaux. Par défaut : 120.

    Returns:
        List[Document]:
            Une nouvelle liste de `Document` où chaque document original
            est remplacé par un ou plusieurs chunks plus petits,
            prêts à être indexés ou utilisés dans une tâche NLP.

    Exemple:
        >>> docs = [Document(page_content="Le Cameroun est un pays d’Afrique centrale. Sa capitale est Yaoundé.")]
        >>> split_docs = split_documents(docs, chunk_size=30, chunk_overlap=5)
        >>> [d.page_content for d in split_docs]
        ['Le Cameroun est un pays d’Afr',
         'rique centrale. Sa capitale es',
         't Yaoundé.']

    Notes:
        - Les séparateurs utilisés par défaut sont :
          ["\\n\\n", "\\n", ".", "!", "?", " ", ""]
        - L’ordre est important : le splitter tente de couper d’abord sur les
          paragraphes, puis sur les phrases, puis les espaces, et en dernier
          recours caractère par caractère.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", "!", "?", " ", ""]  # les séparateurs pour a utiliser pour delimiter les chunks
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
