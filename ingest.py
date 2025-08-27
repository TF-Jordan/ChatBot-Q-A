import os
from pathlib import Path
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.console import Console

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

from utils import load_documents, split_documents, validate_environment_vars

console = Console()

DATA_DIR = os.environ.get("DATA_DIR", "data")
INDEX_DIR = os.environ.get("INDEX_DIR", "index")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
COLLECTION = os.environ.get("COLLECTION", "qa_local")

def main():
    console.print("[bold cyan]🚀 Début de l'ingestion des documents[/bold cyan]")
    
    # Validation de l'environnement
    validate_environment_vars()
    
    # Vérifier que le répertoire de données existe
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        console.print(f"[red]❌ Le répertoire {DATA_DIR} n'existe pas. Création...[/red]")
        data_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[yellow]📁 Ajoutez des fichiers .pdf, .txt ou .md dans {DATA_DIR} et relancez.[/yellow]")
        return
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        
        # Étape 1: Chargement des documents
        task1 = progress.add_task("📖 Chargement des documents...", total=None)
        docs = load_documents(DATA_DIR)
        progress.update(task1, completed=True)
        
        if not docs:
            console.print(f"[yellow]⚠️  Aucun document trouvé dans {DATA_DIR}.[/yellow]")
            console.print(f"[yellow]📝 Formats supportés: .pdf, .txt, .md[/yellow]")
            return
        
        console.print(f"[green]✅ {len(docs)} documents chargés[/green]")
        
        # Étape 2: Division en chunks
        task2 = progress.add_task("✂️  Division en chunks...", total=None)
        chunks = split_documents(docs)
        progress.update(task2, completed=True)
        
        console.print(f"[green]✅ {len(chunks)} chunks créés[/green]")
        
        # Étape 3: Création des embeddings
        task3 = progress.add_task("🧠 Initialisation des embeddings...", total=None)
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        progress.update(task3, completed=True)
        
        # Étape 4: Indexation
        task4 = progress.add_task("📚 Indexation dans Chroma...", total=None)
        
        # Créer ou mettre à jour l'index
        db = Chroma(
            collection_name=COLLECTION,
            embedding_function=embeddings,
            persist_directory=INDEX_DIR,
        )
        
        # Générer des IDs uniques basés sur le contenu
        ids = [f"chunk_{hash(chunk.page_content + str(chunk.metadata))}_{i}" 
               for i, chunk in enumerate(chunks)]
        
        # Ajout par batch pour éviter les problèmes de mémoire
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            db.add_documents(batch_chunks, ids=batch_ids)
            
            progress.update(task4, 
                          description=f"📚 Indexation... {min(i+batch_size, len(chunks))}/{len(chunks)}")
        
        progress.update(task4, completed=True)
    
    # Affichage final des statistiques
    try:
        final_count = db._collection.count()
        console.print(f"[bold green]🎉 Indexation terminée![/bold green]")
        console.print(f"[green]📊 {final_count} documents dans la collection '{COLLECTION}'[/green]")
        console.print(f"[green]📁 Index sauvegardé dans: {INDEX_DIR}[/green]")
    except Exception as e:
        console.print(f"[yellow]⚠️  Impossible de récupérer les stats finales: {e}[/yellow]")

if __name__ == "__main__":
    main()