import os
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress
from rich.table import Table

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import Chroma

console = Console()

INDEX_DIR = os.environ.get("INDEX_DIR", "index")
COLLECTION = os.environ.get("COLLECTION", "qa_local")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.environ.get("LLM_MODEL", "koesn/mistral-7b-instruct")
TOP_K = int(os.environ.get("TOP_K", "4"))

SYSTEM_PROMPT = (
    "Tu es un assistant de question-réponse concis et rigoureux. "
    "Réponds en français. Si l'information n'est pas dans le contexte, dis-le clairement. "
    "Cite les sources sous forme de liste à la fin."
)


def format_context(docs):
    """Format le contexte de manière plus lisible."""
    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "inconnu")
        # Tronquer le contenu s'il est trop long
        content = d.page_content
        if len(content) > 500:
            content = content[:500] + "..."
        lines.append(f"[DOC {i}] Source: {src}\n{content}")
    return "\n\n".join(lines)


def display_stats(db):
    """Affiche les statistiques de la base de données."""
    try:
        count = db._collection.count()
        table = Table(title="Statistiques de la base")
        table.add_column("Métrique", style="cyan")
        table.add_column("Valeur", style="green")

        table.add_row("Collection", COLLECTION)
        table.add_row("Documents indexés", str(count))
        table.add_row("Modèle d'embedding", EMBED_MODEL)
        table.add_row("Modèle LLM", LLM_MODEL)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Erreur lors de la récupération des stats: {e}[/red]")


def main():
    console.print(Panel(
        "[bold]QA Local v2.0[/bold] - Ctrl+C pour quitter\n"
        "Commandes spéciales:\n"
        "• /stats - Afficher les statistiques\n"
        "• /help - Aide\n"
        "• /clear - Effacer l'écran",
        subtitle="RAG + Ollama + Chroma"
    ))

    try:
        with Progress() as progress:
            task = progress.add_task("Initialisation...", total=3)

            embeddings = OllamaEmbeddings(model=EMBED_MODEL)
            progress.update(task, advance=1)

            db = Chroma(
                collection_name=COLLECTION,
                embedding_function=embeddings,
                persist_directory=INDEX_DIR,
            )
            progress.update(task, advance=1)

            retriever = db.as_retriever(search_kwargs={"k": TOP_K})
            llm = ChatOllama(
                model=LLM_MODEL,
                temperature=0.2,
                num_ctx=4096,
            )
            progress.update(task, advance=1)

    except Exception as e:
        console.print(f"[red]Erreur d'initialisation: {e}[/red]")
        return

    while True:
        try:
            question = console.input("[bold green]❓ [/bold green]")

            if not question.strip():
                continue

            # Commandes spéciales
            if question.lower() == "/stats":
                display_stats(db)
                continue
            elif question.lower() == "/help":
                console.print(Panel(
                    "Posez votre question et j'y répondrai en me basant sur les documents indexés.\n"
                    "Commandes:\n"
                    "• /stats - Statistiques de la base\n"
                    "• /clear - Effacer l'écran\n"
                    "• Ctrl+C - Quitter"
                ))
                continue
            elif question.lower() == "/clear":
                console.clear()
                continue

            with console.status("🤔 Recherche en cours...", spinner="dots"):
                docs = retriever.get_relevant_documents(question)

            if not docs:
                console.print("[yellow]Aucun document pertinent trouvé.[/yellow]")
                continue

            context = format_context(docs)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Réponds à la question en t'appuyant uniquement sur le CONTEXTE.\n"
                        "Si ce n'est pas suffisant, dis clairement ce qui manque.\n\n"
                        f"CONTEXTE:\n{context}\n\nQUESTION: {question}"
                    ),
                },
            ]

            with console.status("💭 Génération de la réponse...", spinner="dots"):
                resp = llm.invoke(messages)

            console.print(Panel(Markdown(str(resp.content)), title="Réponse"))

            # Afficher les sources de manière plus élégante
            sources_table = Table(title="Sources consultées")
            sources_table.add_column("N°", style="cyan", width=4)
            sources_table.add_column("Source", style="green")

            for i, d in enumerate(docs, 1):
                src = d.metadata.get("source", "inconnu")
                sources_table.add_row(str(i), src)

            console.print(sources_table)

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Au revoir ! 👋[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[red]Erreur: {e}[/red]")


if __name__ == "__main__":
    main()
