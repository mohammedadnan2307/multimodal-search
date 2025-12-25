"""Command-line interface for multimodal search."""

import argparse
import sys

from src.search import SearchEngine, SearchMode


def cmd_index(args):
    print("Building search index...")
    engine = SearchEngine()
    stats = engine.index_documents(use_clip=not args.demo)
    
    if "error" in stats:
        print(f"Error: {stats['error']}")
        return 1
    
    print(f"\nIndexing complete:")
    print(f"  • {stats['total_items']} items indexed")
    print(f"  • {stats['text_items']} text chunks")
    print(f"  • {stats['image_items']} images")
    print(f"  • Embedder: {stats['embedder']}")
    return 0


def cmd_search(args):
    engine = SearchEngine()
    
    if not engine.store:
        print("No index found. Run 'python cli.py index' first.")
        return 1
    
    mode_map = {
        "all": SearchMode.ALL,
        "text": SearchMode.TEXT,
        "image": SearchMode.IMAGE
    }
    mode = mode_map.get(args.mode, SearchMode.ALL)
    
    results = engine.search(args.query, top_k=args.top_k, mode=mode)
    
    if not results:
        print("No results found.")
        return 0
    
    print(f"\nResults for: '{args.query}'\n")
    for i, r in enumerate(results, 1):
        print(f"{i}. [{r.score:.2%}] [{r.modality}] {r.document} (p.{r.page})")
        if r.modality == "text":
            preview = r.content[:100] + "..." if len(r.content) > 100 else r.content
            print(f"   {preview}")
        print()
    
    return 0


def cmd_serve(args):
    from app import app
    print(f"Starting server at http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)


def cmd_stats(args):
    engine = SearchEngine()
    
    if not engine.store:
        print("No index found.")
        return 1
    
    stats = engine.store.stats()
    print("\nIndex Statistics:")
    print(f"  • Total items: {stats['total_items']}")
    print(f"  • Text chunks: {stats['text_items']}")
    print(f"  • Images: {stats['image_items']}")
    print(f"  • Documents: {stats['documents']}")
    print(f"  • Embedder: {stats['embedder']}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Multimodal semantic search engine",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Index command
    index_parser = subparsers.add_parser("index", help="Build search index from PDFs")
    index_parser.add_argument("--demo", action="store_true", help="Use demo embedder (no GPU)")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the index")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("-k", "--top-k", type=int, default=5, help="Number of results")
    search_parser.add_argument("-m", "--mode", choices=["all", "text", "image"], default="all")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start web server")
    serve_parser.add_argument("-p", "--port", type=int, default=5000, help="Port number")
    serve_parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    # Stats command
    subparsers.add_parser("stats", help="Show index statistics")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    commands = {
        "index": cmd_index,
        "search": cmd_search,
        "serve": cmd_serve,
        "stats": cmd_stats
    }
    
    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
