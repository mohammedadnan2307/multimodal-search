import base64
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from src.search import SearchEngine, SearchMode

import config

app = Flask(__name__, static_folder="static")
engine = SearchEngine()


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/api/search", methods=["POST"])
def search():
    data = request.json
    query = data.get("query", "")
    top_k = data.get("top_k", config.DEFAULT_TOP_K)
    mode = data.get("mode", "all")
    
    if not query:
        return jsonify({"error": "Query required"}), 400
    
    # Map mode string to enum
    mode_map = {
        "all": SearchMode.ALL,
        "text": SearchMode.TEXT,
        "image": SearchMode.IMAGE
    }
    search_mode = mode_map.get(mode, SearchMode.ALL)
    
    results = engine.search(query, top_k=top_k, mode=search_mode)
    
    # Format response
    formatted = []
    for r in results:
        item = {
            "id": r.id,
            "score": round(r.score, 4),
            "modality": r.modality,
            "document": r.document,
            "page": r.page,
        }
        
        if r.modality == "text":
            item["content"] = r.content
        else:
            try:
                img_path = Path(r.content)
                if img_path.exists():
                    with open(img_path, "rb") as f:
                        item["image"] = base64.b64encode(f.read()).decode()
            except Exception:
                pass
        
        formatted.append(item)
    
    return jsonify({"results": formatted, "query": query})


@app.route("/api/stats")
def stats():
    if engine.store:
        return jsonify(engine.store.stats())
    return jsonify({"error": "No index loaded"})


@app.route("/api/index", methods=["POST"])
def index_documents():
    data = request.json or {}
    use_clip = data.get("use_clip", True)
    
    result = engine.index_documents(use_clip=use_clip)
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
