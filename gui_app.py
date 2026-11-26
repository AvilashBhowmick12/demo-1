"""Browser GUI for the resume shortlister.

Run from the project root (the folder that contains "data"):
    uv add flask        # once
    uv run python gui_app.py
then open http://127.0.0.1:5055 (it opens by itself).

It is the same engine as chat_app.py (src/shortlist.py + src/chat.py) with a web page on top:
chat box, ranked candidate cards, a drop-down to search only one part of the resumes
(experience, projects, skills, ...), links that open the original resume, and an upload box
that adds a new resume without rebuilding the index.
"""
import os
import threading
import webbrowser
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from chat_app import DATA_DIR, EMBEDDING_MODEL, RESUME_DIR, RESUME_TYPES
from src.chat import ChatShortlister
from src.sections import SEARCHABLE_SECTIONS
from src.shortlist import ResumeShortlister

GUI_DIR = Path(__file__).parent / "gui"
PORT = int(os.environ.get("PORT", "5055"))

# value sent by the page -> sections handed to the search (None = let the message decide)
SCOPES = {"auto": None, "all": [], "experience+projects": ["experience", "projects"],
          **{name: [name] for name in SEARCHABLE_SECTIONS}}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024     # uploads up to 25 MB
_lock = threading.Lock()                                  # one conversation, one request at a time
shortlister = None
chat = None


def _spec_json() -> dict:
    s = chat.spec
    return {"query": s.query, "must_have": s.must_have, "nice_to_have": s.nice_to_have,
            "min_years": s.min_years, "max_years": s.max_years, "strict": s.strict,
            "sections": s.sections, "top_n": s.top_n}


def _result_json(rank: int, r: dict) -> dict:
    return {
        "rank": rank,
        "name": chat.display_name(r),
        "placeholder": bool(r.get("name_placeholder")),
        "file": r["source_file"],
        "file_url": "/files/" + r["source_file"],
        "score": r["score"],
        "keyword_score": r["keyword_score"],
        "semantic_score": r["semantic_score"],
        "has": [{"skill": t.split("|")[0], "count": r["mentions"].get(t, 0)}
                for t in r["matched_must"] + r["matched_nice"]],
        "missing": [t.split("|")[0] for t in r["missing_must"]],
        "years": r.get("years"),
        "years_source": r.get("years_source"),
        "sections": r.get("sections") or [],
        "excerpt": r.get("excerpt") or [],
    }


def _payload(reply: str) -> dict:
    # the terminal reply ends with "(Search used: ... Type /table ...)"; the page shows the criteria in its side panel
    reply = reply.split("\n\n(Search used:")[0]
    out = {"kind": chat.last_kind, "reply": reply, "spec": _spec_json(),
           "candidates": len(shortlister.candidates)}
    if chat.last_kind == "search":
        out["summary"] = chat.summary_line() if chat.results else ""
        out["results"] = [_result_json(i, r) for i, r in enumerate(chat.results, 1)]
    return out


@app.get("/")
def index():
    return send_from_directory(GUI_DIR, "index.html")


@app.get("/api/status")
def status():
    return jsonify({"candidates": len(shortlister.candidates), "spec": _spec_json()})


@app.post("/api/chat")
def api_chat():
    body = request.get_json(silent=True) or {}
    message = str(body.get("message", "")).strip()
    if not message:
        return jsonify({"error": "empty message"}), 400
    scope = SCOPES.get(str(body.get("scope", "auto")), None)
    with _lock:
        reply = chat.handle(message, scope=scope)
        return jsonify(_payload(reply))


@app.post("/api/reset")
def api_reset():
    with _lock:
        chat.reset()
    return jsonify({"ok": True, "spec": _spec_json()})


@app.post("/api/upload")
def api_upload():
    """Save an uploaded resume into data/resume and add it to the index (seconds, no rebuild)."""
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no file"}), 400
    name = Path(f.filename).name                         # drop any folder part of the name
    if Path(name).suffix.lower() not in RESUME_TYPES:
        return jsonify({"error": f"Unsupported file type. Use one of: {', '.join(RESUME_TYPES)}"}), 400
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    dest = RESUME_DIR / name
    with _lock:
        f.save(dest)
        try:
            info = shortlister.add_resume(str(dest), data_dir=DATA_DIR)
        except Exception as e:                           # e.g. unreadable file, OCR not installed
            return jsonify({"error": f"Could not add the resume: {type(e).__name__}: {e}"}), 500
        out = {"info": info, "candidates": len(shortlister.candidates)}
        if info.get("status") == "added" and not chat.spec.is_empty():
            try:                                          # the old shortlist predates this resume: refresh it
                chat._do_search(chat.spec)
                chat.last_kind = "search"
                out["refreshed"] = _payload("Re-ran your last search with the new resume included.")
            except Exception as e:
                out["refresh_error"] = f"{type(e).__name__}: {e}"
    return jsonify(out)


@app.get("/files/<path:name>")
def files(name):
    return send_from_directory(RESUME_DIR.resolve(), name)   # refuses paths that leave the folder


def main():
    global shortlister, chat
    shortlister = ResumeShortlister(embedding_model=EMBEDDING_MODEL)
    shortlister.ensure(DATA_DIR)          # loads the saved index, or builds it on the very first run
    chat = ChatShortlister(shortlister)
    url = f"http://127.0.0.1:{PORT}"
    print(f"\nResume shortlister GUI ready: {url}   (Ctrl+C to stop)\n")
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=PORT, threaded=True)


if __name__ == "__main__":
    main()