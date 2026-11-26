from pathlib import Path

from src.shortlist import ResumeShortlister
from src.chat import ChatShortlister

DATA_DIR = "data"
RESUME_DIR = Path(DATA_DIR) / "resume"   # run this script from the project root (the folder that contains "data")

HELP = """Describe the role and skills you want, then keep refining. Examples:
  I need a DevOps engineer with AWS and Terraform, bonus for Kubernetes
  only those with Docker
  add python as a must-have
  show 20 results
  why is #2 ranked above #3?
  compare the top two
Commands:
  /add <file>   add ONE new resume without rebuilding (file name inside data\\resume, or a full path)
  /table        show the last search as a plain ranked list with all scores
  /reset        start over
  /help         show this help
  exit          quit"""

# Multilingual model: resumes in French, Russian etc. are matched properly (all-MiniLM-L6-v2 is English-only).
# Changing this value makes the index rebuild automatically on the next start (a few minutes).
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


RESUME_TYPES = (".pdf", ".docx", ".txt")

def find_resume_file(arg: str):
    """Accept a full path or a file name in data/resume; the extension is optional."""
    arg = arg.strip().strip('"').strip("'")
    if not arg:
        return None
    for base in (Path(arg), RESUME_DIR / arg):
        if base.is_file():
            return base
        for ext in RESUME_TYPES:
            alt = base.with_name(base.name + ext)
            if alt.is_file():
                return alt
    return None


def describe_added(info: dict) -> str:
    if info.get("status") != "added":
        return f"  {info.get('candidate_id', '?')}: {info.get('status')}"
    years = "unknown" if info.get("years") is None else f"~{info['years']:g}"
    return (f"  Added {info['candidate_id']} | name: {info.get('name') or '(not found)'} | "
            f"experience: {years} yrs | {info['chunks']} chunks | read via: {info.get('read_via', '?')}")


def handle_add(shortlister, chat, arg: str) -> None:
    if not hasattr(shortlister, "add_resume"):
        print("add_resume() is missing from src/shortlist.py - apply the patches first.\n")
        return
    path = find_resume_file(arg)
    if path is None:
        print(f"File not found: {arg!r}. Put it in {RESUME_DIR} and use /add <file name>, or give the full path.\n")
        return
    try:
        info = shortlister.add_resume(str(path), data_dir=DATA_DIR)
    except Exception as e:
        print(f"Could not add the resume: {type(e).__name__}: {e}\n")
        return
    print(describe_added(info))
    if info.get("status") != "added":
        print()
        return
    if chat.spec.is_empty():
        print("Done. Now describe the role you are hiring for.\n")
        return
    # The earlier shortlist was built before this resume existed, so run the same search again.
    print("\nRe-running your last search with the new resume included:\n")
    try:
        print(chat._do_search(chat.spec) + "\n")
    except Exception as e:
        print(f"Added, but re-running the search failed ({type(e).__name__}: {e}). Ask your question again.\n")


if __name__ == "__main__":
    shortlister = ResumeShortlister(embedding_model=EMBEDDING_MODEL)
    shortlister.ensure(DATA_DIR)   # loads the saved index, or builds it on the first run
    chat = ChatShortlister(shortlister)

    print("\nResume shortlister ready.\n" + HELP + "\n")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in ("exit", "quit"):
            break
        if message == "/help":
            print(HELP)
            continue
        if message == "/reset":
            chat.reset()
            print("Started over.")
            continue
        if message == "/table":
            print("\n" + (chat.last_table or "No search yet.") + "\n")
            continue
        if message.lower() == "/add" or message.lower().startswith("/add "):
            handle_add(shortlister, chat, message[4:])
            continue
        if message:
            print("\n" + chat.handle(message) + "\n")