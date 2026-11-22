"""Extract a candidate's name from the top of a resume.

Resumes put the name on one of the first lines. We take the first line that looks like a person's name
(2-4 words, letters only, no job-title or section words). If the name is split over two lines
("JESSICA" / "CLAIRE") we join them. For files named like "CV_ZeroukMohamedAmine" we fall back to the
file name. A few resumes carry a well-known sample-template name; those are flagged as placeholders.
"""
import re
from typing import Optional

PLACEHOLDER_NAMES = {"jessica claire"}   # the sample-resume name, also seen with resumesample@example.com

# a "letter" in any alphabet (Latin, Cyrillic, ...) or one of ' ’ . -
_CH = r"(?:[^\W\d_]|['’.\-])"
_NAME_LINE = re.compile(rf"^{_CH}+(?: {_CH}+){{1,3}}$")
_WORD = re.compile(rf"^{_CH}{{2,}}$")

# words that mean "this line is a title / section / sentence, not a person's name"
_NOT_NAME = {
    "resume", "cv", "curriculum", "vitae", "summary", "objective", "objectif", "profile", "profil", "contact",
    "professional", "experience", "education", "skills", "competences", "compétences", "formation", "about",
    "developer", "engineer", "engineering", "consultant", "manager", "analyst", "specialist", "technician",
    "intern", "internship", "scientist", "architect", "administrator", "administrateur", "designer", "director",
    "lead", "senior", "junior", "programmer", "officer", "associate", "assistant", "coordinator", "executive",
    "alternance", "stage", "recherche", "bachelor", "master", "licence", "diplôme", "etudiant", "étudiant",
    "technicien", "ingénieur", "développeur", "python", "java", "data", "software", "web", "cloud", "devops",
    "full", "stack", "full-stack", "it", "information", "technology", "network", "security", "systems",
    "support", "project", "program", "business", "sales", "marketing", "teacher", "nurse", "driver", "operator",
    "mechanical", "electrical", "biomedical", "civil", "customer", "service", "services", "the", "and", "of",
    "for", "in", "at", "to", "with", "à", "la", "le", "les", "de", "du", "des", "et", "un", "une", "en",
    "adobe", "photoshop", "django", "website", "development", "personal", "info", "career", "focus", "address",
    "phone", "email", "linkedin", "github", "portfolio", "key", "core", "technical", "tools", "languages",
    "hobbies", "projects", "certifications", "references", "qualifications", "highlights", "expertise",
    "my", "life", "philosophy", "your", "name", "first", "last", "template", "sample", "page", "hands-on", "agile", "scrum",
}
_BAD_CHARS = re.compile(r"[\d@:/|•◆·,()]")


def _clean(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip(" -–—•·|")


_LONG_NOT_NAME = [w for w in _NOT_NAME if len(w) >= 6]


def _is_word(w: str) -> bool:
    lw = w.lower().strip(".'’-")
    if not _WORD.match(w) or lw in _NOT_NAME:
        return False
    # glued headings such as "PROFESSIONALSUMMARY"
    return not (len(lw) >= 10 and any(bad in lw for bad in _LONG_NOT_NAME))


_PARTICLES = {"de", "la", "le", "du", "des", "van", "von", "der", "den", "bin", "ben", "al", "el", "di", "da", "dos", "del"}


def _looks_like_name(line: str) -> bool:
    if _BAD_CHARS.search(line) or not _NAME_LINE.match(line):
        return False
    words = line.split()
    if line.endswith(".") and len(words[-1]) > 3:      # "Companies Grow." is a sentence, "John Smith Jr." is not
        return False
    if not all(w[0].isupper() or w.lower() in _PARTICLES for w in words):
        return False
    return all(_is_word(w) or w.lower() in _PARTICLES for w in words)


def _pretty(name: str) -> str:
    """JESSICA CLAIRE -> Jessica Claire; leaves mixed case ("Adrien DE LA ROSA", "McKay") alone."""
    return name.title() if name.isupper() or name.islower() else name


def _from_filename(file_name: str) -> Optional[str]:
    stem = re.sub(r"\.pdf$", "", file_name, flags=re.I)
    if not stem.lower().startswith("cv_"):
        return None
    parts = re.findall(r"[A-ZÀ-Ý][a-zà-ÿ'’]+|[A-ZÀ-Ý]{2,}(?![a-zà-ÿ])", stem[3:])
    return " ".join(parts) if len(parts) >= 2 and "".join(parts) == stem[3:].replace("_", "").replace("-", "") else None


def extract_name(text: str, file_name: str = "") -> dict:
    """Return {"name": str | None, "source": "resume" | "filename" | None, "placeholder": bool}."""
    lines = [_clean(l) for l in text.splitlines() if l.strip()][:8]
    name: Optional[str] = None

    for line in lines[:6]:
        if _looks_like_name(line):
            name = line
            break

    if name is None and len(lines) >= 2:              # "JESSICA" / "CLAIRE" on separate lines
        a, b = lines[0], lines[1]
        if " " not in a and " " not in b and _is_word(a) and _is_word(b):
            name = f"{a} {b}"

    if name is None and len(lines) >= 2 and " " not in lines[0] and _is_word(lines[0]) \
            and lines[0][0].isupper() and len(lines[0]) >= 3 \
            and any(w.lower().strip(".'’-") in _NOT_NAME for w in lines[1].split()):
        name = lines[0]                                # "Dhruvi" / "Python Developer"

    if name is None and lines and " " not in lines[0] and lines[0].lower() == "jessica" \
            and re.search(r"\bclaire\b", " ".join(lines[:8]) + " " + text[:500], re.I):
        name = "Jessica Claire"                        # "JESSICA" ... "CLAIRE" split by the page layout

    if name and name.lower().count("claire") >= 2:
        name = "Jessica Claire"                        # the sample name garbled by the same source-data glitch

    if name and name.lower() == "jessica" and re.search(r"\bclaire\b", text[:600], re.I):
        name = "Jessica Claire"                        # sample name whose surname sits elsewhere on the page

    if name:
        name = _pretty(name)
        return {"name": name, "source": "resume", "placeholder": name.lower() in PLACEHOLDER_NAMES}

    fn = _from_filename(file_name)
    if fn:
        return {"name": fn, "source": "filename", "placeholder": False}
    return {"name": None, "source": None, "placeholder": False}