"""Split a resume's text into sections (summary, skills, experience, projects, ...).

A section starts at a heading line such as "Work History" or "SKILLS:" and runs until the next
heading. Headings are recognised from a list of common names (English and French). OCR often
splits a heading over two lines ("Education and" / "Training"), so two short lines are also
tried together.

Searchable sections (what a recruiter can ask for):
    summary, skills, experience, projects, education, certifications, accomplishments
Headings such as "Languages" or "Interests" are recognised only so that they END the previous
section; their text is kept under "other".
"""
import re
from typing import Dict, Iterable, List, Optional

SEARCHABLE_SECTIONS = ["summary", "skills", "experience", "projects", "education",
                       "certifications", "accomplishments"]

_ALIASES: Dict[str, List[str]] = {
    "summary": ["summary", "professional summary", "career summary", "summary of qualifications",
                "objective", "resume objective", "career objective", "professional objective", "career focus",
                "profile", "professional profile", "about me", "personal statement", "profil"],
    "skills": ["skills", "technical skills", "skill highlights", "key skills", "core skills",
               "core competencies", "areas of expertise", "technical expertise", "highlights",
               "qualifications", "skills and abilities", "technologies",
               "competences", "competence", "competences cles"],
    "experience": ["experience", "work history", "work experience", "professional experience",
                   "employment history", "employment", "career history", "relevant experience",
                   "professional background", "work and experience",
                   "experience professionnelle", "experiences professionnelles",
                   "parcours professionnel"],
    "projects": ["projects", "project experience", "personal projects", "key projects",
                 "academic projects", "selected projects", "projets"],
    "education": ["education", "education and training", "education training", "academic background",
                  "academic qualifications", "training", "trainings", "formation", "formations",
                  "education et formation"],
    "certifications": ["certifications", "certification", "certificates", "certificate",
                       "licenses", "licenses and certifications", "certifications and licenses",
                       "certifications and training", "courses"],
    "accomplishments": ["accomplishments", "achievements", "key achievements", "awards",
                        "awards and honors", "honors and awards"],
    # recognised only to end the section before them
    "other": ["languages", "langues", "interests", "hobbies", "interets", "affiliations",
              "professional affiliations", "activities and honors", "additional information",
              "links", "websites portfolios profiles", "contact", "references", "publications",
              "volunteer experience", "volunteering"],
}

_HEADING = {alias: sec for sec, aliases in _ALIASES.items() for alias in aliases}
_MAX_HEADING_CHARS = 45


def _norm(line: str) -> str:
    """'  WORK HISTORY: ' -> 'work history'; accents are folded so French headings match."""
    s = line.lower().strip()
    s = s.replace("&", " and ").replace("é", "e").replace("è", "e").replace("ê", "e").replace("’", "'")
    s = re.sub(r"[^a-z0-9' ]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _heading_of(line: str) -> Optional[str]:
    s = line.strip().lstrip("•●▪-*·​ ").strip()
    if not s or len(s) > _MAX_HEADING_CHARS:
        return None
    return _HEADING.get(_norm(s))


def split_sections(text: str) -> Dict[str, str]:
    """Return {section name: text}. Text before the first heading is "header" (name, contact).

    A section that appears twice (e.g. 'Summary' and later 'Professional Summary') is joined.
    """
    lines = text.split("\n")
    sections: Dict[str, List[str]] = {}
    current = "header"
    i = 0
    while i < len(lines):
        line = lines[i]
        sec = _heading_of(line)
        consumed = 1
        if sec is None and i + 1 < len(lines) and line.strip() and lines[i + 1].strip():
            # OCR split heading: "Education and" + "Training"
            pair = line.strip() + " " + lines[i + 1].strip()
            if len(line.strip()) <= 20 and len(lines[i + 1].strip()) <= 20:
                sec = _heading_of(pair)
                if sec is not None:
                    consumed = 2
        if sec is not None:
            current = sec
            sections.setdefault(current, [])
            i += consumed
            continue
        sections.setdefault(current, []).append(line)
        i += 1
    return {k: "\n".join(v).strip() for k, v in sections.items() if "\n".join(v).strip()}


def section_text(sections: Dict[str, str], wanted: Iterable[str]) -> str:
    """Join the requested sections (in the order asked); '' if the resume has none of them."""
    return "\n".join(sections[w] for w in wanted if w in sections)


def normalise_section_names(names: Iterable[str]) -> List[str]:
    """Map loose names an LLM or user might give ('work experience', 'certs') to canonical ones."""
    out: List[str] = []
    for n in names or []:
        n = str(n).strip().lower()
        if not n or n in ("all", "any", "whole", "full", "none", "null", "anywhere"):
            continue
        sec = _HEADING.get(_norm(n)) or ({"cert": "certifications", "certs": "certifications",
                                          "work": "experience", "jobs": "experience",
                                          "skill": "skills", "project": "projects"}.get(n))
        if sec in SEARCHABLE_SECTIONS and sec not in out:
            out.append(sec)
    return out