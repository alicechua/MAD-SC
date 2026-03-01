"""Offline pre-processing pipeline: COHA Linear Text → SQLite FTS5 index.

Run this script **once** after downloading the COHA Linear Text files.
It builds ``coha_corpus.db``, which the MAD-SC data loader queries at runtime.

Usage
-----
    python scripts/build_coha_index.py [options]

Options
-------
    --data-dir DIR      Directory containing COHA files (default: data/coha)
    --db PATH           Output SQLite database path  (default: coha_corpus.db)
    --batch-size N      Rows committed per transaction (default: 5000)
    --reset             Drop and recreate the database before indexing

Supported input formats
-----------------------
1.  BYU ``@@textid`` linear text
        @@fic_1810_1
        word1 word2 word3 . word4 word5 .
        @@fic_1810_2
        ...

2.  XML ``<text id="..." year="...">`` blocks
        <text id="fic_1960_42" year="1960">
        word1 word2 . word3 word4 .
        </text>

3.  CSV / TSV exports from english-corpora.org
        ##,YEAR,GENRE,TEXT
        1,1960,FIC,"context window with target word"

The script scans ``--data-dir`` recursively for ``.txt``, ``.csv``, and
``.tsv`` files.  Year is extracted from the text ID, XML attributes, or
the filename (whichever comes first).

Tokenization fixes applied to linear text
------------------------------------------
BYU linear text is pre-tokenised with spaces before punctuation and split
contractions.  This script reverses those transformations before sentence
segmentation:
    "do n't go ."  →  "Don't go."
    "John 's cat ." →  "John's cat."
    "Hello , world ." → "Hello, world."
"""

from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# NLTK sentence tokeniser — downloaded on first run if absent
# ---------------------------------------------------------------------------

def _ensure_nltk() -> None:
    import nltk  # noqa: PLC0415
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
            return
        except LookupError:
            pass
    # Neither found — download punkt_tab (NLTK ≥ 3.9) with punkt as fallback
    for resource in ("punkt_tab", "punkt"):
        try:
            nltk.download(resource, quiet=True)
            return
        except Exception:
            continue


def _sent_tokenize(text: str) -> list[str]:
    from nltk.tokenize import sent_tokenize  # noqa: PLC0415
    return sent_tokenize(text)


# ---------------------------------------------------------------------------
# Database schema
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS sentences (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    year          INTEGER NOT NULL,
    text_id       TEXT    NOT NULL,
    sentence_text TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_year ON sentences(year);

CREATE VIRTUAL TABLE IF NOT EXISTS sentences_fts
    USING fts5(
        sentence_text,
        content     = sentences,
        content_rowid = id,
        tokenize    = 'unicode61 remove_diacritics 1'
    );

CREATE TRIGGER IF NOT EXISTS sentences_ai
    AFTER INSERT ON sentences
BEGIN
    INSERT INTO sentences_fts(rowid, sentence_text)
    VALUES (new.id, new.sentence_text);
END;

CREATE TRIGGER IF NOT EXISTS sentences_ad
    AFTER DELETE ON sentences
BEGIN
    INSERT INTO sentences_fts(sentences_fts, rowid, sentence_text)
    VALUES ('delete', old.id, old.sentence_text);
END;
"""

_DROP = """
DROP TABLE  IF EXISTS sentences;
DROP TABLE  IF EXISTS sentences_fts;
DROP TRIGGER IF EXISTS sentences_ai;
DROP TRIGGER IF EXISTS sentences_ad;
"""


def _create_schema(conn: sqlite3.Connection, reset: bool) -> None:
    if reset:
        for stmt in _DROP.strip().split(";"):
            if stmt.strip():
                conn.execute(stmt)
        conn.commit()
    for stmt in _DDL.strip().split(";"):
        if stmt.strip():
            conn.execute(stmt)
    conn.commit()


# ---------------------------------------------------------------------------
# Year extraction
# ---------------------------------------------------------------------------

_YEAR_RE = re.compile(r"(?<!\d)(1[89]\d{2}|20[01]\d)(?!\d)")  # 1800–2019


def _extract_year(text_id: str, filename: Path) -> int | None:
    """Return the first plausible year found in text_id, then in filename."""
    for source in (text_id, filename.stem, str(filename)):
        m = _YEAR_RE.search(source)
        if m:
            y = int(m.group())
            if 1800 <= y <= 2019:
                return y
    return None


# ---------------------------------------------------------------------------
# Tokenisation repair
# ---------------------------------------------------------------------------

# Order matters: run contraction fixes before space-before-punctuation.
_CONTRACTION_FIXES: list[tuple[re.Pattern, str]] = [
    # n't  ("do n't" → "don't")
    (re.compile(r"\b(n)\s+(\'t)\b",     re.IGNORECASE), r"\1\2"),
    # Modal contractions — sha/wo special cases
    (re.compile(r"\b(sha)\s+(ll)\b",    re.IGNORECASE), r"shall"),
    (re.compile(r"\b(wo)\s+(n\'t)\b",   re.IGNORECASE), r"won't"),
    # Clitic forms: 've, 're, 'll, 'd, 'm, 's
    (re.compile(r"\s+(\'ve|\'re|\'ll|\'d|\'m|\'s)\b", re.IGNORECASE), r"\1"),
]

_SPACE_BEFORE_PUNCT = re.compile(r"\s+([.,;:!?)\]>»])")
_OPEN_QUOTE         = re.compile(r"``\s*")
_CLOSE_QUOTE        = re.compile(r"\s*''")
_XML_TAGS           = re.compile(r"<[^>]+>")
_SGML_MARKERS       = re.compile(r"^@@\S+\s*", re.MULTILINE)   # @@ textid lines
_WHITESPACE         = re.compile(r"\s+")


def _fix_tokenisation(raw: str) -> str:
    """Reconstruct natural prose from BYU pre-tokenised linear text."""
    text = _XML_TAGS.sub(" ", raw)          # strip XML/SGML tags
    text = _SGML_MARKERS.sub(" ", text)     # strip @@ text-ID markers
    for pattern, repl in _CONTRACTION_FIXES:
        text = pattern.sub(repl, text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _OPEN_QUOTE.sub('"', text)
    text = _CLOSE_QUOTE.sub('"', text)
    text = _WHITESPACE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# File parsers — yield (year, text_id, raw_block) tuples
# ---------------------------------------------------------------------------

def _parse_xml(content: str, path: Path):
    """Format 2: <text id="..." year="..."> blocks."""
    for attrs, body in re.findall(
        r"<text\b([^>]*)>(.*?)</text>", content, re.DOTALL | re.IGNORECASE
    ):
        year_attr = re.search(r'year=["\']?(\d{4})["\']?', attrs)
        id_attr   = re.search(r'id=["\']([^"\']+)["\']',   attrs)
        text_id   = id_attr.group(1) if id_attr else path.stem
        year      = (int(year_attr.group(1)) if year_attr
                     else _extract_year(text_id, path))
        if year:
            yield year, text_id, body


def _parse_at_markers(content: str, path: Path):
    """Format 1: @@textid blocks."""
    blocks = re.split(r"^@@", content, flags=re.MULTILINE)
    for block in blocks:
        if not block.strip():
            continue
        first_line, _, body = block.partition("\n")
        text_id = first_line.strip() or path.stem
        year    = _extract_year(text_id, path)
        if year:
            yield year, text_id, body


def _parse_txt(path: Path):
    """Dispatch to the correct text parser, falling back to whole-file."""
    try:
        content = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return

    if re.search(r"<text\b", content, re.IGNORECASE):
        yield from _parse_xml(content, path)
    elif re.search(r"^@@", content, re.MULTILINE):
        yield from _parse_at_markers(content, path)
    else:
        # Format 3: whole file = one corpus chunk, year from filename
        year = _extract_year("", path)
        if year:
            yield year, path.stem, content


def _parse_csv(path: Path, sep: str = ","):
    """CSV / TSV export from english-corpora.org."""
    _YEAR_COLS = {"year", "##year", "yr"}
    _TEXT_COLS = {"text", "context", "sentence", "kwic", "concordance"}
    try:
        with open(path, encoding="utf-8", errors="ignore", newline="") as fh:
            reader = csv.DictReader(fh, delimiter=sep)
            if not reader.fieldnames:
                return
            lmap = {f.lower().lstrip("#").strip(): f for f in reader.fieldnames}
            year_col = next((lmap[k] for k in _YEAR_COLS if k in lmap), None)
            text_col = next((lmap[k] for k in _TEXT_COLS if k in lmap), None)
            if not year_col or not text_col:
                return
            for row in reader:
                try:
                    year = int(str(row[year_col]).strip()[:4])
                except (ValueError, TypeError):
                    continue
                if not (1800 <= year <= 2019):
                    continue
                text_id = str(row.get("##", row.get("id", path.stem))).strip()
                yield year, text_id, row[text_col]
    except OSError:
        return


# ---------------------------------------------------------------------------
# Sentence extraction from a raw block
# ---------------------------------------------------------------------------

_MIN_TOKENS = 5   # discard very short fragments


def _extract_sentences(raw: str, is_csv_row: bool = False) -> list[str]:
    """Clean raw block and return a list of well-formed sentences."""
    cleaned = raw if is_csv_row else _fix_tokenisation(raw)
    # Strip residual XML / @@ markers that slipped through
    cleaned = _XML_TAGS.sub(" ", cleaned)
    cleaned = _SGML_MARKERS.sub(" ", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if not cleaned:
        return []
    sentences = _sent_tokenize(cleaned)
    return [s.strip() for s in sentences if len(s.split()) >= _MIN_TOKENS]


# ---------------------------------------------------------------------------
# Per-file processing
# ---------------------------------------------------------------------------

def _process_file(
    path: Path,
    conn: sqlite3.Connection,
    batch_size: int,
) -> int:
    """Parse *path*, segment sentences, insert into DB.  Returns row count."""
    is_csv = path.suffix.lower() in (".csv", ".tsv")
    sep    = "\t" if path.suffix.lower() == ".tsv" else ","

    gen = _parse_csv(path, sep) if is_csv else _parse_txt(path)

    rows: list[tuple[int, str, str]] = []
    inserted = 0

    for year, text_id, raw in gen:
        is_csv_row = is_csv and "\n" not in raw   # single concordance line
        for sent in _extract_sentences(raw, is_csv_row=is_csv_row):
            rows.append((year, text_id, sent))
            if len(rows) >= batch_size:
                conn.executemany(
                    "INSERT INTO sentences(year, text_id, sentence_text) VALUES (?,?,?)",
                    rows,
                )
                conn.commit()
                inserted += len(rows)
                rows.clear()

    if rows:
        conn.executemany(
            "INSERT INTO sentences(year, text_id, sentence_text) VALUES (?,?,?)",
            rows,
        )
        conn.commit()
        inserted += len(rows)

    return inserted


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Index COHA Linear Text files into a SQLite FTS5 database.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--data-dir",   default="data/coha",     metavar="DIR",
                   help="Directory containing COHA source files (scanned recursively).")
    p.add_argument("--db",         default="coha_corpus.db", metavar="PATH",
                   help="Output SQLite database path.")
    p.add_argument("--batch-size", default=5000, type=int,   metavar="N",
                   help="Rows committed per transaction (tune for memory vs. speed).")
    p.add_argument("--reset",      action="store_true",
                   help="Drop and recreate the DB before indexing.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print("MAD-SC — COHA Indexer")
    print("=" * 50)

    # ── NLTK ────────────────────────────────────────────────────────────────
    print("Checking NLTK sentence tokeniser…")
    _ensure_nltk()

    # ── Source files ─────────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory '{data_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    files: list[Path] = []
    for ext in ("*.txt", "*.csv", "*.tsv"):
        files.extend(p for p in data_dir.rglob(ext)
                     if p.name != "PLACE_COHA_FILES_HERE.txt")
    if not files:
        print(f"No .txt/.csv/.tsv files found under '{data_dir}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} source file(s) in '{data_dir}'.")

    # ── Database ─────────────────────────────────────────────────────────────
    db_path = Path(args.db)
    print(f"Database: {db_path.resolve()}"
          + (" (RESET requested)" if args.reset else ""))

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")      # concurrent read while writing
    conn.execute("PRAGMA synchronous=NORMAL")    # faster writes, safe with WAL
    _create_schema(conn, reset=args.reset)

    # ── Process ───────────────────────────────────────────────────────────────
    total = 0
    for i, path in enumerate(files, 1):
        n = _process_file(path, conn, args.batch_size)
        total += n
        status = f"[{i:>{len(str(len(files)))}}/{len(files)}]  {path.name:<40}  {n:>8,} sentences"
        print(status)

    # ── Summary ───────────────────────────────────────────────────────────────
    row_count = conn.execute("SELECT COUNT(*) FROM sentences").fetchone()[0]
    conn.close()

    print("=" * 50)
    print(f"Indexing complete.")
    print(f"  Total sentences inserted this run : {total:,}")
    print(f"  Total sentences in database       : {row_count:,}")
    print(f"  Database path                     : {db_path.resolve()}")
    print()
    print("You can now launch the app:")
    print("  streamlit run app.py")


if __name__ == "__main__":
    main()
