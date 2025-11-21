# claude_ocr.py
import os
import base64
import json
import requests
import tempfile
import fitz  # PyMuPDF

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
ANTHROPIC_API_VER = os.environ.get("ANTHROPIC_API_VER", "2023-06-01")
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "20000"))

SYSTEM_PROMPT = """Tu es un expert en extraction de données de factures PDF vers Markdown.

**Règles d'extraction :**
1. Fidélité ABSOLUE aux données (montants, dates, références, tableaux)
2. Marquer chaque page : <!-- PAGE N -->
3. Tableaux en format Markdown propre
4. Extrais TOUTES les informations visibles
5. Si illisible : [ILLISIBLE]

**Format de sortie pour chaque page :**
<!-- PAGE N -->
[Contenu intégral de la page]
---"""

def _pdf_to_base64(pdf_path: str) -> str:
    with open(pdf_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("ascii")


def _call_claude_on_pdf(pdf_path: str, start_page: int, end_page: int) -> str:
    """
    Envoie un petit PDF (sous-ensemble) à Claude.
    start_page / end_page = numérotation globale à utiliser dans les balises <!-- PAGE N -->.
    """
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY non configurée dans les variables d'environnement")

    pdf_b64 = _pdf_to_base64(pdf_path)

    user_instruction = (
        f"Le PDF joint contient les pages {start_page} à {end_page} du document complet.\n"
        f"Pour la première page de ce fichier, utilise exactement la balise <!-- PAGE {start_page} -->,\n"
        f"puis <!-- PAGE {start_page+1} -->, etc, jusqu'à <!-- PAGE {end_page} -->.\n"
        "Ne rajoute aucune autre balise de page ni résumé global.\n"
        "Transcris fidèlement tout le contenu en Markdown (tables, textes, montants...)."
    )

    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_instruction,
                    },
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_b64,
                        },
                    },
                ],
            }
        ],
    }

    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_API_VER,
        "content-type": "application/json",
    }

    resp = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers=headers,
        data=json.dumps(payload),
        timeout=900,  # max 15 min
    )

    if not resp.ok:
        raise RuntimeError(f"Claude API error {resp.status_code}: {resp.text}")

    data = resp.json()
    chunks = [
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    markdown = "\n\n".join(chunks).strip()
    return markdown


def ocr_pdf_to_markdown_batched(pdf_path: str, batch_size: int = 5) -> str:
    """
    Découpe pdf_path en sous-PDF de batch_size pages,
    envoie chaque sous-PDF à Claude, concatène tous les Markdown.
    """
    # On lit juste le nombre de pages
    src = fitz.open(pdf_path)
    total_pages = len(src)
    src.close()

    all_chunks = []

    for start in range(1, total_pages + 1, batch_size):
        end = min(start + batch_size - 1, total_pages)

        # Création d'un sous-PDF temporaire contenant les pages start..end
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pdf")
        os.close(tmp_fd)
        try:
            src = fitz.open(pdf_path)
            sub = fitz.open()
            sub.insert_pdf(src, from_page=start - 1, to_page=end - 1)
            sub.save(tmp_path)
            sub.close()
            src.close()

            print(f"[OCR] Envoi à Claude des pages {start} à {end} (fichier {tmp_path})", flush=True)
            chunk_md = _call_claude_on_pdf(tmp_path, start, end)
            all_chunks.append(chunk_md)
            print(f"[OCR] Pages {start}-{end} traitées, longueur chunk={len(chunk_md)}", flush=True)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    # Concaténation propre
    global_md = "\n\n".join(ch for ch in all_chunks if ch).strip()
    return global_md


def save_markdown(markdown: str, md_path: str):
    """Écrit le Markdown dans un fichier local."""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)
