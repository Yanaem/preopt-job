# claude_ocr.py
import os
import json
import requests
from google.cloud import storage

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

def generate_signed_url(gcs_uri: str, expires_seconds: int = 3600) -> str:
    """
    gcs_uri : 'gs://bucket/path/to/file.pdf'
    Retourne une URL signée HTTP utilisable par Claude.
    """
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"URI GCS invalide: {gcs_uri}")
    _, bucket_name, *path_parts = gcs_uri.split("/")
    blob_name = "/".join(path_parts)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    url = blob.generate_signed_url(
        version="v4",
        expiration=expires_seconds,
        method="GET",
    )
    return url

def call_claude_with_pdf_url(pdf_url: str) -> str:
    """Appelle Claude et renvoie le Markdown."""
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY non configurée")

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
                        "type": "document",
                        "source": {
                            "type": "url",
                            "url": pdf_url,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Transcris ce document PDF complet en Markdown "
                            "avec les balises <!-- PAGE N -->."
                        ),
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
        timeout=600,  # 10 minutes max
    )

    if not resp.ok:
        raise RuntimeError(f"Claude API error {resp.status_code}: {resp.text}")

    data = resp.json()
    chunks = [
        block["text"]
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    markdown = "\n\n".join(chunks).strip()
    return markdown

def upload_markdown_to_gcs(markdown: str, gcs_uri_md: str):
    """
    gcs_uri_md : 'gs://bucket/path/to/file.md'
    """
    if not gcs_uri_md.startswith("gs://"):
        raise ValueError(f"URI GCS invalide: {gcs_uri_md}")
    _, bucket_name, *path_parts = gcs_uri_md.split("/")
    blob_name = "/".join(path_parts)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_string(markdown, content_type="text/markdown; charset=utf-8")
