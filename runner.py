import argparse, os, subprocess, sys, tempfile
from datetime import datetime, timezone

from google.cloud import storage
import requests

# ---------- GCS utils ----------

def _parse_gs(gs_uri: str):
    if not gs_uri or not gs_uri.startswith("gs://"):
        raise ValueError(f"URI attendu: gs://bucket/chemin, reçu: {gs_uri!r}")
    without = gs_uri[5:]
    parts = without.split("/", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"URI gs:// invalide: {gs_uri}")
    return parts[0], parts[1]

def download(gcs_uri, local_path):
    bucket_name, blob_name = _parse_gs(gcs_uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)
    if not blob.exists():
        raise FileNotFoundError(f"Objet introuvable: {gcs_uri}")
    blob.download_to_filename(local_path)

def upload(local_path, gcs_uri):
    bucket_name, blob_name = _parse_gs(gcs_uri)
    client = storage.Client()
    blob = client.bucket(bucket_name).blob(blob_name)

    ct = None
    lp = local_path.lower()
    if lp.endswith(".pdf"):
        ct = "application/pdf"
    elif lp.endswith(".md"):
        ct = "text/markdown; charset=utf-8"
    elif lp.endswith(".txt"):
        ct = "text/plain; charset=utf-8"

    blob.upload_from_filename(local_path, content_type=ct)

def _build_out_uri(input_uri: str, out_suffix: str, out_dir: str | None) -> str:
    in_bucket, in_blob = _parse_gs(input_uri)
    base_dir, base_name = os.path.split(in_blob)
    stem, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".pdf"
    new_name = stem + out_suffix + ext

    if out_dir:
        if not out_dir.startswith("gs://"):
            raise ValueError("--out-dir doit être un URI gs://bucket[/prefix]")
        out_bucket, out_prefix = _parse_gs(out_dir)
        out_prefix = out_prefix.rstrip("/")
        out_blob = f"{out_prefix}/{new_name}" if out_prefix else new_name
        return f"gs://{out_bucket}/{out_blob}"

    out_blob = f"{base_dir}/{new_name}" if base_dir else new_name
    return f"gs://{in_bucket}/{out_blob}"

def _build_md_uri_from_output_pdf(output_pdf_uri: str) -> str:
    """
    gs://.../xxx_opt.pdf -> gs://.../xxx_opt_ocr.md
    """
    out_bucket, out_blob = _parse_gs(output_pdf_uri)
    base_dir, base_name = os.path.split(out_blob)
    stem, _ = os.path.splitext(base_name)
    md_name = stem + "_ocr.md"
    md_blob = f"{base_dir}/{md_name}" if base_dir else md_name
    return f"gs://{out_bucket}/{md_blob}"


# ---------- Supabase status (pour Lovable) ----------

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

def update_ocr_job(status: str, **extra_fields):
    """
    Met à jour la ligne ocr_jobs avec l'id OCR_JOB_ID (ENV).
    fields supplémentaires : optimized_pdf_path, markdown_path, error, etc.
    """
    job_id = os.environ.get("OCR_JOB_ID")
    if not job_id:
        print("[runner] OCR_JOB_ID non défini, skip update_ocr_job.", flush=True)
        return
    if not SUPABASE_URL or not SUPABASE_SERVICE_KEY:
        print("[runner] SUPABASE_URL / SUPABASE_SERVICE_KEY manquants, skip update_ocr_job.", flush=True)
        return

    url = f"{SUPABASE_URL}/rest/v1/ocr_jobs?id=eq.{job_id}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }
    body = {
        "status": status,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    body.update(extra_fields)
    try:
        resp = requests.patch(url, json=body, headers=headers, timeout=10)
        print(f"[runner] update_ocr_job status={status} -> {resp.status_code}", flush=True)
    except Exception as e:
        print(f"[runner] Erreur update_ocr_job: {e}", flush=True)


# ---------- Runner ----------

def main():
    p = argparse.ArgumentParser(description="GCS -> script PDF -> GCS (args UI + ENV fallbacks)")
    p.add_argument("--input", required=False, help="gs://... (sinon ENV GCS_INPUT)")
    p.add_argument("--output", required=False, help="gs://... (sinon auto-output ou ENV GCS_OUTPUT)")
    p.add_argument("--auto-output", action="store_true",
                   help="Si --output omis, construire automatiquement le nom de sortie")
    p.add_argument("--out-suffix", default="_opt",
                   help="Suffixe ajouté avant l'extension (défaut: _opt; override via ENV GCS_OUT_SUFFIX)")
    p.add_argument("--out-dir", default=None,
                   help="URI gs://bucket[/prefix] où écrire la sortie (override via ENV GCS_OUT_DIR)")
    p.add_argument("--script", default="preopt_pdf_sharp_petitfinmarche.py")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Tout ce qui suit est transmis au script principal")
    args = p.parse_args()

    # Normalise d’éventuels tirets longs “–”
    args.extra = [a.replace("–", "-") for a in args.extra]

    # Fallback via ENV
    env_input = os.environ.get("GCS_INPUT")
    env_output = os.environ.get("GCS_OUTPUT")
    env_out_dir = os.environ.get("GCS_OUT_DIR")
    env_out_suffix = os.environ.get("GCS_OUT_SUFFIX")

    if not args.input and env_input:
        args.input = env_input
    if not args.out_dir and env_out_dir:
        args.out_dir = env_out_dir
    if args.out_suffix == "_opt" and env_out_suffix:
        args.out_suffix = env_out_suffix

    if not args.output and env_output:
        args.output = env_output

    if not args.input:
        raise ValueError("Fournir --input OU définir ENV GCS_INPUT")

    # Auto-output PAR DÉFAUT si output manquant
    if not args.output:
        args.output = _build_out_uri(args.input, args.out_suffix, args.out_dir)
        args.auto_output = True

    print(f"[runner] input  = {args.input}", flush=True)
    print(f"[runner] output = {args.output}", flush=True)
    print(f"[runner] script = {args.script}", flush=True)
    print(f"[runner] extra  = {args.extra}", flush=True)

    update_ocr_job("running")

    try:
        with tempfile.TemporaryDirectory() as td:
            in_local  = os.path.join(td, "in.pdf")
            out_local = os.path.join(td, "out.pdf")
            md_local  = os.path.join(td, "out_ocr.md")  # le script OCR écrit ici

            print(f"[runner] Téléchargement: {args.input} -> {in_local}", flush=True)
            download(args.input, in_local)
            print("[runner] OK download", flush=True)

            cmd = ["python", args.script, "--input", in_local, "--output", out_local] + args.extra
            print("[runner] Lancement:", " ".join(cmd), flush=True)

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.stdout:
                print("[script][stdout]\n" + proc.stdout, flush=True)
            if proc.stderr:
                print("[script][stderr]\n" + proc.stderr, flush=True)
            if proc.returncode != 0:
                print(f"[runner] Script exit code={proc.returncode}", flush=True)
                update_ocr_job("failed", error=f"Script exit code {proc.returncode}")
                sys.exit(proc.returncode)

            if not os.path.exists(out_local) or os.path.getsize(out_local) == 0:
                raise RuntimeError("Le script n'a pas produit de fichier PDF de sortie")

            # Upload PDF optimisé
            print(f"[runner] Upload PDF: {out_local} -> {args.output}", flush=True)
            upload(out_local, args.output)
            print(f"[runner] Terminé PDF: {args.output}", flush=True)
            update_ocr_job("pdf_optimized", optimized_pdf_path=args.output)

            # Upload Markdown si présent
            if os.path.exists(md_local) and os.path.getsize(md_local) > 0:
                md_gcs_uri = _build_md_uri_from_output_pdf(args.output)
                print(f"[runner] Upload Markdown: {md_local} -> {md_gcs_uri}", flush=True)
                upload(md_local, md_gcs_uri)
                print(f"[runner] Terminé Markdown: {md_gcs_uri}", flush=True)
                update_ocr_job("markdown_generated", optimized_pdf_path=args.output, markdown_path=md_gcs_uri)
            else:
                print("[runner] Aucun Markdown OCR trouvé (out_ocr.md), skip upload.", flush=True)

    except Exception as e:
        print(f"[runner][ERROR] {type(e).__name__}: {e}", flush=True)
        update_ocr_job("failed", error=str(e))
        sys.exit(1)


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
