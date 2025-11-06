import argparse, os, subprocess, sys, tempfile
from google.cloud import storage

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
    content_type = "application/pdf" if local_path.lower().endswith(".pdf") else None
    blob.upload_from_filename(local_path, content_type=content_type)

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

    # Par défaut: même dossier que l’entrée
    out_blob = f"{base_dir}/{new_name}" if base_dir else new_name
    return f"gs://{in_bucket}/{out_blob}"

# ---------- Runner ----------

def main():
    p = argparse.ArgumentParser(description="GCS -> script PDF -> GCS (args UI + ENV fallbacks)")
    # NOTE: --input/--output désormais OPTIONNELS (fallback ENV)
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

    # Normalise d’éventuels tirets longs “–” collés depuis l’UI
    args.extra = [a.replace("–", "-") for a in args.extra]

    # --------- Fallback via ENV (ne PAS casser les args UI du Job) ----------
    env_input = os.environ.get("GCS_INPUT")
    env_output = os.environ.get("GCS_OUTPUT")
    env_auto_output = os.environ.get("GCS_AUTO_OUTPUT", "")
    env_out_dir = os.environ.get("GCS_OUT_DIR")
    env_out_suffix = os.environ.get("GCS_OUT_SUFFIX")

    # --input
    if not args.input and env_input:
        args.input = env_input

    # --out-dir / --out-suffix / --auto-output
    if not args.out_dir and env_out_dir:
        args.out_dir = env_out_dir
    if args.out_suffix == "_opt" and env_out_suffix:
        args.out_suffix = env_out_suffix
    if (not args.auto_output) and (env_auto_output.lower() in ("1", "true", "yes")):
        args.auto_output = True

    # --output
    if not args.output and env_output:
        args.output = env_output

    # Validations minimales
    if not args.input:
        raise ValueError("Fournir --input OU définir ENV GCS_INPUT")
    if not args.output and not args.auto_output:
        raise ValueError("Fournir --output OU activer auto-output (--auto-output ou ENV GCS_AUTO_OUTPUT=true)")

    # Résolution auto-output si nécessaire
    if not args.output and args.auto_output:
        args.output = _build_out_uri(args.input, args.out_suffix, args.out_dir)

    # Logs utiles (diagnostic)
    print(f"[runner] input = {args.input}", flush=True)
    print(f"[runner] output = {args.output}", flush=True)
    print(f"[runner] script = {args.script}", flush=True)
    print(f"[runner] extra  = {args.extra}", flush=True)

    try:
        with tempfile.TemporaryDirectory() as td:
            in_local  = os.path.join(td, "in.pdf")
            out_local = os.path.join(td, "out.pdf")

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
                sys.exit(proc.returncode)

            if not os.path.exists(out_local) or os.path.getsize(out_local) == 0:
                raise RuntimeError("Le script n'a pas produit de fichier de sortie")

            print(f"[runner] Upload: {out_local} -> {args.output}", flush=True)
            upload(out_local, args.output)
            print(f"[runner] Terminé: {args.output}", flush=True)

    except Exception as e:
        print(f"[runner][ERROR] {type(e).__name__}: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
