import argparse, os, subprocess, sys, tempfile
from google.cloud import storage

def _parse_gs(gs_uri: str):
    if not gs_uri.startswith("gs://"):
        raise ValueError(f"URI attendu: gs://bucket/chemin, reçu: {gs_uri}")
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

def _build_out_uri(args) -> str:
    # Déduit l’URI de sortie à partir de l’entrée, du suffixe et du out-dir éventuel
    in_bucket, in_blob = _parse_gs(args.input)
    base_dir, base_name = os.path.split(in_blob)
    stem, ext = os.path.splitext(base_name)
    if not ext:
        ext = ".pdf"
    new_name = stem + args.out_suffix + ext

    if args.out_dir:
        if not args.out_dir.startswith("gs://"):
            raise ValueError("--out-dir doit être un URI gs://bucket[/prefix]")
        out_bucket, out_prefix = _parse_gs(args.out_dir)
        out_prefix = out_prefix.rstrip("/")
        out_blob = f"{out_prefix}/{new_name}" if out_prefix else new_name
        return f"gs://{out_bucket}/{out_blob}"

    # Par défaut: même dossier que l’entrée
    out_blob = f"{base_dir}/{new_name}" if base_dir else new_name
    return f"gs://{in_bucket}/{out_blob}"

def main():
    p = argparse.ArgumentParser(description="GCS -> script PDF -> GCS (auto-naming output)")
    p.add_argument("--input", required=True, help="gs://bucket/path/input.pdf")
    p.add_argument("--output", required=False, help="gs://bucket/path/output.pdf (optionnel si --auto-output)")
    p.add_argument("--auto-output", action="store_true",
                   help="Si --output est omis, construire automatiquement le nom de sortie")
    p.add_argument("--out-suffix", default="_opt",
                   help="Suffixe ajouté avant l'extension (défaut: _opt)")
    p.add_argument("--out-dir", default=None,
                   help="URI gs://bucket[/prefix] où écrire la sortie (sinon même dossier que l’entrée)")
    p.add_argument("--script", default="preopt_pdf_sharp_petitfinmarche.py")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Tout ce qui suit est transmis au script principal")
    args = p.parse_args()

    # Normalise d’éventuels tirets longs “–” collés depuis l’UI
    args.extra = [a.replace("–", "-") for a in args.extra]

    # Résolution de la sortie
    if args.output:
        if args.auto_output:
            print("[runner] --output fourni, --auto-output ignoré", flush=True)
    else:
        if not args.auto_output:
            raise ValueError("Fournir --output ou bien --auto-output")
        args.output = _build_out_uri(args)

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

            # Capture la sortie du script pour la voir dans les logs
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
