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

def main():
    p = argparse.ArgumentParser(description="GCS -> script PDF -> GCS")
    p.add_argument("--input", required=True, help="gs://bucket/in.pdf")
    p.add_argument("--output", required=True, help="gs://bucket/out.pdf")
    p.add_argument("--script", default="preopt_pdf_sharp_petitfinmarche.py")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[])
    args = p.parse_args()

    # Normalise d’éventuels tirets longs “–” collés depuis l’UI
    args.extra = [a.replace("–", "-") for a in args.extra]

    print(f"[runner] args.input={args.input}", flush=True)
    print(f"[runner] args.output={args.output}", flush=True)
    print(f"[runner] script={args.script}", flush=True)
    print(f"[runner] extra={args.extra}", flush=True)

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
                raise RuntimeError("Le script n'a pas produit de fichier out.pdf")

            print(f"[runner] Upload: {out_local} -> {args.output}", flush=True)
            upload(out_local, args.output)
            print(f"[runner] Terminé: {args.output}", flush=True)

    except Exception as e:
        # Toujours loguer l’exception pour la voir dans “Container”
        print(f"[runner][ERROR] {type(e).__name__}: {e}", flush=True)
        sys.exit(1)

if __name__ == "__main__":
    # Assure un flush immédiat si l’ENTRYPOINT n’est pas en -u
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    main()
