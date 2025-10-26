import argparse, os, subprocess, sys, tempfile
from google.cloud import storage

def _parse_gs(gs_uri):
    if not gs_uri.startswith("gs://"):
        raise ValueError("URI attendu: gs://bucket/chemin")
    without = gs_uri[len("gs://"):]
    parts = without.split("/", 1)
    if len(parts) != 2:
        raise ValueError("URI gs:// invalide")
    return parts[0], parts[1]

def download(gcs_uri, local_path):
    bucket_name, blob_name = _parse_gs(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def upload(local_path, gcs_uri):
    bucket_name, blob_name = _parse_gs(gcs_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    content_type = "application/pdf" if local_path.lower().endswith(".pdf") else None
    blob.upload_from_filename(local_path, content_type=content_type)

def main():
    p = argparse.ArgumentParser(description="Wrapper GCS -> script PDF -> GCS")
    p.add_argument("--input", required=True, help="gs://bucket/in.pdf")
    p.add_argument("--output", required=True, help="gs://bucket/out.pdf")
    p.add_argument("--script", default="preopt_pdf_sharp_petitfinmarche.py", help="Nom du script à exécuter")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Arguments pass-through pour le script")
    args = p.parse_args()

    with tempfile.TemporaryDirectory() as td:
        in_local  = os.path.join(td, "in.pdf")
        out_local = os.path.join(td, "out.pdf")

        print("[runner] Téléchargement:", args.input, "->", in_local, flush=True)
        download(args.input, in_local)

        cmd = ["python", args.script, "--input", in_local, "--output", out_local]
        if args.extra:
            cmd += args.extra
        print("[runner] Lancement:", " ".join(cmd), flush=True)
        subprocess.check_call(cmd)

        print("[runner] Upload:", out_local, "->", args.output, flush=True)
        upload(out_local, args.output)
        print("[runner] Terminé:", args.output, flush=True)

if __name__ == "__main__":
    main()
