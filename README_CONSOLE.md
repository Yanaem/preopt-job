# Cloud Run Jobs – Bundle (console uniquement)

Contenu:
- Dockerfile
- requirements.txt
- runner.py (gère téléchargement GCS -> exécution -> upload GCS)
- preopt_pdf_sharp_petitfinmarche.py (votre script)

## Étapes (tout en Console GCP)

1) **Cloud Storage** → créez (ou choisissez) un bucket → **Upload** ce ZIP.
2) **Artifact Registry** → *Repositories* → **Create repository**
   - Format: Docker
   - Location: votre région (ex: europe-west1)
   - Nom: jobs-preopt
3) **Cloud Build** → *Builds* → **Create Build**
   - *Source*: Cloud Storage → sélectionnez ce ZIP.
   - *Build type*: **Dockerfile**
   - *Dockerfile directory*: `/` (racine du ZIP)
   - *Destination*: `europe-west1-docker.pkg.dev/PROJECT_ID/jobs-preopt/preopt:1`
   - Lancez le build. À la fin, vous verrez l'image publiée.
4) **Cloud Run** → *Jobs* → **Create Job**
   - *URL de l'image du conteneur*: l'URL ci-dessus.
   - Onglet **Conteneurs** → **Arguments** (un par cellule) :
     ```
     --input
     gs://VOTRE_BUCKET/in.pdf
     --output
     gs://VOTRE_BUCKET/out.pdf
     --extra
     --verbose
     --super-sample
     1.5
     --post-unsharp
     0.5
     ```
   - **CPU/Mémoire**: 2 vCPU / 2–4 GiB (selon taille PDF)
   - **Délai d'expiration**: 3600
   - **Max retries**: 0
5) **Droits GCS**
   - **Cloud Storage** → votre bucket → **Permissions** → **Grant access**
   - Principal: `<PROJECT_NUMBER>-compute@developer.gserviceaccount.com`
   - Rôle: `Storage Object Admin`
6) **Exécuter** le job → *Execute* → regardez les **Logs**.
