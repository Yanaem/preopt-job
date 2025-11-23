#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
OCR ULTRA-PRÉCIS PDF → MARKDOWN
ADAPTÉ POUR UN USAGE LIBRAIRIE / CLOUD RUN

- Même logique que invoice_ocr.py original (page par page, PDF natif, cache Anthropic)
- Prompt identique à invoice_ocr_promptgpt.py
- Plus aucune dépendance à Tkinter ou UI
- Point d'entrée principal : ocr_pdf_to_markdown_batched(pdf_path, batch_size=5)
"""

import os
import sys
import re
import requests
from pathlib import Path
from pypdf import PdfReader, PdfWriter
from io import BytesIO
import base64
import time
import json
from typing import Tuple, Dict, List

# ====== Configuration ======
API_URL = "https://api.anthropic.com"
API_VER = "2023-06-01"
MODEL = "claude-sonnet-4-5"
MAX_TOKENS = 64000

REQUEST_TIMEOUT = 600
MAX_RETRIES = 5
BACKOFF_BASE = 2
BACKOFF_MAX = 120
INTER_REQUEST_DELAY = 2
STOP_ON_CRITICAL = False

# ====== Prompt Système (identique à invoice_ocr_promptgpt.py) ======
SYSTEM_PROMPT = """Vous allez jouer le rôle d'un assistant qui reformate le texte brut d'une facture en un document structuré en **Markdown**, sans aucune perte d'information. Le texte d'entrée est le résultat d'un OCR d'une facture PDF en français (texte brut sans mise en page). **Votre objectif est de reproduire fidèlement toutes les informations extraites de la facture, en les organisant par sections et tableaux Markdown, sans rien inventer ni omettre.**

**Consignes importantes :** Ne générez **aucune information** qui n'apparaît pas explicitement dans le texte OCR fourni. Ne faites **aucune supposition** et ne tentez pas de deviner du contenu manquant. **N'ajoutez pas** de texte explicatif, **ne reformulez pas** le contenu original. Si le texte OCR comporte des erreurs ou des éléments incompréhensibles, laissez-les tels quels dans la limite du possible (ou signalez-les comme `[CHAMP MANQUANT]` s'ils sont illisibles). En particulier, si une donnée attendue n'est pas présente dans le texte (par exemple un numéro de facture manquant, une adresse illisible, etc.), indiquez clairement `[CHAMP MANQUANT]` à sa place plutôt que d'inventer quoi que ce soit.

Formatez la sortie en sections avec des titres **Markdown** clairs pour chaque catégorie d'informations de la facture. Utilisez par exemple la syntaxe de titre Markdown (`## Titre de la section`) pour chaque section principale. Respectez l'ordre et la hiérarchie suivants (si l'information est disponible dans le texte) :

- **Informations émetteur** : Identifiez le vendeur / l'émetteur de la facture (nom de la société ou du prestataire, adresse complète, et toute autre information le concernant présente dans le texte, comme son SIRET, son numéro de TVA intracommunautaire, coordonnées de contact, etc.).  
- **Informations client** : Identifiez le client / destinataire de la facture (nom ou raison sociale, adresse, et éventuelles autres infos comme un numéro de client, si mentionné).  
- **Détails de la facture** : Regroupe les informations générales de la facture, par exemple le numéro de facture, la date d'émission, la date de la vente ou de la prestation, la date d'échéance de paiement, le numéro de commande ou de devis lié le cas échéant, etc. Listez chaque détail pertinent sur une ligne séparée ou sous-forme de sous-éléments si nécessaire (par exemple, « **Numéro de facture :** XXXXXX »).  
- **Tableau des lignes** : Présentez sous forme de tableau Markdown toutes les lignes d'articles ou prestations figurant sur la facture. Chaque ligne du tableau doit correspondre à une ligne de facture. Conservez les colonnes telles qu'elles apparaissent dans le texte d'origine (par exemple : **Description**, **Quantité**, **Prix Unitaire**, **Total HT**, **TVA**, **Total TTC** ...). Utilisez la première ligne du tableau pour les en-têtes de colonnes si ces en-têtes sont présentes dans le texte OCR ; sinon, conservez la structure implicite. **Ne fusionnez pas** et ne réorganisez pas les colonnes : respectez l'ordre original. Si certaines valeurs dans le tableau sont manquantes ou illisibles, insérez `[CHAMP MANQUANT]` dans la cellule correspondante. Veillez à ce que le tableau Markdown soit correctement formaté avec des barres verticales `|` séparant chaque colonne et une ligne de séparation `---` sous la ligne d'en-têtes.  
- **Montants** : Indiquez ici les totaux et récapitulatifs figurant après les lignes de détail. Cela comprend généralement le **Total HT** (hors taxes), le détail de la TVA (par taux, si disponible), le **Total TTC** (toutes taxes comprises), et éventuellement d'autres montants comme des frais annexes, remises ou acomptes déjà versés. Chaque ligne de ce récapitulatif doit reprendre exactement le libellé et le montant tels qu'ils apparaissent dans le texte OCR (par ex. « **Total HT :** 100,00 € », « **TVA 20% :** 20,00 € », « **Total TTC :** 120,00 € »). S'il manque un montant attendu, utilisez `[CHAMP MANQUANT]`.  
- **Informations de paiement** : Si le texte comporte des indications sur le paiement, mentionnez-les dans cette section. Par exemple : modalités ou conditions de paiement (*paiement à 30 jours*, *à régler par virement bancaire*, etc.), coordonnées bancaires du bénéficiaire (IBAN, BIC) si présentes, ainsi que les mentions de pénalités de retard ou d'escompte en cas de paiement anticipé. Chaque information doit figurer sur une ligne distincte ou sous forme de liste à puces si cela s'y prête. Si aucune information de paiement n'est présente, vous pouvez omettre cette section ou la marquer `[CHAMP MANQUANT]` selon le contexte.  
- **Mentions légales** : Recueillez ici toutes les autres mentions textuelles présentes sur la facture qui n'ont pas été incluses dans les sections ci-dessus. Cela peut inclure par exemple : la forme juridique et le capital de l'entreprise émettrice, son numéro SIRET/SIREN et RCS, son numéro de TVA intracommunautaire (s'il ne figurait pas déjà en section émetteur), des mentions du type *« TVA non applicable, article 293 B du CGI »*, l'adresse du site web, le contact du service client, ou toute note de bas de page (du style *« Merci de votre confiance »* ou conditions générales succinctes). **Aucune information visible dans le texte ne doit être ignorée.** Séparez les différentes mentions par des points ou mettez-les sur des lignes distinctes si besoin pour la lisibilité. Si aucune mention légale ou note complémentaire n'apparaît, indiquez `[CHAMP MANQUANT]` dans cette section également (sauf si toutes les infos étaient déjà classées ailleurs).

**Important :** Respectez **scrupuleusement le contenu et la formulation du texte original.** Ne reformulez pas les intitulés (par exemple si l'OCR a capturé « Montant total TTC » ne le transformez pas en « Total TTC » – laissez tel quel). Ne changez pas le format des dates, n'arrondissez pas les montants, n'interprétez pas les abréviations. Votre tâche n'est **que de structurer et organiser** le texte, pas de le traduire ni de le résumer. Enfin, la réponse que vous produirez **doit uniquement contenir le document Markdown formaté** (commençant par les sections ci-dessus), sans aucune explication supplémentaire en dehors des données de la facture.

Commencez maintenant la conversion en suivant ces consignes. Bonne organisation !"""

# ====================================================================


def calculate_backoff_delay(attempt: int) -> int:
    """Backoff exponentiel"""
    return min(BACKOFF_BASE ** attempt, BACKOFF_MAX)


def handle_api_error(error: Exception, attempt: int, context: str) -> Tuple[bool, int]:
    """Gestion erreurs avec backoff"""
    error_str = str(error).lower()

    non_retryable = ["invalid api key", "authentication failed", "permission denied"]
    for non_retry in non_retryable:
        if non_retry in error_str:
            print(f"\n      ❌ Erreur non-récupérable : {error}", flush=True)
            return False, 0

    if attempt >= MAX_RETRIES:
        print(f"\n      ❌ Échec après {MAX_RETRIES} tentatives", flush=True)
        return False, 0

    wait_time = calculate_backoff_delay(attempt)

    if "timeout" in error_str:
        print(f"      ⏳ Timeout {context} (tentative {attempt}/{MAX_RETRIES})", flush=True)
    elif "429" in error_str or "rate limit" in error_str:
        print(f"      🚦 Rate limit (tentative {attempt}/{MAX_RETRIES})", flush=True)
        wait_time = max(wait_time, 60)
    elif "overloaded" in error_str:
        print(f"      🔥 API surchargée (tentative {attempt}/{MAX_RETRIES})", flush=True)
        wait_time = max(wait_time, 30)
    else:
        print(f"      ⚠️  Erreur {context} (tentative {attempt}/{MAX_RETRIES}): {error}", flush=True)

    print(f"      ⏱️  Attente {wait_time}s...", flush=True)
    return True, wait_time


def get_pdf_info(pdf_path: str) -> Dict:
    """Récupère les infos du PDF"""
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        page_count = len(reader.pages)

    file_size = os.path.getsize(pdf_path)

    return {
        "page_count": page_count,
        "file_size_bytes": file_size,
        "file_size_mb": file_size / (1024 * 1024),
    }


def extract_single_page_to_base64(pdf_path: str, page_num: int) -> Tuple[str, int]:
    """
    Extrait UNE page du PDF et la convertit en base64

    Returns:
        (pdf_base64, size_kb)
    """
    writer = PdfWriter()

    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        writer.add_page(reader.pages[page_num - 1])

    buffer = BytesIO()
    writer.write(buffer)
    buffer.seek(0)
    pdf_bytes = buffer.read()

    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
    size_kb = len(pdf_bytes) / 1024

    return pdf_base64, size_kb


def process_page_as_pdf_base64(
    pdf_path: str, page_num: int, api_key: str, is_first_page: bool = False
) -> Tuple[str, Dict]:
    """
    Traite UNE page comme PDF base64 avec cache optimal.
    Prompt et structure de requête identiques à invoice_ocr_promptgpt.py.
    """
    print(f"      📄 Page {page_num}", flush=True)

    # Extraire la page en PDF base64
    print(f"         📦 Extraction PDF page {page_num}...", end=" ", flush=True)
    pdf_base64, size_kb = extract_single_page_to_base64(pdf_path, page_num)
    print(f"{size_kb:.1f} KB", flush=True)

    url = f"{API_URL}/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": API_VER,
        "anthropic-beta": "pdfs-2024-09-25,prompt-caching-2024-07-31",
        "content-type": "application/json",
    }

    # Même schéma que process_page_with_cache dans invoice_ocr_promptgpt.py
    body = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
        "system": [
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": pdf_base64,
                        },
                    }
                ],
            }
        ],
    }

    print(f"         🔄 Traitement OCR...", end=" ", flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=body,
                timeout=REQUEST_TIMEOUT,
            )

            if response.status_code == 200:
                json_response = response.json()

                usage = json_response.get("usage", {})
                cache_creation = usage.get("cache_creation_input_tokens", 0)
                cache_read = usage.get("cache_read_input_tokens", 0)
                input_tokens = usage.get("input_tokens", 0)
                output_tokens = usage.get("output_tokens", 0)

                if is_first_page:
                    print("✅", flush=True)
                    print(f"         💾 Cache créé : {cache_creation:,} tokens", flush=True)
                else:
                    print("✅", flush=True)
                    print(
                        f"         ⚡ Cache HIT : {cache_read:,} tokens | PDF : {input_tokens:,} tokens",
                        flush=True,
                    )

                print(f"         📤 Output : {output_tokens:,} tokens", flush=True)

                # Récupération du texte
                markdown = "\n\n".join(
                    [
                        block.get("text", "")
                        for block in json_response.get("content", [])
                        if block.get("type") == "text"
                    ]
                ).strip()

                # Ajout explicite du marqueur de page, comme dans promptgpt
                markdown = f"<!-- PAGE {page_num} -->\n\n{markdown}\n\n---"

                stats = {
                    "cache_creation": cache_creation,
                    "cache_read": cache_read,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }

                return markdown, stats

            error_msg = f"HTTP {response.status_code}"
            try:
                error_detail = response.json()
                error_msg += f": {error_detail.get('error', {}).get('message', response.text[:200])}"
            except Exception:
                error_msg += f": {response.text[:200]}"

            should_retry, wait_time = handle_api_error(
                Exception(error_msg),
                attempt,
                f"page {page_num}",
            )

            if not should_retry:
                raise Exception(error_msg)

            time.sleep(wait_time)

        except requests.exceptions.Timeout as e:
            should_retry, wait_time = handle_api_error(e, attempt, f"page {page_num} timeout")
            if not should_retry:
                raise
            time.sleep(wait_time)

        except requests.exceptions.RequestException as e:
            should_retry, wait_time = handle_api_error(e, attempt, f"page {page_num} réseau")
            if not should_retry:
                raise
            time.sleep(wait_time)

    raise Exception(f"Échec page {page_num} après {MAX_RETRIES} tentatives")


def clean_decorative_elements_safe(content: str) -> str:
    """
    Nettoyage sécurisé des bordures décoratives :

    - cible les lignes faites quasi uniquement de *, -, =, _, #, \ et espaces
    - ignore les tableaux (lignes avec '|')
    - ignore les textes normaux (lettres/chiffres)
    - remplace par '---' et fusionne les répétitions
    """
    lines = content.splitlines()
    out: List[str] = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        # Ne jamais toucher aux tableaux
        if "|" in line:
            out.append(line.rstrip())
            continue

        stripped = line.strip()
        if not stripped:
            out.append("")
            continue

        # Enlever éventuellement des ** / __ autour (Markdown)
        stripped = stripped.strip("*_")

        # On retire les backslashes et espaces pour analyser le "cœur"
        core = stripped.replace("\\", "").replace(" ", "")

        # S'il reste des lettres/chiffres → ce n'est pas décoratif
        if re.search(r"[A-Za-z0-9]", core):
            out.append(line.rstrip())
            continue

        # Si le cœur est long et ne contient que des caractères décoratifs
        if core and len(core) >= 20 and all(c in "*-=_#" for c in core):
            # On remplace par une seule ligne '---',
            # en évitant d'en empiler plusieurs d'affilée
            if out and out[-1] == "---":
                continue
            out.append("---")
        else:
            out.append(line.rstrip())

    cleaned = "\n".join(out)

    # Compacte les gros blocs de lignes vides
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    # Supprime les espaces de fin de ligne
    cleaned = re.sub(r" +$", "", cleaned, flags=re.MULTILINE)

    return cleaned


def merge_duplicate_table_headers(markdown: str) -> str:
    """Fusionne les tableaux avec en-têtes dupliqués"""
    lines = markdown.split("\n")
    cleaned = []

    i = 0
    while i < len(lines):
        line = lines[i]

        if "|" in line and line.strip().count("|") >= 3:
            header = lines[i].strip()

            if i + 1 < len(lines):
                separator = lines[i + 1].strip()

                if re.match(r"^\|[\s\-:]+\|", separator):
                    cleaned.append(lines[i])
                    cleaned.append(lines[i + 1])
                    i += 2

                    table_rows = []
                    while i < len(lines):
                        current = lines[i].strip()

                        if (
                            current == header
                            and i + 1 < len(lines)
                            and lines[i + 1].strip() == separator
                        ):
                            i += 2
                            continue

                        if not current or current.startswith("<!--"):
                            break

                        if current.startswith("|"):
                            table_rows.append(lines[i])
                            i += 1
                        else:
                            break

                    cleaned.extend(table_rows)
                    continue

        cleaned.append(lines[i])
        i += 1

    return "\n".join(cleaned)


def validate_markdown_quality(markdown: str, expected_pages: int, start_page: int) -> Dict:
    """Valide la qualité du markdown"""
    issues = {"critical": [], "warnings": [], "stats": {}}

    page_markers = re.findall(r"<!-- PAGE (\d+) -->", markdown)
    page_numbers = [int(p) for p in page_markers]

    if len(page_numbers) != expected_pages:
        issues["critical"].append(f"❌ Pages : {len(page_numbers)}/{expected_pages}")

    illegible_count = len(re.findall(r"\[ILLISIBLE\]", markdown, re.IGNORECASE))
    to_verify_count = len(re.findall(r"\[À VÉRIFIER", markdown, re.IGNORECASE))
    amounts = re.findall(r"\d{1,3}(?:[ \.]?\d{3})*,\d{2}", markdown)
    table_count = len(re.findall(r"\|.*\|.*\|", markdown))

    if illegible_count > 0:
        issues["warnings"].append(f"⚠️  {illegible_count} élément(s) illisible(s)")

    issues["stats"]["elements_illegibles"] = illegible_count
    issues["stats"]["elements_a_verifier"] = to_verify_count
    issues["stats"]["montants_detectes"] = len(amounts)
    issues["stats"]["lignes_tableaux"] = table_count
    issues["stats"]["caracteres"] = len(markdown)

    return issues


def save_progress(pdf_path: str, completed_pages: Dict):
    """Sauvegarde progression"""
    progress_file = Path(pdf_path).with_suffix(".progress.json")
    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(completed_pages, f, indent=2, ensure_ascii=False)


def load_progress(pdf_path: str) -> Dict:
    """Charge progression"""
    progress_file = Path(pdf_path).with_suffix(".progress.json")
    if progress_file.exists():
        with open(progress_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def calculate_costs(stats_list: List[Dict]) -> Dict:
    """Calcule les coûts"""
    total_input = sum(s.get("input_tokens", 0) for s in stats_list)
    total_cache_creation = sum(s.get("cache_creation", 0) for s in stats_list)
    total_cache_read = sum(s.get("cache_read", 0) for s in stats_list)
    total_output = sum(s.get("output_tokens", 0) for s in stats_list)

    PRICE_INPUT = 3.0
    PRICE_CACHE_WRITE = 3.75
    PRICE_CACHE_READ = 0.30
    PRICE_OUTPUT = 15.0

    cost_input = (total_input * PRICE_INPUT) / 1_000_000
    cost_cache_write = (total_cache_creation * PRICE_CACHE_WRITE) / 1_000_000
    cost_cache_read = (total_cache_read * PRICE_CACHE_READ) / 1_000_000
    cost_output = (total_output * PRICE_OUTPUT) / 1_000_000
    total_cost = cost_input + cost_cache_write + cost_cache_read + cost_output

    total_tokens_without_cache = total_input + total_cache_creation + total_cache_read
    cost_without_cache = (total_tokens_without_cache * PRICE_INPUT) / 1_000_000 + cost_output

    savings = cost_without_cache - total_cost
    savings_percent = (savings / cost_without_cache * 100) if cost_without_cache > 0 else 0

    return {
        "total_input": total_input,
        "total_cache_creation": total_cache_creation,
        "total_cache_read": total_cache_read,
        "total_output": total_output,
        "cost_with_cache": total_cost,
        "cost_without_cache": cost_without_cache,
        "savings": savings,
        "savings_percent": savings_percent,
    }


def ocr_pdf_to_markdown_batched(pdf_path: str, batch_size: int = 5) -> str:
    """
    Point d'entrée "Cloud Run friendly", même signature que claude_ocr.ocr_pdf_to_markdown_batched.

    - pdf_path : chemin du PDF déjà présent sur le disque (ex: /tmp/input.pdf)
    - batch_size : ignoré (on reste en vrai page-par-page), présent pour compatibilité.

    Retourne : un unique string Markdown avec toutes les pages concaténées.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("Variable ANTHROPIC_API_KEY non définie.")

    pdf_info = get_pdf_info(pdf_path)
    page_count = pdf_info["page_count"]

    print("=" * 70, flush=True)
    print("🔬 EXTRACTION PDF → MARKDOWN (PDF BASE64 + CACHE OPTIMAL)", flush=True)
    print("=" * 70, flush=True)
    print(f"📄 Fichier : {Path(pdf_path).name}", flush=True)
    print(f"📊 Pages : {page_count}", flush=True)
    print(f"💾 Taille : {pdf_info['file_size_mb']:.2f} MB", flush=True)
    print("=" * 70, flush=True)

    completed_pages = load_progress(pdf_path)
    if completed_pages:
        print(f"📂 Reprise automatique : {len(completed_pages)} page(s) déjà traitées", flush=True)

    start_time = time.time()
    all_markdown: List[str] = []
    all_stats: List[Dict] = []

    for page_num in range(1, page_count + 1):
        page_key = str(page_num)

        if page_key in completed_pages:
            print(f"      ✓ Page {page_num} (déjà traitée, reprise)", flush=True)
            all_markdown.append(completed_pages[page_key]["markdown"])
            all_stats.append(completed_pages[page_key]["stats"])
            continue

        if page_num > 1 and INTER_REQUEST_DELAY > 0:
            time.sleep(INTER_REQUEST_DELAY)

        try:
            is_first = page_num == 1 and len(completed_pages) == 0

            markdown, stats = process_page_as_pdf_base64(
                pdf_path, page_num, api_key, is_first_page=is_first
            )

            # Nettoyage léger (bordures décoratives)
            markdown = clean_decorative_elements_safe(markdown)

            all_markdown.append(markdown)
            all_stats.append(stats)

            completed_pages[page_key] = {
                "markdown": markdown,
                "stats": stats,
            }

            if page_num % 5 == 0:
                save_progress(pdf_path, completed_pages)
                print(f"         💾 Progression sauvegardée (page {page_num})", flush=True)

            print(f"         ✅ Page {page_num} terminée\n", flush=True)

        except Exception as e:
            print(f"\n         ❌ Erreur page {page_num}: {e}", flush=True)

            if STOP_ON_CRITICAL:
                raise

            error_md = f"<!-- PAGE {page_num} -->\n**[ERREUR EXTRACTION]**\n---"
            all_markdown.append(error_md)
            all_stats.append(
                {
                    "cache_creation": 0,
                    "cache_read": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                }
            )

            print(f"         ⚠️  Marquée comme erreur, continuation...\n", flush=True)

    duration = time.time() - start_time

    print("\n" + "=" * 70, flush=True)
    print("🔧 FINALISATION", flush=True)
    print("=" * 70, flush=True)
    print("\n   🔗 Fusion des pages...", flush=True)

    final_markdown = "\n\n".join(all_markdown)
    final_markdown = merge_duplicate_table_headers(final_markdown)

    md_size_kb = len(final_markdown.encode("utf-8")) / 1024
    costs = calculate_costs(all_stats)
    validation = validate_markdown_quality(final_markdown, page_count, 1)

    progress_file = Path(pdf_path).with_suffix(".progress.json")
    if progress_file.exists():
        try:
            progress_file.unlink()
            print("   🗑️  Fichier de progression supprimé", flush=True)
        except OSError:
            pass

    print("\n" + "=" * 70, flush=True)
    print("✅ EXTRACTION TERMINÉE", flush=True)
    print("=" * 70, flush=True)
    print(f"📄 Pages extraites  : {page_count}", flush=True)
    print(f"💾 Taille Markdown  : {md_size_kb:.1f} KB", flush=True)
    print(f"⏱️  Durée totale     : {duration//60:.0f}min {duration%60:.0f}s", flush=True)
    print(f"⚡ Vitesse moyenne  : {duration/page_count:.1f}s/page", flush=True)

    print("\n" + "-" * 70, flush=True)
    print("💰 STATISTIQUES DE COÛT (PDF Base64 + Cache)", flush=True)
    print("-" * 70, flush=True)
    print(f"📥 Input tokens         : {costs['total_input']:,}", flush=True)
    print(f"📝 Cache creation       : {costs['total_cache_creation']:,}", flush=True)
    print(f"💾 Cache read           : {costs['total_cache_read']:,}", flush=True)
    print(f"📤 Output tokens        : {costs['total_output']:,}", flush=True)

    total_cached = costs["total_cache_read"]
    total_input_all = costs["total_input"] + costs["total_cache_creation"]
    cache_eff = (
        (total_cached / (total_cached + total_input_all) * 100)
        if (total_cached + total_input_all) > 0
        else 0
    )

    print(f"\n🎯 Efficacité cache     : {cache_eff:.1f}%", flush=True)
    print(f"💵 Coût AVEC cache      : ${costs['cost_with_cache']:.4f}", flush=True)
    print(f"💵 Coût SANS cache      : ${costs['cost_without_cache']:.4f}", flush=True)

    if costs["savings"] > 0:
        print(
            f"💰 ÉCONOMIE             : ${costs['savings']:.4f} ({costs['savings_percent']:.1f}%) 💸",
            flush=True,
        )

    print("\n" + "-" * 70, flush=True)
    print("🔍 QUALITÉ", flush=True)
    print("-" * 70, flush=True)

    if not validation["critical"] and not validation["warnings"]:
        print("✅ Extraction parfaite", flush=True)
    elif not validation["critical"]:
        print(f"✅ Extraction réussie avec {len(validation['warnings'])} avertissement(s)", flush=True)
    else:
        print(f"⚠️  {len(validation['critical'])} problème(s) détectés", flush=True)

    if validation["stats"]:
        stats = validation["stats"]
        print(
            f"📊 {stats.get('montants_detectes', 0)} montants, "
            f"{stats.get('lignes_tableaux', 0)} lignes tableaux",
            flush=True,
        )

    print("=" * 70 + "\n", flush=True)

    return final_markdown


def save_markdown(markdown: str, md_path: str):
    """Écrit le Markdown dans un fichier local."""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)


def main():
    """
    Entrée CLI simple (utile pour debug local, pas pour Cloud Run) :
        python invoice_ocr.py /chemin/vers/facture.pdf
    """
    if len(sys.argv) < 2:
        print("Usage: invoice_ocr.py <chemin_pdf>", file=sys.stderr)
        sys.exit(1)

    pdf_path = sys.argv[1]
    md = ocr_pdf_to_markdown_batched(pdf_path)
    md_path = Path(pdf_path).with_suffix(".md")
    save_markdown(md, md_path)
    print(f"📝 Fichier Markdown écrit : {md_path}", flush=True)


if __name__ == "__main__":
    main()
