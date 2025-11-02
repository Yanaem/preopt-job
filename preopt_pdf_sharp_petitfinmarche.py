#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preopt_pdf_sharp.py — Pré-optimisation PDF pour OCR avec accent sur la netteté

Optimisations clés :
- Pipeline de traitement amélioré et corrigé
- Meilleure gestion de la netteté
- Correction des bugs potentiels
- Performance optimisée
"""

import argparse
import io
import os
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image


# --------------------- Config ---------------------

@dataclass
class PreoptConfig:
    dpi: int = 400
    keep_text_pages: bool = True
    normalize_background: bool = True
    denoise_h: int = 10
    clahe: bool = True
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    gamma: float = 1.05
    unsharp: float = 0.8
    unsharp_radius: float = 1.0
    binarize: str = "none"
    adaptive_block: int = 25
    adaptive_C: int = 10
    remove_speckles_area: int = 12
    morph_open: int = 2
    morph_close: int = 2
    deskew: bool = True
    max_skew_deg: float = 10.0
    crop: bool = True
    crop_margin: int = 5
    output_color: str = "gray"
    image_format: str = "JPEG"
    jpeg_quality: int = 85
    super_sample: float = 1.0
    post_unsharp: float = 0.0
    post_unsharp_radius: float = 0.8
    deblur_rl_iter: int = 0
    deblur_rl_psf: int = 3


# --------------------- Utils ---------------------

def _make_odd(n: int, minimum: int = 3) -> int:
    """Assure qu'un nombre est impair."""
    n = max(n, minimum)
    return n if n % 2 == 1 else n + 1


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    """Conversion RGB vers niveaux de gris."""
    if len(rgb.shape) == 2:
        return rgb
    if rgb.shape[2] == 4:  # RGBA
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Normalise l'arrière-plan pour compenser l'éclairage inégal."""
    # Kernel adaptatif basé sur la taille de l'image
    k = _make_odd(max(15, int(max(gray.shape) * 0.02)))
    
    # Méthode plus robuste avec morphologie
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    bg = cv2.GaussianBlur(bg, (k, k), 0)
    
    # Normalisation avec protection contre division par zéro
    bg_float = bg.astype(np.float32) + 1.0
    gray_float = gray.astype(np.float32)
    normalized = cv2.divide(gray_float, bg_float, scale=255.0)
    
    return np.clip(normalized, 0, 255).astype(np.uint8)


def denoise(gray: np.ndarray, h: int) -> np.ndarray:
    """Débruitage adaptatif."""
    if h <= 0:
        return gray
    # Ajustement automatique selon la taille
    template_size = min(7, max(3, gray.shape[0] // 100))
    search_size = template_size * 3
    return cv2.fastNlMeansDenoising(
        gray, None, 
        h=h, 
        templateWindowSize=template_size, 
        searchWindowSize=search_size
    )


def clahe_contrast(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    """CLAHE avec ajustement adaptatif."""
    # Ajuster la taille de tuile en fonction de l'image
    h, w = gray.shape
    tile_x = max(4, min(tile, w // 8))
    tile_y = max(4, min(tile, h // 8))
    
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_x, tile_y))
    return clahe.apply(gray)


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    """Correction gamma optimisée."""
    if abs(gamma - 1.0) < 1e-3:
        return gray
    
    # LUT pré-calculée pour performance
    inv_gamma = 1.0 / gamma
    lut = np.array([
        ((i / 255.0) ** inv_gamma) * 255 
        for i in range(256)
    ], dtype=np.uint8)
    
    return cv2.LUT(gray, lut)


def unsharp_mask(gray: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """Masque de netteté amélioré."""
    if amount <= 0:
        return gray
    
    # Sigma optimal pour le rayon donné
    sigma = max(0.3, radius)
    
    # Flou gaussien
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    
    # Masque de netteté avec clipping pour éviter les artefacts
    sharpened = cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)
    
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def binarize(gray: np.ndarray, method: str, block: int, C: int) -> np.ndarray:
    """Binarisation robuste avec plusieurs méthodes."""
    if method == "none":
        return gray
    
    if method == "otsu":
        # Otsu avec pré-traitement
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw
    
    # Binarisation adaptative
    block = _make_odd(block, 11)
    
    # Gaussian adaptive (meilleur pour OCR)
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, C
    )


def clean_speckles(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Nettoyage des taches avec analyse de composantes connexes."""
    if min_area <= 0:
        return binary
    
    # Inverser pour traiter le texte comme foreground
    inverted = cv2.bitwise_not(binary)
    
    # Analyse des composantes
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        inverted, connectivity=8, ltype=cv2.CV_32S
    )
    
    # Créer un masque pour les petites composantes
    output = np.zeros_like(binary)
    
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            output[labels == lbl] = 255
    
    return cv2.bitwise_not(output)


def morphology(binary: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    """Opérations morphologiques avec kernels adaptatifs."""
    out = binary.copy()
    
    if open_k > 0:
        # Kernel rectangulaire pour opening (enlève bruit)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    
    if close_k > 0:
        # Kernel elliptique pour closing (comble trous)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    
    return out


def estimate_skew(binary: np.ndarray, max_deg: float) -> float:
    """Estimation robuste de l'angle d'inclinaison."""
    h, w = binary.shape
    
    # Downscale pour performance si image trop grande
    if max(h, w) > 2000:
        scale = 2000 / max(h, w)
        small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = binary
    
    # Détection de contours
    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    
    # Détection de lignes avec paramètres adaptatifs
    min_line_length = max(50, small.shape[1] // 4)
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=20
    )
    
    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # Filtrer les angles pertinents
            if abs(angle) < max_deg:
                angles.append(angle)
            elif abs(angle - 180) < max_deg:
                angles.append(angle - 180)
            elif abs(angle + 180) < max_deg:
                angles.append(angle + 180)
    
    if len(angles) > 3:
        # Utiliser la médiane pour robustesse
        return float(np.median(angles))
    
    # Fallback: minAreaRect
    coords = np.column_stack(np.where(small > 0)[::-1])
    if len(coords) < 5:
        return 0.0
    
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Corriger l'angle
    if angle < -45:
        angle = 90 + angle
    
    return float(np.clip(angle, -max_deg, max_deg))


def rotate(img: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
    """Rotation avec gestion optimale des bordures."""
    if abs(angle) < 0.05:
        return img
    
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    
    # Matrice de rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculer les nouvelles dimensions pour éviter le rognage
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    
    # Ajuster la translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    
    return cv2.warpAffine(
        img, M, (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )


def crop_to_content(binary: np.ndarray, margin: int) -> Tuple[int, int, int, int]:
    """Recadrage intelligent du contenu."""
    h, w = binary.shape
    
    # Trouver les pixels de contenu
    coords = np.column_stack(np.where(binary < 250))
    
    if len(coords) == 0:
        return 0, 0, w, h
    
    # Bounding box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    # Ajouter une marge
    x0 = max(0, x_min - margin)
    y0 = max(0, y_min - margin)
    x1 = min(w, x_max + margin + 1)
    y1 = min(h, y_max + margin + 1)
    
    return int(x0), int(y0), int(x1), int(y1)


def render_page_rgb(page: fitz.Page, dpi: int) -> np.ndarray:
    """Rasterisation de page PDF en RGB."""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB)
    
    # Conversion en numpy array
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.h, pix.w, 3)


def downscale(img: np.ndarray, scale: float) -> np.ndarray:
    """Réduction avec anti-aliasing optimal."""
    if scale <= 1.0:
        return img
    
    h, w = img.shape[:2]
    new_w = int(round(w / scale))
    new_h = int(round(h / scale))
    
    # INTER_AREA est optimal pour le downscaling
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def apply_rl_deblur(gray: np.ndarray, psf_size: int, iterations: int) -> np.ndarray:
    """Déconvolution Richardson-Lucy (optionnel)."""
    if iterations <= 0:
        return gray
    
    try:
        from skimage.restoration import richardson_lucy
        from scipy.signal import convolve2d
    except ImportError:
        return gray
    
    # PSF gaussien
    psf_size = _make_odd(psf_size, 3)
    sigma = psf_size / 3.0
    
    # Créer PSF 2D
    ax = np.arange(-psf_size // 2 + 1, psf_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    psf = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    psf = psf / psf.sum()
    
    # Normaliser l'image
    img_float = gray.astype(np.float64) / 255.0
    
    # RL deconvolution
    deconv = richardson_lucy(img_float, psf, num_iter=iterations, clip=False)
    
    # Reconvertir
    result = np.clip(deconv * 255.0, 0, 255).astype(np.uint8)
    
    return result


def insert_image_page(out_doc: fitz.Document, img: Image.Image, 
                     dpi: int, fmt: str, quality: int):
    """Insertion d'image dans PDF avec paramètres optimaux."""
    w_px, h_px = img.size
    page_w = w_px * 72.0 / dpi
    page_h = h_px * 72.0 / dpi
    
    page = out_doc.new_page(width=page_w, height=page_h)
    
    # Sauvegarder l'image en mémoire
    buf = io.BytesIO()
    
    if fmt.upper() == "JPEG":
        # JPEG optimisé pour OCR
        img.save(buf, format="JPEG", quality=quality, 
                optimize=True, subsampling=0)  # subsampling=0 pour meilleure qualité
    else:
        # PNG avec compression
        img.save(buf, format="PNG", optimize=True, compress_level=6)
    
    stream = buf.getvalue()
    rect = fitz.Rect(0, 0, page_w, page_h)
    page.insert_image(rect, stream=stream)


# --------------------- Core ---------------------

def process_page(rgb: np.ndarray, cfg: PreoptConfig) -> np.ndarray:
    """Pipeline complet de traitement d'une page."""
    
    # 1. Conversion en niveaux de gris
    gray = _to_gray(rgb)
    
    # 2. Normalisation de l'arrière-plan
    if cfg.normalize_background:
        gray = normalize_background(gray)
    
    # 3. Débruitage
    if cfg.denoise_h > 0:
        gray = denoise(gray, cfg.denoise_h)
    
    # 4. Amélioration du contraste (CLAHE)
    if cfg.clahe:
        gray = clahe_contrast(gray, cfg.clahe_clip, cfg.clahe_tile)
    
    # 5. Correction gamma
    gray = apply_gamma(gray, cfg.gamma)
    
    # 6. Netteté initiale
    if cfg.unsharp > 0:
        gray = unsharp_mask(gray, cfg.unsharp, cfg.unsharp_radius)
    
    # 7. Déconvolution Richardson-Lucy (optionnel)
    if cfg.deblur_rl_iter > 0:
        gray = apply_rl_deblur(gray, cfg.deblur_rl_psf, cfg.deblur_rl_iter)
    
    # 8. Binarisation et traitement géométrique
    if cfg.binarize != "none":
        binary = binarize(gray, cfg.binarize, cfg.adaptive_block, cfg.adaptive_C)
        
        # Nettoyage des taches
        if cfg.remove_speckles_area > 0:
            binary = clean_speckles(binary, cfg.remove_speckles_area)
        
        # Correction d'inclinaison (deskew)
        if cfg.deskew:
            angle = estimate_skew(binary, cfg.max_skew_deg)
            if abs(angle) > 0.1:
                binary = rotate(binary, angle)
                gray = rotate(gray, angle)
        
        # Morphologie
        binary = morphology(binary, cfg.morph_open, cfg.morph_close)
        
        # Recadrage
        if cfg.crop:
            x0, y0, x1, y1 = crop_to_content(binary, cfg.crop_margin)
            binary = binary[y0:y1, x0:x1]
            gray = gray[y0:y1, x0:x1]
        
        # Sortie selon le mode
        if cfg.output_color == "binary":
            final = binary
        else:
            final = gray
    
    else:
        # Mode sans binarisation : deskew et crop avec seuillage temporaire
        temp_binary = binarize(gray, "otsu", cfg.adaptive_block, cfg.adaptive_C)
        
        if cfg.deskew:
            angle = estimate_skew(temp_binary, cfg.max_skew_deg)
            if abs(angle) > 0.1:
                gray = rotate(gray, angle)
                temp_binary = rotate(temp_binary, angle)
        
        if cfg.crop:
            x0, y0, x1, y1 = crop_to_content(temp_binary, cfg.crop_margin)
            gray = gray[y0:y1, x0:x1]
        
        final = gray
    
    # 9. Netteté finale (post-traitement)
    if cfg.post_unsharp > 0 and cfg.output_color != "binary":
        final = unsharp_mask(final, cfg.post_unsharp, cfg.post_unsharp_radius)
    
    return final


def preoptimize_pdf(input_path: str, output_path: str, cfg: PreoptConfig, 
                   pages_expr: str = "", verbose: bool = False):
    """Fonction principale de pré-optimisation PDF."""
    
    in_doc = fitz.open(input_path)
    out_doc = fitz.open()
    total = len(in_doc)
    selection = parse_pages_selection(pages_expr, total)
    
    # DPI de rendu (avec super-sampling)
    dpi_render = int(round(cfg.dpi * max(1.0, cfg.super_sample)))
    
    if verbose:
        print(f"[INFO] Entrée : {input_path}")
        print(f"[INFO] Sortie : {output_path}")
        print(f"[INFO] Pages : {len(selection)}/{total}")
        print(f"[INFO] DPI target={cfg.dpi}, super-sample={cfg.super_sample:.1f} "
              f"-> DPI rendu={dpi_render}")
        print(f"[INFO] Mode={cfg.output_color}, RL={cfg.deblur_rl_iter}it, "
              f"post_unsharp={cfg.post_unsharp:.1f}")
    
    try:
        for j, page_idx in enumerate(selection, 1):
            page = in_doc.load_page(page_idx)
            
            # Conservation des pages avec texte natif
            if cfg.keep_text_pages:
                text = page.get_text("text").strip()
                if text and len(text) > 50:  # Seuil minimal
                    out_doc.insert_pdf(in_doc, from_page=page_idx, to_page=page_idx)
                    if verbose:
                        print(f"[PAGE {page_idx+1}/{total}] Texte natif → copie directe")
                    continue
            
            # Rasterisation
            if verbose:
                print(f"[PAGE {page_idx+1}/{total}] Traitement...", end=" ", flush=True)
            
            rgb = render_page_rgb(page, dpi_render)
            
            # Traitement
            processed = process_page(rgb, cfg)
            
            # Downscale si super-sampling activé
            if cfg.super_sample > 1.0:
                processed = downscale(processed, cfg.super_sample)
            
            # Conversion en PIL Image
            if processed.ndim == 2:
                pil_img = Image.fromarray(processed, mode="L")
            else:
                pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            
            # Format de sortie
            fmt = "PNG" if cfg.output_color == "binary" else cfg.image_format
            insert_image_page(out_doc, pil_img, cfg.dpi, fmt, cfg.jpeg_quality)
            
            if verbose:
                print("✓")
    
    finally:
        # Sauvegarde avec compression
        try:
            out_doc.save(output_path, deflate=True, garbage=4, clean=True)
        except Exception as e:
            if verbose:
                print(f"[WARN] Sauvegarde optimale échouée, mode basique: {e}")
            out_doc.save(output_path)
        
        out_doc.close()
        in_doc.close()


# --------------------- CLI ---------------------

def parse_pages_selection(pages: str, total: int) -> List[int]:
    """Parse l'expression de sélection de pages."""
    if not pages:
        return list(range(total))
    
    selected = set()
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str) if start_str.strip() else 1
            end = int(end_str) if end_str.strip() else total
            
            for p in range(start, end + 1):
                if 1 <= p <= total:
                    selected.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= total:
                selected.add(p - 1)
    
    return sorted(selected)


def parse_args():
    """Parse les arguments de ligne de commande."""
    p = argparse.ArgumentParser(
        description="Pré-optimisation PDF pour OCR avec netteté améliorée",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  # Mode GUI
  python preopt_pdf_sharp.py --gui
  
  # Ligne de commande basique
  python preopt_pdf_sharp.py --input doc.pdf --output doc_opt.pdf
  
  # Avec super-sampling et netteté post-traitement
  python preopt_pdf_sharp.py --input doc.pdf --output doc_opt.pdf \\
    --super-sample 1.5 --post-unsharp 0.5
  
  # Avec déconvolution RL
  python preopt_pdf_sharp.py --input doc.pdf --output doc_opt.pdf \\
    --deblur-rl-iter 10 --deblur-rl-psf 5
  
  # Pages spécifiques
  python preopt_pdf_sharp.py --input doc.pdf --output doc_opt.pdf \\
    --pages "1-5,10,15-20"
        """
    )
    
    # Arguments principaux
    p.add_argument("--input", help="Fichier PDF d'entrée")
    p.add_argument("--output", help="Fichier PDF de sortie")
    p.add_argument("--gui", action="store_true", help="Mode interface graphique")
    p.add_argument("--dpi", type=int, default=400, help="Résolution cible (défaut: 400)")
    p.add_argument("--pages", type=str, default="", 
                   help="Pages à traiter (ex: '1-5,10,15-20')")
    p.add_argument("--verbose", action="store_true", help="Mode verbeux")
    
    # Pipeline de base
    p.add_argument("--keep-text-pages", dest="keep_text_pages", 
                   action="store_true", default=True,
                   help="Conserver les pages avec texte natif (défaut)")
    p.add_argument("--no-keep-text-pages", dest="keep_text_pages", 
                   action="store_false")
    
    p.add_argument("--normalize-background", dest="normalize_background",
                   action="store_true", default=True,
                   help="Normaliser l'arrière-plan (défaut)")
    p.add_argument("--no-normalize-background", dest="normalize_background",
                   action="store_false")
    
    p.add_argument("--denoise-h", type=int, default=10,
                   help="Niveau de débruitage (0=off, défaut: 10)")
    
    p.add_argument("--clahe", action="store_true", default=True,
                   help="Activer CLAHE (défaut)")
    p.add_argument("--no-clahe", dest="clahe", action="store_false")
    p.add_argument("--clahe-clip", type=float, default=2.0,
                   help="CLAHE clip limit (défaut: 2.0)")
    p.add_argument("--clahe-tile", type=int, default=8,
                   help="CLAHE tile grid size (défaut: 8)")
    
    p.add_argument("--gamma", type=float, default=1.05,
                   help="Correction gamma (défaut: 1.05)")
    
    p.add_argument("--unsharp", type=float, default=0.8,
                   help="Netteté initiale (0=off, défaut: 0.8)")
    p.add_argument("--unsharp-radius", type=float, default=1.0,
                   help="Rayon netteté initiale (défaut: 1.0)")
    
    p.add_argument("--binarize", choices=["none", "adaptive", "otsu"], 
                   default="none",
                   help="Méthode de binarisation (défaut: none)")
    p.add_argument("--adaptive-block", type=int, default=25,
                   help="Taille bloc adaptive threshold (défaut: 25)")
    p.add_argument("--adaptive-C", type=int, default=10,
                   help="Constante adaptive threshold (défaut: 10)")
    
    p.add_argument("--remove-speckles-area", type=int, default=12,
                   help="Aire min pour garder composante (défaut: 12)")
    
    p.add_argument("--morph-open", type=int, default=2,
                   help="Taille kernel morpho opening (défaut: 2)")
    p.add_argument("--morph-close", type=int, default=2,
                   help="Taille kernel morpho closing (défaut: 2)")
    
    p.add_argument("--deskew", action="store_true", default=True,
                   help="Corriger l'inclinaison (défaut)")
    p.add_argument("--no-deskew", dest="deskew", action="store_false")
    p.add_argument("--max-skew-deg", type=float, default=10.0,
                   help="Angle max de correction (défaut: 10.0°)")
    
    p.add_argument("--crop", action="store_true", default=True,
                   help="Recadrer au contenu (défaut)")
    p.add_argument("--no-crop", dest="crop", action="store_false")
    p.add_argument("--crop-margin", type=int, default=5,
                   help="Marge de recadrage en pixels (défaut: 5)")
    
    p.add_argument("--output-color", choices=["gray", "binary"], default="gray",
                   help="Mode de sortie (défaut: gray)")
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="Qualité JPEG (40-95, défaut: 85)")
    
    # Netteté avancée
    p.add_argument("--super-sample", type=float, default=1.0,
                   help="Super-sampling (1.0=off, 1.5-2.0 recommandé)")
    p.add_argument("--post-unsharp", type=float, default=0.0,
                   help="Netteté post-traitement (0=off, 0.3-0.8 conseillé)")
    p.add_argument("--post-unsharp-radius", type=float, default=0.8,
                   help="Rayon netteté post (défaut: 0.8)")
    p.add_argument("--deblur-rl-iter", type=int, default=0,
                   help="Iterations Richardson-Lucy (0=off, 5-15 utile)")
    p.add_argument("--deblur-rl-psf", type=int, default=3,
                   help="Taille PSF Richardson-Lucy (impair, défaut: 3)")
    
    args = p.parse_args()
    
    # Mode GUI si demandé ou si arguments manquants
    if args.gui or not args.input or not args.output:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
            
            root = tk.Tk()
            root.withdraw()
            
            if not args.input:
                args.input = filedialog.askopenfilename(
                    title="Sélectionnez le PDF d'entrée",
                    filetypes=[("PDF", "*.pdf"), ("Tous", "*.*")]
                )
            
            if not args.input:
                messagebox.showerror("Erreur", "Aucun fichier d'entrée sélectionné")
                return None
            
            if not args.output:
                base, _ = os.path.splitext(args.input)
                default_name = os.path.basename(base) + "_sharp.pdf"
                
                args.output = filedialog.asksaveasfilename(
                    title="Enregistrer le PDF optimisé",
                    defaultextension=".pdf",
                    initialfile=default_name,
                    filetypes=[("PDF", "*.pdf")]
                )
            
            if not args.output:
                messagebox.showerror("Erreur", "Aucun fichier de sortie spécifié")
                return None
            
            root.destroy()
            
        except ImportError:
            # Pas de tkinter : mode console
            if not args.input:
                args.input = input("Chemin du PDF d'entrée: ").strip().strip('"\'')
            if not args.output:
                args.output = input("Chemin du PDF de sortie: ").strip().strip('"\'')
    
    # Vérifications
    if not os.path.exists(args.input):
        p.error(f"Fichier d'entrée introuvable: {args.input}")
    
    # Éviter écrasement
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_sharp{ext}"
        if args.verbose:
            print(f"[WARN] Sortie renommée en: {args.output}")
    
    # Construction de la config
    cfg = PreoptConfig(
        dpi=max(150, min(600, args.dpi)),
        keep_text_pages=args.keep_text_pages,
        normalize_background=args.normalize_background,
        denoise_h=max(0, args.denoise_h),
        clahe=args.clahe,
        clahe_clip=max(1.0, min(4.0, args.clahe_clip)),
        clahe_tile=max(4, min(16, args.clahe_tile)),
        gamma=max(0.5, min(2.0, args.gamma)),
        unsharp=max(0.0, min(2.0, args.unsharp)),
        unsharp_radius=max(0.1, min(5.0, args.unsharp_radius)),
        binarize=args.binarize,
        adaptive_block=max(3, args.adaptive_block),
        adaptive_C=args.adaptive_C,
        remove_speckles_area=max(0, args.remove_speckles_area),
        morph_open=max(0, args.morph_open),
        morph_close=max(0, args.morph_close),
        deskew=args.deskew,
        max_skew_deg=max(0.0, min(45.0, args.max_skew_deg)),
        crop=args.crop,
        crop_margin=max(0, args.crop_margin),
        output_color=args.output_color,
        image_format="JPEG" if args.output_color == "gray" else "PNG",
        jpeg_quality=max(40, min(95, args.jpeg_quality)),
        super_sample=max(1.0, min(3.0, args.super_sample)),
        post_unsharp=max(0.0, min(2.0, args.post_unsharp)),
        post_unsharp_radius=max(0.1, min(5.0, args.post_unsharp_radius)),
        deblur_rl_iter=max(0, min(30, args.deblur_rl_iter)),
        deblur_rl_psf=max(3, args.deblur_rl_psf),
    )
    
    return args.input, args.output, cfg, args.pages, args.verbose


def main():
    """Point d'entrée principal."""
    result = parse_args()
    if result is None:
        return 1
    
    input_path, output_path, cfg, pages_expr, verbose = result
    
    try:
        preoptimize_pdf(input_path, output_path, cfg, pages_expr, verbose)
        
        if verbose:
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"\n✅ PDF optimisé créé avec succès!")
            print(f"   Fichier: {output_path}")
            print(f"   Taille: {file_size:.2f} MB")
        else:
            print(f"✅ {output_path}")
        
        return 0
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
