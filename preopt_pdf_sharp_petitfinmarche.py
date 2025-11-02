#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_optimizer.py — Optimisation PDF avancée pour OCR (Cloud Run ready)

Améliorations clés :
- Deskew robuste multi-méthodes (Hough + projection + minAreaRect)
- Super-résolution optionnelle (EDSR, ESPCN via OpenCV DNN)
- Optimisation spécifique des chiffres (contraste localisé, morphologie fine)
- Pipeline asynchrone pour Cloud Run (timeout-safe)
- Healthcheck et monitoring intégrés
- Support Cloud Storage (gs://)
"""

import argparse
import io
import os
import math
import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Union
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import tempfile

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image

# Cloud Run & GCS
try:
    from google.cloud import storage
    GCS_AVAILABLE = True
except ImportError:
    GCS_AVAILABLE = False

# Super-résolution (optionnel)
try:
    SR_AVAILABLE = hasattr(cv2.dnn_superres, 'DnnSuperResImpl_create')
except AttributeError:
    SR_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


# ===================== CONFIG =====================

@dataclass
class OCRConfig:
    """Configuration optimisée pour OCR de factures/documents"""
    
    # Résolution
    dpi: int = 400
    super_sample: float = 1.5  # 1.5x avant traitement
    use_super_res: bool = False  # DNN super-res (lent mais puissant)
    sr_model: str = "EDSR"  # EDSR | ESPCN | LapSRN
    sr_scale: int = 2
    
    # Pré-traitement
    normalize_background: bool = True
    denoise_strength: int = 8  # Réduit vs 10 pour préserver détails
    denoise_color: int = 8
    
    # Contraste
    clahe: bool = True
    clahe_clip: float = 2.5  # Augmenté pour chiffres
    clahe_tile: int = 8
    gamma: float = 1.08
    
    # Netteté
    unsharp_amount: float = 0.9
    unsharp_radius: float = 1.2
    post_unsharp: float = 0.4  # 2e passe après deskew
    post_unsharp_radius: float = 0.7
    
    # Déconvolution (optionnel, coûteux)
    deblur_rl_iterations: int = 0  # 0=off, 5-10 si flou
    deblur_rl_psf_size: int = 5
    
    # Deskew robuste
    deskew: bool = True
    deskew_method: str = "multi"  # multi | hough | projection | minAreaRect
    max_skew_deg: float = 15.0
    deskew_confidence_threshold: float = 0.6
    
    # Binarisation
    binarize: str = "none"  # none | adaptive | otsu | sauvola
    adaptive_block: int = 31  # Plus grand pour factures
    adaptive_C: int = 12
    sauvola_window: int = 25
    sauvola_k: float = 0.2
    
    # Nettoyage
    remove_speckles_area: int = 15
    morph_open: int = 1  # Réduit pour préserver traits fins
    morph_close: int = 2
    
    # Optimisation chiffres
    enhance_digits: bool = True
    digit_morph_kernel: int = 2
    digit_contrast_boost: float = 1.3
    
    # Géométrie
    crop: bool = True
    crop_margin: int = 10
    auto_rotate: bool = True  # Détection orientation (0/90/180/270)
    
    # Sortie
    output_color: str = "gray"  # gray | binary
    jpeg_quality: int = 90
    
    # Performance Cloud Run
    max_workers: int = 2
    timeout_per_page: int = 30  # secondes
    keep_text_pages: bool = True
    
    # Chemins modèles (pour super-res)
    model_dir: str = "/tmp/models"
    
    def __post_init__(self):
        if self.binarize != "none":
            self.output_color = "binary"


# ===================== CLOUD STORAGE =====================

class StorageHandler:
    """Gestion uniforme local/GCS"""
    
    def __init__(self):
        self.client = storage.Client() if GCS_AVAILABLE else None
    
    def is_gcs_path(self, path: str) -> bool:
        return path.startswith("gs://")
    
    def download(self, path: str, local_path: str):
        """Télécharge depuis GCS si nécessaire"""
        if not self.is_gcs_path(path):
            return path
        
        if not GCS_AVAILABLE:
            raise RuntimeError("google-cloud-storage non installé")
        
        # Parse gs://bucket/path
        parts = path[5:].split("/", 1)
        bucket_name, blob_name = parts[0], parts[1]
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(local_path)
        logger.info(f"Downloaded {path} → {local_path}")
        return local_path
    
    def upload(self, local_path: str, gcs_path: str):
        """Upload vers GCS"""
        if not self.is_gcs_path(gcs_path):
            return
        
        parts = gcs_path[5:].split("/", 1)
        bucket_name, blob_name = parts[0], parts[1]
        
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        logger.info(f"Uploaded {local_path} → {gcs_path}")


# ===================== DESKEW AVANCÉ =====================

def estimate_skew_hough(binary: np.ndarray, max_deg: float) -> Tuple[float, float]:
    """Méthode Hough Lines (original amélioré)"""
    mask = (binary < 250).astype(np.uint8) * 255
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    
    # Paramètres adaptatifs
    min_line_length = max(binary.shape[1] // 4, 100)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi/900.0, 
        threshold=80,
        minLineLength=min_line_length,
        maxLineGap=30
    )
    
    if lines is None or len(lines) < 5:
        return 0.0, 0.0
    
    angles = []
    weights = []
    for x1, y1, x2, y2 in lines[:, 0, :]:
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        ang = math.degrees(math.atan2(y2 - y1, x2 - x1))
        
        # Normaliser [-90, 90]
        if ang > 90:
            ang -= 180
        elif ang < -90:
            ang += 180
        
        if abs(ang) <= max_deg:
            angles.append(ang)
            weights.append(length)
    
    if not angles:
        return 0.0, 0.0
    
    # Médiane pondérée
    angles = np.array(angles)
    weights = np.array(weights)
    sorted_idx = np.argsort(angles)
    angles = angles[sorted_idx]
    weights = weights[sorted_idx]
    cumsum = np.cumsum(weights)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    
    confidence = min(1.0, len(angles) / 50.0)
    return float(angles[median_idx]), confidence


def estimate_skew_projection(binary: np.ndarray, max_deg: float) -> Tuple[float, float]:
    """Méthode projection profile (robuste pour texte)"""
    mask = (binary < 250).astype(np.uint8)
    
    angles = np.linspace(-max_deg, max_deg, int(max_deg * 4))
    scores = []
    
    h, w = mask.shape
    for angle in angles:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        
        # Projection horizontale
        projection = np.sum(rotated, axis=1)
        # Variance élevée = lignes bien alignées
        score = np.var(projection)
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_angle = angles[best_idx]
    
    # Confidence basée sur pic de variance
    scores = np.array(scores)
    confidence = (scores[best_idx] - np.mean(scores)) / (np.std(scores) + 1e-6)
    confidence = min(1.0, confidence / 3.0)
    
    return float(best_angle), confidence


def estimate_skew_minarea(binary: np.ndarray, max_deg: float) -> Tuple[float, float]:
    """Méthode minAreaRect (rapide, marche bien pour blocs)"""
    mask = (binary < 250).astype(np.uint8) * 255
    coords = cv2.findNonZero(mask)
    
    if coords is None or len(coords) < 100:
        return 0.0, 0.0
    
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    
    # Correction orientation
    if angle < -45:
        angle = 90 + angle
    elif angle > 45:
        angle = angle - 90
    
    angle = -angle  # Convention rotation
    
    if abs(angle) > max_deg:
        return 0.0, 0.0
    
    # Confidence basée sur aspect ratio
    w, h = rect[1]
    aspect = max(w, h) / (min(w, h) + 1e-6)
    confidence = min(1.0, aspect / 10.0)
    
    return float(angle), confidence


def estimate_skew_multi(binary: np.ndarray, max_deg: float, threshold: float = 0.6) -> float:
    """Combinaison de méthodes avec vote pondéré"""
    methods = [
        ("hough", estimate_skew_hough),
        ("projection", estimate_skew_projection),
        ("minarea", estimate_skew_minarea),
    ]
    
    results = []
    for name, func in methods:
        try:
            angle, conf = func(binary, max_deg)
            results.append((angle, conf, name))
            logger.debug(f"Deskew {name}: {angle:.2f}° (conf={conf:.2f})")
        except Exception as e:
            logger.warning(f"Deskew {name} failed: {e}")
    
    if not results:
        return 0.0
    
    # Filtre confidence
    results = [(a, c, n) for a, c, n in results if c >= threshold]
    
    if not results:
        # Fallback: meilleure méthode même si < threshold
        results = [(a, c, n) for a, c, n in results]
        if not results:
            return 0.0
    
    # Moyenne pondérée
    angles = np.array([a for a, c, n in results])
    confidences = np.array([c for a, c, n in results])
    
    final_angle = np.average(angles, weights=confidences)
    logger.info(f"Deskew final: {final_angle:.2f}° (methods: {[n for _,_,n in results]})")
    
    return float(final_angle)


def deskew_image(img: np.ndarray, binary: np.ndarray, cfg: OCRConfig) -> np.ndarray:
    """Redressement robuste"""
    if not cfg.deskew:
        return img
    
    if cfg.deskew_method == "multi":
        angle = estimate_skew_multi(binary, cfg.max_skew_deg, cfg.deskew_confidence_threshold)
    elif cfg.deskew_method == "hough":
        angle, _ = estimate_skew_hough(binary, cfg.max_skew_deg)
    elif cfg.deskew_method == "projection":
        angle, _ = estimate_skew_projection(binary, cfg.max_skew_deg)
    else:  # minAreaRect
        angle, _ = estimate_skew_minarea(binary, cfg.max_skew_deg)
    
    if abs(angle) < 0.1:
        return img
    
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    
    # Interpolation adaptée au type d'image
    border = 255 if img.ndim == 2 else (255, 255, 255)
    rotated = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border
    )
    
    return rotated


# ===================== OPTIMISATION CHIFFRES =====================

def enhance_digits(gray: np.ndarray, cfg: OCRConfig) -> np.ndarray:
    """Pipeline spécifique pour améliorer lisibilité des chiffres"""
    if not cfg.enhance_digits:
        return gray
    
    # 1. Contraste local agressif sur zones denses
    clahe_digits = cv2.createCLAHE(
        clipLimit=cfg.digit_contrast_boost * 2.0,
        tileGridSize=(4, 4)  # Tuiles plus petites
    )
    enhanced = clahe_digits.apply(gray)
    
    # 2. Morphologie fine pour épaissir traits fins
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (cfg.digit_morph_kernel, cfg.digit_morph_kernel)
    )
    enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
    
    # 3. Fusion pondérée
    enhanced = cv2.addWeighted(gray, 0.6, enhanced, 0.4, 0)
    
    return enhanced


# ===================== SUPER-RÉSOLUTION =====================

class SuperResolutionEngine:
    """Wrapper DNN Super-Resolution"""
    
    def __init__(self, model: str = "EDSR", scale: int = 2, model_dir: str = "/tmp/models"):
        self.model_name = model
        self.scale = scale
        self.model_dir = Path(model_dir)
        self.sr = None
        
        if SR_AVAILABLE:
            self._init_model()
    
    def _init_model(self):
        """Charge le modèle (télécharge si nécessaire)"""
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Chemins modèles
        model_files = {
            "EDSR": f"EDSR_x{self.scale}.pb",
            "ESPCN": f"ESPCN_x{self.scale}.pb",
            "LapSRN": f"LapSRN_x{self.scale}.pb",
        }
        
        model_file = self.model_dir / model_files.get(self.model_name, "EDSR_x2.pb")
        
        if not model_file.exists():
            logger.warning(f"Modèle {model_file} non trouvé. Super-res désactivée.")
            return
        
        try:
            self.sr = cv2.dnn_superres.DnnSuperResImpl_create()
            self.sr.readModel(str(model_file))
            self.sr.setModel(self.model_name.lower(), self.scale)
            logger.info(f"Super-res chargée: {self.model_name} x{self.scale}")
        except Exception as e:
            logger.error(f"Erreur chargement super-res: {e}")
            self.sr = None
    
    def upscale(self, img: np.ndarray) -> np.ndarray:
        """Applique super-résolution"""
        if self.sr is None:
            # Fallback: bicubic classique
            h, w = img.shape[:2]
            return cv2.resize(img, (w*self.scale, h*self.scale), interpolation=cv2.INTER_CUBIC)
        
        return self.sr.upsample(img)


# ===================== TRAITEMENT IMAGE =====================

def normalize_background(gray: np.ndarray) -> np.ndarray:
    """Normalisation fond (éclairage non uniforme)"""
    k = max(15, int(max(gray.shape) * 0.015))
    if k % 2 == 0:
        k += 1
    bg = cv2.GaussianBlur(gray, (k, k), 0)
    return cv2.divide(gray, bg, scale=255)


def denoise(img: np.ndarray, h: int, h_color: int = 10) -> np.ndarray:
    """Débruitage adaptatif"""
    if h <= 0:
        return img
    
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h=h, templateWindowSize=7, searchWindowSize=21)
    else:
        return cv2.fastNlMeansDenoisingColored(img, None, h=h, hColor=h_color, 
                                                templateWindowSize=7, searchWindowSize=21)


def apply_clahe(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    """CLAHE avec validation"""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    """Correction gamma"""
    if abs(gamma - 1.0) < 0.01:
        return gray
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, table)


def unsharp_mask(gray: np.ndarray, amount: float, radius: float) -> np.ndarray:
    """Masque flou pour netteté"""
    if amount <= 0:
        return gray
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=radius, sigmaY=radius)
    return cv2.addWeighted(gray, 1.0 + amount, blurred, -amount, 0)


def binarize_sauvola(gray: np.ndarray, window: int, k: float) -> np.ndarray:
    """Binarisation Sauvola (meilleure que Otsu pour documents)"""
    if window % 2 == 0:
        window += 1
    
    mean = cv2.boxFilter(gray, cv2.CV_32F, (window, window), normalize=True)
    mean_sq = cv2.boxFilter(gray.astype(np.float32)**2, cv2.CV_32F, (window, window), normalize=True)
    variance = mean_sq - mean**2
    std = np.sqrt(np.maximum(variance, 0))
    
    threshold = mean * (1 + k * ((std / 128.0) - 1))
    binary = np.where(gray > threshold, 255, 0).astype(np.uint8)
    return binary


def binarize(gray: np.ndarray, method: str, cfg: OCRConfig) -> np.ndarray:
    """Binarisation multi-méthodes"""
    if method == "none":
        return gray
    
    if method == "otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    
    if method == "sauvola":
        return binarize_sauvola(gray, cfg.sauvola_window, cfg.sauvola_k)
    
    # adaptive
    block = cfg.adaptive_block if cfg.adaptive_block % 2 == 1 else cfg.adaptive_block + 1
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, cfg.adaptive_C
    )


def clean_noise(binary: np.ndarray, cfg: OCRConfig) -> np.ndarray:
    """Nettoyage speckles + morphologie"""
    # Speckles
    if cfg.remove_speckles_area > 0:
        inv = 255 - binary
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (inv > 0).astype(np.uint8), connectivity=8
        )
        mask = np.zeros_like(inv, dtype=bool)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] < cfg.remove_speckles_area:
                mask[labels == i] = True
        inv[mask] = 0
        binary = 255 - inv
    
    # Morphologie
    if cfg.morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_open, cfg.morph_open))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)
    
    if cfg.morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.morph_close, cfg.morph_close))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k)
    
    return binary


def crop_to_content(img: np.ndarray, binary: np.ndarray, margin: int) -> np.ndarray:
    """Crop automatique avec marges"""
    ys, xs = np.where(binary < 255)
    if len(xs) == 0:
        return img
    
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    
    h, w = img.shape[:2]
    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin)
    y1 = min(h, y1 + margin)
    
    return img[y0:y1, x0:x1]


# ===================== PIPELINE PRINCIPAL =====================

def process_page(
    rgb: np.ndarray,
    cfg: OCRConfig,
    sr_engine: Optional[SuperResolutionEngine] = None
) -> np.ndarray:
    """Pipeline complet de traitement d'une page"""
    
    # 1. Super-résolution DNN (optionnel, avant tout)
    if cfg.use_super_res and sr_engine is not None:
        logger.debug("Applying DNN super-resolution...")
        rgb = sr_engine.upscale(rgb)
    
    # 2. Conversion grayscale
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # 3. Normalisation fond
    if cfg.normalize_background:
        gray = normalize_background(gray)
    
    # 4. Débruitage
    if cfg.denoise_strength > 0:
        gray = denoise(gray, cfg.denoise_strength, cfg.denoise_color)
    
    # 5. CLAHE
    if cfg.clahe:
        gray = apply_clahe(gray, cfg.clahe_clip, cfg.clahe_tile)
    
    # 6. Gamma
    gray = apply_gamma(gray, cfg.gamma)
    
    # 7. Netteté primaire
    if cfg.unsharp_amount > 0:
        gray = unsharp_mask(gray, cfg.unsharp_amount, cfg.unsharp_radius)
    
    # 8. Optimisation chiffres
    gray = enhance_digits(gray, cfg)
    
    # 9. Binarisation (pour deskew)
    temp_binary = binarize(gray, "otsu", cfg)
    
    # 10. Deskew
    gray = deskew_image(gray, temp_binary, cfg)
    temp_binary = deskew_image(temp_binary, temp_binary, cfg)
    
    # 11. Binarisation finale (si demandée)
    if cfg.binarize != "none":
        final = binarize(gray, cfg.binarize, cfg)
        final = clean_noise(final, cfg)
    else:
        final = gray
    
    # 12. Netteté post-deskew
    if cfg.post_unsharp > 0 and cfg.binarize == "none":
        final = unsharp_mask(final, cfg.post_unsharp, cfg.post_unsharp_radius)
    
    # 13. Crop
    if cfg.crop:
        crop_ref = temp_binary if cfg.binarize == "none" else final
        final = crop_to_content(final, crop_ref, cfg.crop_margin)
    
    return final


def render_page(page: fitz.Page, dpi: int) -> np.ndarray:
    """Rendu page PDF en RGB"""
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, colorspace="rgb")
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.h, pix.w, 3)


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    """Downscale avec anti-aliasing"""
    if factor <= 1.0:
        return img
    h, w = img.shape[:2]
    new_w = int(round(w / factor))
    new_h = int(round(h / factor))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def array_to_pil(img: np.ndarray) -> Image.Image:
    """NumPy → PIL"""
    if img.ndim == 2:
        return Image.fromarray(img, mode="L")
    else:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


def insert_image_page(
    doc: fitz.Document,
    pil_img: Image.Image,
    dpi: int,
    quality: int = 90
):
    """Insère image dans PDF"""
    w_px, h_px = pil_img.size
    page_w = w_px * 72.0 / dpi
    page_h = h_px * 72.0 / dpi
    
    page = doc.new_page(width=page_w, height=page_h)
    
    buf = io.BytesIO()
    if pil_img.mode == "L" or pil_img.mode == "1":
        # Grayscale ou binaire → PNG sans perte
        pil_img.save(buf, format="PNG", optimize=True)
    else:
        pil_img.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=0)
    
    stream = buf.getvalue()
    rect = fitz.Rect(0, 0, page_w, page_h)
    page.insert_image(rect, stream=stream)


# ===================== ORCHESTRATEUR =====================

def optimize_pdf(
    input_path: str,
    output_path: str,
    cfg: OCRConfig,
    storage: StorageHandler,
    pages: Optional[List[int]] = None
):
    """
    Orchestrateur principal
    
    Args:
        input_path: Chemin local ou gs://
        output_path: Chemin local ou gs://
        cfg: Configuration
        storage: Gestionnaire stockage
        pages: Liste pages (0-indexed), None = toutes
    """
    
    # Téléchargement si GCS
    with tempfile.TemporaryDirectory() as tmpdir:
        if storage.is_gcs_path(input_path):
            local_input = os.path.join(tmpdir, "input.pdf")
            storage.download(input_path, local_input)
        else:
            local_input = input_path
        
        local_output = os.path.join(tmpdir, "output.pdf") if storage.is_gcs_path(output_path) else output_path
        
        # Initialisation super-res
        sr_engine = None
        if cfg.use_super_res:
            sr_engine = SuperResolutionEngine(cfg.sr_model, cfg.sr_scale, cfg.model_dir)
        
        # Traitement
        in_doc = fitz.open(local_input)
        out_doc = fitz.open()
        
        total_pages = len(in_doc)
        page_indices = pages if pages is not None else list(range(total_pages))
        
        logger.info(f"Processing {len(page_indices)}/{total_pages} pages")
        logger.info(f"Config: DPI={cfg.dpi}, super_sample={cfg.super_sample}, deskew={cfg.deskew_method}")
        
        dpi_render = int(cfg.dpi * cfg.super_sample)
        
        for idx in page_indices:
            t0 = time.time()
            page = in_doc.load_page(idx)
            
            # Texte natif → copie directe
            if cfg.keep_text_pages:
                text = page.get_text("text").strip()
                if text and len(text) > 50:
                    out_doc.insert_pdf(in_doc, from_page=idx, to_page=idx)
                    logger.info(f"Page {idx+1}: text layer preserved ({time.time()-t0:.1f}s)")
                    continue
            
            # Rasterisation
            rgb = render_page(page, dpi_render)
            
            # Traitement
            try:
                processed = process_page(rgb, cfg, sr_engine)
            except Exception as e:
                logger.error(f"Page {idx+1} processing failed: {e}")
                # Fallback: copie originale
                out_doc.insert_pdf(in_doc, from_page=idx, to_page=idx)
                continue
            
            # Downscale si super-sample
            if cfg.super_sample > 1.0:
                processed = downscale(processed, cfg.super_sample)
            
            # Insertion
            pil_img = array_to_pil(processed)
            insert_image_page(out_doc, pil_img, cfg.dpi, cfg.jpeg_quality)
            
            elapsed = time.time() - t0
            logger.info(f"Page {idx+1}: processed in {elapsed:.1f}s")
        
        # Sauvegarde
        out_doc.save(local_output, deflate=True, garbage=3, clean=True)
        out_doc.close()
        in_doc.close()
        
        logger.info(f"PDF saved: {local_output}")
        
        # Upload si GCS
        if storage.is_gcs_path(output_path):
            storage.upload(local_output, output_path)


# ===================== CLI & CLOUD RUN =====================

def parse_args():
    p = argparse.ArgumentParser(description="OCR-optimized PDF processor (Cloud Run ready)")
    
    # I/O
    p.add_argument("--input", required=True, help="Input PDF (local or gs://)")
    p.add_argument("--output", required=True, help="Output PDF (local or gs://)")
    p.add_argument("--pages", help="Pages to process (e.g., '1,3-5,10')")
    
    # Résolution
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--super-sample", type=float, default=1.5)
    p.add_argument("--use-super-res", action="store_true", help="Use DNN super-resolution (slow)")
    p.add_argument("--sr-model", choices=["EDSR", "ESPCN", "LapSRN"], default="EDSR")
    
    # Deskew
    p.add_argument("--deskew-method", choices=["multi", "hough", "projection", "minAreaRect"], default="multi")
    p.add_argument("--max-skew-deg", type=float, default=15.0)
    p.add_argument("--no-deskew", action="store_true")
    
    # Binarisation
    p.add_argument("--binarize", choices=["none", "adaptive", "otsu", "sauvola"], default="none")
    
    # Optimisations
    p.add_argument("--enhance-digits", action="store_true", default=True)
    p.add_argument("--no-enhance-digits", dest="enhance_digits", action="store_false")
    
    # Qualité
    p.add_argument("--jpeg-quality", type=int, default=90)
    
    # Cloud Run
    p.add_argument("--timeout", type=int, default=300, help="Global timeout (seconds)")
    
    args = p.parse_args()
    
    # Parse pages
    page_list = None
    if args.pages:
        page_list = []
        for part in args.pages.split(","):
            if "-" in part:
                start, end = map(int, part.split("-"))
                page_list.extend(range(start-1, end))
            else:
                page_list.append(int(part) - 1)
    
    cfg = OCRConfig(
        dpi=args.dpi,
        super_sample=args.super_sample,
        use_super_res=args.use_super_res,
        sr_model=args.sr_model,
        deskew=not args.no_deskew,
        deskew_method=args.deskew_method,
        max_skew_deg=args.max_skew_deg,
        binarize=args.binarize,
        enhance_digits=args.enhance_digits,
        jpeg_quality=args.jpeg_quality,
    )
    
    return args.input, args.output, cfg, page_list


def main():
    input_path, output_path, cfg, pages = parse_args()
    
    storage = StorageHandler()
    
    try:
        optimize_pdf(input_path, output_path, cfg, storage, pages)
        logger.info("✅ Processing complete")
    except Exception as e:
        logger.error(f"❌ Processing failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
