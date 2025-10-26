#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preopt_pdf_sharp.py — Pré-optimisation PDF pour OCR avec accent sur la netteté

Ajouts clés vs preopt_pdf.py :
- --super-sample : rendu à DPI x facteur puis redescale (anti-alias) -> plus net
- --post-unsharp  : 2e passe de netteté après tout le pipeline
- --deblur-rl     : optionnel, déconvolution Richardson–Lucy (si scikit-image dispo)
- --sharpen-level : profils de netteté (conservative|medium|hard)

Dépendances
-----------
pip install pymupdf opencv-python-headless pillow numpy
# (optionnel pour RL) : pip install scikit-image
"""

import argparse
import io
import os
import math
import time
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
    binarize: str = "none"             # "none" | "adaptive" | "otsu"
    adaptive_block: int = 25
    adaptive_C: int = 10
    remove_speckles_area: int = 12
    morph_open: int = 2
    morph_close: int = 2
    deskew: bool = True
    max_skew_deg: float = 10.0
    crop: bool = True
    crop_margin: int = 5

    # sortie
    output_color: str = "gray"         # "gray" | "binary"
    image_format: str = "JPEG"         # "JPEG" (binaire -> PNG)
    jpeg_quality: int = 85

    # NETTETE avancée
    super_sample: float = 1.0          # 1.0 = off; 1.5 ou 2.0 recommandé
    post_unsharp: float = 0.0          # 0=off ; 0.3–0.8 utile
    post_unsharp_radius: float = 0.8
    deblur_rl_iter: int = 0            # 0=off ; 5–15 max
    deblur_rl_psf: int = 3             # taille noyau RL (impair)


# --------------------- Utils ---------------------

def _make_odd(n: int, minimum: int = 3) -> int:
    n = max(n, minimum)
    return n if n % 2 == 1 else n + 1


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def normalize_background(gray: np.ndarray) -> np.ndarray:
    k = _make_odd(max(15, int(max(gray.shape) * 0.01)))
    bg = cv2.medianBlur(gray, k)
    return cv2.divide(gray, bg, scale=255)


def denoise(gray: np.ndarray, h: int) -> np.ndarray:
    if h <= 0:
        return gray
    return cv2.fastNlMeansDenoising(gray, None, h=h, templateWindowSize=7, searchWindowSize=21)


def clahe_contrast(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 1e-3:
        return gray
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.float32)
    return cv2.LUT(gray, lut.astype("uint8"))


def unsharp_mask(gray: np.ndarray, amount: float, radius: float) -> np.ndarray:
    if amount <= 0:
        return gray
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=radius, sigmaY=radius)
    return cv2.addWeighted(gray, 1.0 + amount, blur, -amount, 0)


def binarize(gray: np.ndarray, method: str, block: int, C: int) -> np.ndarray:
    if method == "none":
        return gray
    if method == "otsu":
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw
    # --- adaptive ---
    block = _make_odd(block)
    # constante correcte + fallback
    adaptive = getattr(cv2, "ADAPTIVE_THRESH_GAUSSIAN_C", None)
    if adaptive is None:
        adaptive = getattr(cv2, "ADAPTIVE_THRESH_MEAN_C", None)
    if adaptive is None:
        # Dernier recours : Otsu
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return bw
    return cv2.adaptiveThreshold(gray, 255, adaptive, cv2.THRESH_BINARY, block, int(C))

def clean_speckles(binary: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return binary
    fg = (255 - binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((fg > 0).astype(np.uint8), connectivity=8)
    remove_mask = np.zeros_like(fg, dtype=bool)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area < min_area:
            remove_mask[labels == lbl] = True
    fg[remove_mask] = 0
    return 255 - fg


def morphology(binary: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    out = binary
    if open_k and open_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k)
    if close_k and close_k > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k)
    return out


def estimate_skew(binary: np.ndarray, max_deg: float) -> float:
    mask = (binary < 250).astype(np.uint8) * 255
    edges = cv2.Canny(mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi/1800.0, threshold=100,
                            minLineLength=max(50, binary.shape[1]//3),
                            maxLineGap=20)
    angles = []
    if lines is not None:
        for x1,y1,x2,y2 in lines[:,0,:]:
            ang = math.degrees(math.atan2((y2 - y1), (x2 - x1)))
            if -max_deg <= ang <= max_deg:
                angles.append(ang)
    if angles:
        return float(np.median(angles))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    coords = np.column_stack((xs, ys))
    rect = cv2.minAreaRect(coords)
    ang = rect[-1]
    ang = -(90 + ang) if ang < -45 else -ang
    return float(max(-max_deg, min(max_deg, ang)))


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def crop_to_content(binary: np.ndarray, margin: int) -> Tuple[int,int,int,int]:
    h, w = binary.shape[:2]
    ys, xs = np.where(binary < 255)
    if len(xs) == 0:
        return 0,0,w,h
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    x0 = max(0, x0 - margin); y0 = max(0, y0 - margin)
    x1 = min(w, x1 + margin); y1 = min(h, y1 + margin)
    return int(x0), int(y0), int(x1), int(y1)


def render_page_rgb(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
    arr = np.frombuffer(pix.samples, dtype=np.uint8)
    return arr.reshape(pix.h, pix.w, pix.n)  # (H,W,3)


def downscale(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return img
    h, w = img.shape[:2]
    new_w = int(round(w / scale))
    new_h = int(round(h / scale))
    interp = cv2.INTER_AREA
    return cv2.resize(img, (new_w, new_h), interpolation=interp)


def apply_rl_deblur(gray: np.ndarray, psf_size: int, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return gray
    try:
        from skimage.restoration import richardson_lucy
    except Exception:
        # scikit-image non dispo : on ignore
        return gray
    psf_size = _make_odd(psf_size, 3)
    gk1 = cv2.getGaussianKernel(psf_size, psf_size/3)
    psf = (gk1 @ gk1.T).astype(np.float32)
    im = (gray.astype(np.float32) / 255.0).clip(0, 1)
    deconv = richardson_lucy(im, psf, iterations=iterations, clip=True)
    out = np.clip(deconv * 255.0, 0, 255).astype(np.uint8)
    return out


def insert_image_page(out_doc: fitz.Document, img: Image.Image, dpi: int, fmt: str, quality: int):
    w_px, h_px = img.size
    page_w = w_px * 72.0 / dpi
    page_h = h_px * 72.0 / dpi
    page = out_doc.new_page(width=page_w, height=page_h)

    buf = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() == "JPEG":
        save_kwargs = dict(format="JPEG", quality=quality, optimize=True, subsampling=1)
    else:
        save_kwargs = dict(format="PNG", optimize=True)
    img.save(buf, **save_kwargs)
    stream = buf.getvalue()

    rect = fitz.Rect(0, 0, page_w, page_h)
    page.insert_image(rect, stream=stream)


# --------------------- Core ---------------------

def process_page(rgb: np.ndarray, cfg: PreoptConfig) -> np.ndarray:
    g = _to_gray(rgb)

    if cfg.normalize_background:
        g = normalize_background(g)

    if cfg.denoise_h > 0:
        g = denoise(g, cfg.denoise_h)

    if cfg.clahe:
        g = clahe_contrast(g, cfg.clahe_clip, cfg.clahe_tile)

    g = apply_gamma(g, cfg.gamma)

    if cfg.unsharp > 0:
        g = unsharp_mask(g, cfg.unsharp, cfg.unsharp_radius)

    # Déconvolution optionnelle (sur gris)
    if cfg.deblur_rl_iter > 0:
        g = apply_rl_deblur(g, cfg.deblur_rl_psf, cfg.deblur_rl_iter)

    # Binarisation éventuelle
    if cfg.binarize != "none":
        bw = binarize(g, cfg.binarize, cfg.adaptive_block, cfg.adaptive_C)
        if cfg.remove_speckles_area > 0:
            bw = clean_speckles(bw, cfg.remove_speckles_area)

        if cfg.deskew:
            angle = estimate_skew(bw, cfg.max_skew_deg)
            if abs(angle) > 0.1:
                bw = rotate(bw, angle)
                g = rotate(g, angle)

        bw = morphology(bw, cfg.morph_open, cfg.morph_close)

        if cfg.crop:
            x0,y0,x1,y1 = crop_to_content(bw, cfg.crop_margin)
            bw = bw[y0:y1, x0:x1]
            g = g[y0:y1, x0:x1]

        # Post-sharpen (binaire : inutile) → on revient en gris si besoin
        if cfg.output_color == "binary":
            return bw
        else:
            if cfg.post_unsharp > 0:
                g = unsharp_mask(g, cfg.post_unsharp, cfg.post_unsharp_radius)
            return g
    else:
        # Pas de binarisation : deskew/crop avec seuillage rapide
        temp = binarize(g, "otsu", cfg.adaptive_block, cfg.adaptive_C)
        if cfg.deskew:
            angle = estimate_skew(temp, cfg.max_skew_deg)
            if abs(angle) > 0.1:
                g = rotate(g, angle)
                temp = rotate(temp, angle)
        if cfg.crop:
            x0,y0,x1,y1 = crop_to_content(temp, cfg.crop_margin)
            g = g[y0:y1, x0:x1]

        if cfg.post_unsharp > 0:
            g = unsharp_mask(g, cfg.post_unsharp, cfg.post_unsharp_radius)
        return g


def preoptimize_pdf(input_path: str, output_path: str, cfg: PreoptConfig, pages_expr: str = "", verbose: bool = False):
    in_doc = fitz.open(input_path)
    out_doc = fitz.open()
    total = len(in_doc)
    selection = parse_pages_selection(pages_expr, total)

    # DPI pour rendu (super-sample)
    dpi_render = int(round(cfg.dpi * max(1.0, cfg.super_sample)))

    if verbose:
        print(f"[INFO] Entrée : {input_path}")
        print(f"[INFO] Sortie : {output_path}")
        print(f"[INFO] Pages : {len(selection)}/{total}")
        print(f"[INFO] DPI target={cfg.dpi}, super-sample={cfg.super_sample} -> DPI rendu={dpi_render}")
        print(f"[INFO] Profil sortie={cfg.output_color}, RL={cfg.deblur_rl_iter} it, post_unsharp={cfg.post_unsharp}")

    try:
        for j, i in enumerate(selection, 1):
            page = in_doc.load_page(i)

            # Si page textuelle et conservation demandée → copie directe
            if cfg.keep_text_pages:
                try:
                    if page.get_text("text").strip():
                        out_doc.insert_pdf(in_doc, from_page=i, to_page=i)
                        if verbose: print(f"[PAGE {i+1}] texte natif -> copie")
                        continue
                except Exception:
                    pass

            # Rasterisation
            rgb_hi = render_page_rgb(page, dpi_render)

            # Traitement
            img = process_page(rgb_hi, cfg)

            # Downscale si super-sample > 1
            if cfg.super_sample > 1.0:
                scale = cfg.super_sample
                img = downscale(img, scale)

            # PIL image selon mode
            if img.ndim == 2:
                pil = Image.fromarray(img, mode="L")
            else:
                pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

            fmt = cfg.image_format if cfg.output_color != "binary" else "PNG"
            qual = cfg.jpeg_quality
            insert_image_page(out_doc, pil, cfg.dpi, fmt, qual)

            if verbose:
                print(f"[PAGE {i+1}] ok")
    finally:
        # Sauvegarde robuste
        try:
            out_doc.save(output_path, deflate=True)
        except Exception:
            out_doc.save(output_path, deflate=True, clean=True, garbage=3)
        out_doc.close()
        in_doc.close()


# --------------------- CLI ---------------------

def parse_pages_selection(pages: str, total: int) -> List[int]:
    if not pages:
        return list(range(total))
    selected = set()
    for part in [p.strip() for p in pages.split(",") if p.strip()]:
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a) if a else 1
            end = int(b) if b else total
            for p in range(start, end+1):
                if 1 <= p <= total:
                    selected.add(p-1)
        else:
            p = int(part)
            if 1 <= p <= total:
                selected.add(p-1)
    return sorted(selected)


def parse_args():
    p = argparse.ArgumentParser(description="Pré-optimisation PDF pour OCR (nettete+).")
    p.add_argument("--input")
    p.add_argument("--output")
    p.add_argument("--gui", action="store_true")
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--pages", type=str, default="")
    p.add_argument("--verbose", action="store_true")

    # pipeline de base
    p.add_argument("--keep-text-pages", dest="keep_text_pages", action="store_true", default=True)
    p.add_argument("--no-keep-text-pages", dest="keep_text_pages", action="store_false")
    p.add_argument("--normalize-background", dest="normalize_background", action="store_true", default=True)
    p.add_argument("--no-normalize-background", dest="normalize_background", action="store_false")
    p.add_argument("--denoise-h", type=int, default=10)
    p.add_argument("--clahe", action="store_true", default=True)
    p.add_argument("--no-clahe", dest="clahe", action="store_false")
    p.add_argument("--clahe-clip", type=float, default=2.0)
    p.add_argument("--clahe-tile", type=int, default=8)
    p.add_argument("--gamma", type=float, default=1.05)
    p.add_argument("--unsharp", type=float, default=0.8)
    p.add_argument("--unsharp-radius", type=float, default=1.0)
    p.add_argument("--binarize", choices=["none","adaptive","otsu"], default="none")
    p.add_argument("--adaptive-block", type=int, default=25)
    p.add_argument("--adaptive-C", type=int, default=10)
    p.add_argument("--remove-speckles-area", type=int, default=12)
    p.add_argument("--morph-open", type=int, default=2)
    p.add_argument("--morph-close", type=int, default=2)
    p.add_argument("--deskew", action="store_true", default=True)
    p.add_argument("--no-deskew", dest="deskew", action="store_false")
    p.add_argument("--max-skew-deg", type=float, default=10.0)
    p.add_argument("--crop", action="store_true", default=True)
    p.add_argument("--no-crop", dest="crop", action="store_false")
    p.add_argument("--crop-margin", type=int, default=5)
    p.add_argument("--output-color", choices=["gray","binary"], default="gray")
    p.add_argument("--jpeg-quality", type=int, default=85)

    # Netteté avancée
    p.add_argument("--super-sample", type=float, default=1.0, help="1.0=off, 1.5 ou 2.0 recommandé")
    p.add_argument("--post-unsharp", type=float, default=0.0, help="0.0=off, 0.3-0.8 conseillé")
    p.add_argument("--post-unsharp-radius", type=float, default=0.8)
    p.add_argument("--deblur-rl-iter", type=int, default=0, help="0=off, 5-15 utile")
    p.add_argument("--deblur-rl-psf", type=int, default=3, help="taille noyau RL (impair)")

    args = p.parse_args()

    # GUI si chemins manquants
    if args.gui or not args.input or not args.output:
        try:
            import tkinter as tk
            from tkinter import filedialog
            root = tk.Tk(); root.withdraw(); root.update_idletasks()
            in_path = args.input or filedialog.askopenfilename(title="Sélectionnez le PDF d'entrée", filetypes=[("PDF","*.pdf")])
            if not in_path:
                p.error("Aucun PDF d'entrée fourni.")
            base,_ = os.path.splitext(in_path)
            out_path = args.output or filedialog.asksaveasfilename(title="Enregistrer le PDF optimisé sous…",
                                                                   defaultextension=".pdf",
                                                                   initialfile=os.path.basename(base+"_sharp.pdf"),
                                                                   filetypes=[("PDF","*.pdf")])
            root.destroy()
        except Exception:
            in_path = args.input or input("Chemin du PDF d'entrée: ").strip().strip('"')
            out_path = args.output or input("Chemin du PDF de sortie: ").strip().strip('"')
    else:
        in_path, out_path = args.input, args.output

    if os.path.abspath(in_path) == os.path.abspath(out_path):
        base,_ = os.path.splitext(in_path)
        out_path = base + "_sharp.pdf"

    cfg = PreoptConfig(
        dpi=args.dpi,
        keep_text_pages=args.keep_text_pages,
        normalize_background=args.normalize_background,
        denoise_h=args.denoise_h,
        clahe=args.clahe,
        clahe_clip=args.clahe_clip,
        clahe_tile=args.clahe_tile,
        gamma=args.gamma,
        unsharp=args.unsharp,
        unsharp_radius=args.unsharp_radius,
        binarize=args.binarize,
        adaptive_block=args.adaptive_block,
        adaptive_C=args.adaptive_C,
        remove_speckles_area=args.remove_speckles_area,
        morph_open=args.morph_open,
        morph_close=args.morph_close,
        deskew=args.deskew,
        max_skew_deg=args.max_skew_deg,
        crop=args.crop,
        crop_margin=args.crop_margin,
        output_color=args.output_color,
        image_format=("JPEG" if args.output_color=="gray" else "PNG"),
        jpeg_quality=max(40, min(95, args.jpeg_quality)),
        super_sample=max(1.0, args.super_sample),
        post_unsharp=max(0.0, args.post_unsharp),
        post_unsharp_radius=max(0.1, args.post_unsharp_radius),
        deblur_rl_iter=max(0, args.deblur_rl_iter),
        deblur_rl_psf=max(3, args.deblur_rl_psf),
    )

    return in_path, out_path, cfg, args.pages, args.verbose


def parse_pages_selection(pages: str, total: int) -> List[int]:
    if not pages:
        return list(range(total))
    selected = set()
    for part in [p.strip() for p in pages.split(",") if p.strip()]:
        if "-" in part:
            a, b = part.split("-", 1)
            start = int(a) if a else 1
            end = int(b) if b else total
            for p in range(start, end+1):
                if 1 <= p <= total:
                    selected.add(p-1)
        else:
            p = int(part)
            if 1 <= p <= total:
                selected.add(p-1)
    return sorted(selected)


def main():
    in_path, out_path, cfg, pages_expr, verbose = parse_args()
    preoptimize_pdf(in_path, out_path, cfg, pages_expr=pages_expr, verbose=verbose)
    if verbose:
        print(f"✅ PDF optimisé écrit dans: {out_path}")


if __name__ == "__main__":
    main()
