#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
preopt_pdf_sharp.py — Pré-optimisation PDF pour OCR avec NETTETÉ MAXIMALE
Optimisé spécifiquement pour factures et documents commerciaux.
"""

import argparse
import io
import os
import math
from dataclasses import dataclass
from typing import Tuple, List

import fitz  # PyMuPDF
import numpy as np
import cv2
from PIL import Image


@dataclass
class PreoptConfig:
    dpi: int = 300
    keep_text_pages: bool = True
    normalize_background: bool = True
    denoise_h: int = 7
    clahe: bool = True
    clahe_clip: float = 2.5
    clahe_tile: int = 8
    gamma: float = 1.1
    bilateral: bool = True
    bilateral_d: int = 9
    bilateral_sigma_color: int = 75
    bilateral_sigma_space: int = 75
    unsharp: float = 1.2
    unsharp_radius: float = 1.5
    high_pass: bool = True
    high_pass_radius: int = 5
    binarize: str = "none"
    adaptive_block: int = 31
    adaptive_C: int = 11
    remove_speckles_area: int = 15
    morph_open: int = 1
    morph_close: int = 1
    deskew: bool = True
    max_skew_deg: float = 10.0
    crop: bool = True
    crop_margin: int = 10
    output_color: str = "gray"
    image_format: str = "PNG"
    jpeg_quality: int = 95
    super_sample: float = 1.5
    post_unsharp: float = 0.7
    post_unsharp_radius: float = 1.2
    edge_enhance: bool = True
    contrast_stretch: bool = True


def _make_odd(n: int, minimum: int = 3) -> int:
    n = max(n, minimum)
    return n if n % 2 == 1 else n + 1


def _to_gray(rgb: np.ndarray) -> np.ndarray:
    if len(rgb.shape) == 2:
        return rgb
    if rgb.shape[2] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def normalize_background(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    kernel_size = max(31, min(101, int(max(h, w) * 0.03)))
    kernel_size = _make_odd(kernel_size)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel, iterations=1)
    background = cv2.GaussianBlur(background, (kernel_size, kernel_size), 0)
    
    gray_float = gray.astype(np.float32)
    bg_float = background.astype(np.float32) + 1.0
    normalized = cv2.divide(gray_float, bg_float, scale=255.0)
    normalized = cv2.normalize(normalized, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized.astype(np.uint8)


def denoise_advanced(gray: np.ndarray, h: int) -> np.ndarray:
    if h <= 0:
        return gray
    h_actual = max(3, min(15, h))
    return cv2.fastNlMeansDenoising(gray, None, h=h_actual, templateWindowSize=7, searchWindowSize=21)


def bilateral_filter(gray: np.ndarray, d: int, sigma_color: int, sigma_space: int) -> np.ndarray:
    d = min(9, max(5, d))
    return cv2.bilateralFilter(gray, d, sigma_color, sigma_space)


def clahe_contrast(gray: np.ndarray, clip: float, tile: int) -> np.ndarray:
    h, w = gray.shape
    tile_size = max(8, min(tile, min(w, h) // 8))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile_size, tile_size))
    return clahe.apply(gray)


def apply_gamma(gray: np.ndarray, gamma: float) -> np.ndarray:
    if abs(gamma - 1.0) < 0.01:
        return gray
    inv_gamma = 1.0 / gamma
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray, lut)


def contrast_stretch(gray: np.ndarray, lower_percentile: float = 1.0, upper_percentile: float = 99.0) -> np.ndarray:
    p_low = np.percentile(gray, lower_percentile)
    p_high = np.percentile(gray, upper_percentile)
    if p_high - p_low < 1:
        return gray
    stretched = (gray.astype(np.float32) - p_low) * (255.0 / (p_high - p_low))
    return np.clip(stretched, 0, 255).astype(np.uint8)


def high_pass_filter(gray: np.ndarray, radius: int) -> np.ndarray:
    if radius < 1:
        return gray
    radius = _make_odd(radius)
    low_pass = cv2.GaussianBlur(gray, (radius, radius), 0)
    gray_float = gray.astype(np.float32)
    low_pass_float = low_pass.astype(np.float32)
    high_pass = gray_float - low_pass_float
    enhanced = gray_float + 0.5 * high_pass
    return np.clip(enhanced, 0, 255).astype(np.uint8)


def unsharp_mask(gray: np.ndarray, amount: float, radius: float) -> np.ndarray:
    if amount <= 0:
        return gray
    sigma = max(0.5, radius)
    blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma, sigmaY=sigma)
    gray_float = gray.astype(np.float32)
    blurred_float = blurred.astype(np.float32)
    sharpened = gray_float + amount * (gray_float - blurred_float)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def edge_enhance(gray: np.ndarray) -> np.ndarray:
    laplacian = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    laplacian = np.absolute(laplacian)
    laplacian = np.uint8(np.clip(laplacian, 0, 255))
    enhanced = cv2.addWeighted(gray, 1.0, laplacian, 0.3, 0)
    return enhanced


def binarize(gray: np.ndarray, method: str, block: int, C: int) -> np.ndarray:
    if method == "none":
        return gray
    if method == "otsu":
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary
    block = _make_odd(block, 11)
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, C)


def clean_speckles(binary: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return binary
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    output = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            output[labels == i] = 255
    return cv2.bitwise_not(output)


def morphology(binary: np.ndarray, open_k: int, close_k: int) -> np.ndarray:
    out = binary.copy()
    if open_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
    if close_k > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return out


def estimate_skew(binary: np.ndarray, max_deg: float) -> float:
    h, w = binary.shape
    if max(h, w) > 2000:
        scale = 2000.0 / max(h, w)
        small = cv2.resize(binary, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = binary
    edges = cv2.Canny(small, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=80, 
                            minLineLength=small.shape[1] // 4, maxLineGap=20)
    if lines is None or len(lines) < 5:
        return 0.0
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        if angle < -45:
            angle += 90
        elif angle > 45:
            angle -= 90
        if abs(angle) <= max_deg:
            angles.append(angle)
    if len(angles) < 5:
        return 0.0
    return float(np.median(angles))


def rotate_image(img: np.ndarray, angle: float, border_value: int = 255) -> np.ndarray:
    if abs(angle) < 0.05:
        return img
    h, w = img.shape[:2]
    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_CUBIC, 
                             borderMode=cv2.BORDER_CONSTANT, borderValue=border_value)
    return rotated


def crop_to_content(binary: np.ndarray, margin: int) -> Tuple[int, int, int, int]:
    h, w = binary.shape
    coords = np.column_stack(np.where(binary < 250))
    if len(coords) == 0:
        return 0, 0, w, h
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    x0 = max(0, x_min - margin)
    y0 = max(0, y_min - margin)
    x1 = min(w, x_max + margin + 1)
    y1 = min(h, y_max + margin + 1)
    return int(x0), int(y0), int(x1), int(y1)


def render_page_rgb(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False, colorspace=fitz.csRGB, annots=True)
    img = np.frombuffer(pix.samples, dtype=np.uint8)
    img = img.reshape(pix.h, pix.w, 3)
    return img


def downscale_quality(img: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return img
    h, w = img.shape[:2]
    new_w = int(round(w / scale))
    new_h = int(round(h / scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def insert_image_to_pdf(out_doc: fitz.Document, img: Image.Image, dpi: int, fmt: str, quality: int):
    w_px, h_px = img.size
    page_w = w_px * 72.0 / dpi
    page_h = h_px * 72.0 / dpi
    page = out_doc.new_page(width=page_w, height=page_h)
    buf = io.BytesIO()
    if fmt.upper() == "JPEG":
        img.save(buf, format="JPEG", quality=quality, optimize=True, subsampling=0)
    elif fmt.upper() == "PNG":
        img.save(buf, format="PNG", optimize=True, compress_level=9)
    else:
        img.save(buf, format=fmt)
    stream = buf.getvalue()
    rect = fitz.Rect(0, 0, page_w, page_h)
    page.insert_image(rect, stream=stream)


def process_page(rgb: np.ndarray, cfg: PreoptConfig) -> np.ndarray:
    gray = _to_gray(rgb)
    
    if cfg.normalize_background:
        gray = normalize_background(gray)
    
    if cfg.denoise_h > 0:
        gray = denoise_advanced(gray, cfg.denoise_h)
    
    if cfg.bilateral:
        gray = bilateral_filter(gray, cfg.bilateral_d, cfg.bilateral_sigma_color, cfg.bilateral_sigma_space)
    
    if cfg.clahe:
        gray = clahe_contrast(gray, cfg.clahe_clip, cfg.clahe_tile)
    
    gray = apply_gamma(gray, cfg.gamma)
    
    if cfg.contrast_stretch:
        gray = contrast_stretch(gray, lower_percentile=1, upper_percentile=99)
    
    if cfg.unsharp > 0:
        gray = unsharp_mask(gray, cfg.unsharp, cfg.unsharp_radius)
    
    if cfg.high_pass:
        gray = high_pass_filter(gray, cfg.high_pass_radius)
    
    if cfg.edge_enhance:
        gray = edge_enhance(gray)
    
    if cfg.binarize != "none" or cfg.deskew or cfg.crop:
        temp_binary = binarize(gray, "otsu", cfg.adaptive_block, cfg.adaptive_C)
        
        if cfg.deskew:
            angle = estimate_skew(temp_binary, cfg.max_skew_deg)
            if abs(angle) > 0.1:
                gray = rotate_image(gray, angle, border_value=255)
                temp_binary = rotate_image(temp_binary, angle, border_value=255)
        
        if cfg.crop:
            x0, y0, x1, y1 = crop_to_content(temp_binary, cfg.crop_margin)
            gray = gray[y0:y1, x0:x1]
            temp_binary = temp_binary[y0:y1, x0:x1]
        
        if cfg.binarize != "none":
            binary = binarize(gray, cfg.binarize, cfg.adaptive_block, cfg.adaptive_C)
            if cfg.remove_speckles_area > 0:
                binary = clean_speckles(binary, cfg.remove_speckles_area)
            binary = morphology(binary, cfg.morph_open, cfg.morph_close)
            if cfg.output_color == "binary":
                final = binary
            else:
                final = gray
        else:
            final = gray
    else:
        final = gray
    
    if cfg.post_unsharp > 0 and cfg.output_color != "binary":
        final = unsharp_mask(final, cfg.post_unsharp, cfg.post_unsharp_radius)
    
    return final


def preoptimize_pdf(input_path: str, output_path: str, cfg: PreoptConfig, pages_expr: str = "", verbose: bool = False):
    in_doc = fitz.open(input_path)
    out_doc = fitz.open()
    total = len(in_doc)
    selection = parse_pages_selection(pages_expr, total)
    dpi_render = int(round(cfg.dpi * cfg.super_sample))
    
    if verbose:
        print(f"╔═══════════════════════════════════════════════════════╗")
        print(f"║  PRÉ-OPTIMISATION PDF POUR OCR - MODE NETTETÉ MAX    ║")
        print(f"╚═══════════════════════════════════════════════════════╝")
        print(f"📄 Entrée       : {input_path}")
        print(f"💾 Sortie       : {output_path}")
        print(f"📊 Pages        : {len(selection)}/{total}")
        print(f"🎯 DPI cible    : {cfg.dpi}")
        print(f"🔍 Super-sample : x{cfg.super_sample:.1f} → DPI rendu={dpi_render}")
        print(f"🖼️  Format       : {cfg.output_color.upper()} ({cfg.image_format})")
        print(f"✨ Netteté      : unsharp={cfg.unsharp:.1f}, post={cfg.post_unsharp:.1f}")
        print()
    
    try:
        for j, page_idx in enumerate(selection, 1):
            page = in_doc.load_page(page_idx)
            
            if cfg.keep_text_pages:
                text = page.get_text("text").strip()
                if text and len(text) > 100:
                    out_doc.insert_pdf(in_doc, from_page=page_idx, to_page=page_idx)
                    if verbose:
                        print(f"📄 Page {page_idx+1:3d}/{total} → Texte natif conservé")
                    continue
            
            if verbose:
                print(f"🔄 Page {page_idx+1:3d}/{total} → Traitement...", end=" ", flush=True)
            
            rgb = render_page_rgb(page, dpi_render)
            processed = process_page(rgb, cfg)
            
            if cfg.super_sample > 1.0:
                processed = downscale_quality(processed, cfg.super_sample)
            
            if processed.ndim == 2:
                pil_img = Image.fromarray(processed, mode="L")
            else:
                pil_img = Image.fromarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB))
            
            fmt = "PNG" if cfg.output_color == "binary" else cfg.image_format
            insert_image_to_pdf(out_doc, pil_img, cfg.dpi, fmt, cfg.jpeg_quality)
            
            if verbose:
                print("✅")
    
    finally:
        if verbose:
            print(f"\n💾 Sauvegarde du PDF optimisé...", end=" ", flush=True)
        
        try:
            out_doc.save(output_path, garbage=4, clean=True, deflate=True, deflate_images=True, deflate_fonts=True)
        except Exception as e:
            if verbose:
                print(f"\n⚠️  Mode basique...", end=" ")
            out_doc.save(output_path)
        
        if verbose:
            print("✅")
        
        out_doc.close()
        in_doc.close()


def parse_pages_selection(pages: str, total: int) -> List[int]:
    if not pages:
        return list(range(total))
    selected = set()
    for part in pages.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start = int(start_str.strip()) if start_str.strip() else 1
            end = int(end_str.strip()) if end_str.strip() else total
            for p in range(start, end + 1):
                if 1 <= p <= total:
                    selected.add(p - 1)
        else:
            p = int(part)
            if 1 <= p <= total:
                selected.add(p - 1)
    return sorted(selected)


def parse_args():
    p = argparse.ArgumentParser(description="🔍 Pré-optimisation PDF pour OCR - NETTETÉ MAXIMALE")
    
    p.add_argument("--input", "-i", help="📄 PDF d'entrée")
    p.add_argument("--output", "-o", help="💾 PDF de sortie")
    p.add_argument("--gui", "-g", action="store_true", help="🖱️  Mode interface graphique")
    p.add_argument("--verbose", "-v", action="store_true", help="📊 Mode verbeux")
    p.add_argument("--dpi", type=int, default=300, help="🎯 Résolution cible (défaut: 300)")
    p.add_argument("--pages", type=str, default="", help="📑 Pages à traiter")
    p.add_argument("--super-sample", type=float, default=1.5, help="🔍 Sur-échantillonnage (défaut: 1.5)")
    p.add_argument("--keep-text-pages", dest="keep_text_pages", action="store_true", default=True)
    p.add_argument("--no-keep-text-pages", dest="keep_text_pages", action="store_false")
    p.add_argument("--normalize-background", dest="normalize_background", action="store_true", default=True)
    p.add_argument("--no-normalize-background", dest="normalize_background", action="store_false")
    p.add_argument("--clahe", action="store_true", default=True)
    p.add_argument("--no-clahe", dest="clahe", action="store_false")
    p.add_argument("--bilateral", action="store_true", default=True)
    p.add_argument("--no-bilateral", dest="bilateral", action="store_false")
    p.add_argument("--high-pass", dest="high_pass", action="store_true", default=True)
    p.add_argument("--no-high-pass", dest="high_pass", action="store_false")
    p.add_argument("--edge-enhance", dest="edge_enhance", action="store_true", default=True)
    p.add_argument("--no-edge-enhance", dest="edge_enhance", action="store_false")
    p.add_argument("--contrast-stretch", dest="contrast_stretch", action="store_true", default=True)
    p.add_argument("--no-contrast-stretch", dest="contrast_stretch", action="store_false")
    p.add_argument("--deskew", action="store_true", default=True)
    p.add_argument("--no-deskew", dest="deskew", action="store_false")
    p.add_argument("--crop", action="store_true", default=True)
    p.add_argument("--no-crop", dest="crop", action="store_false")
    p.add_argument("--denoise-h", type=int, default=7)
    p.add_argument("--clahe-clip", type=float, default=2.5)
    p.add_argument("--gamma", type=float, default=1.1)
    p.add_argument("--unsharp", type=float, default=1.2)
    p.add_argument("--unsharp-radius", type=float, default=1.5)
    p.add_argument("--post-unsharp", type=float, default=0.7)
    p.add_argument("--post-unsharp-radius", type=float, default=1.2)
    p.add_argument("--binarize", choices=["none", "adaptive", "otsu"], default="none")
    p.add_argument("--adaptive-block", type=int, default=31)
    p.add_argument("--adaptive-C", type=int, default=11)
    p.add_argument("--output-color", choices=["gray", "binary"], default="gray")
    p.add_argument("--image-format", choices=["JPEG", "PNG"], default="PNG")
    p.add_argument("--jpeg-quality", type=int, default=95)
    
    args = p.parse_args()
    
    if args.gui or not args.input or not args.output:
        try:
            import tkinter as tk
            from tkinter import filedialog, messagebox
            
            root = tk.Tk()
            root.withdraw()
            
            if not args.input:
                args.input = filedialog.askopenfilename(title="📄 Sélectionnez le PDF à optimiser", filetypes=[("PDF", "*.pdf"), ("Tous fichiers", "*.*")])
            
            if not args.input:
                messagebox.showerror("Erreur", "❌ Aucun fichier sélectionné")
                return None
            
            if not args.output:
                base, _ = os.path.splitext(args.input)
                default_name = os.path.basename(base) + "_SHARP.pdf"
                args.output = filedialog.asksaveasfilename(title="💾 Enregistrer le PDF optimisé", defaultextension=".pdf", initialfile=default_name, filetypes=[("PDF", "*.pdf")])
            
            if not args.output:
                messagebox.showerror("Erreur", "❌ Aucun fichier de sortie")
                return None
            
            root.destroy()
            
        except ImportError:
            if not args.input:
                args.input = input("📄 Chemin PDF d'entrée : ").strip().strip('"\'')
            if not args.output:
                args.output = input("💾 Chemin PDF de sortie : ").strip().strip('"\'')
    
    if not os.path.exists(args.input):
        p.error(f"❌ Fichier introuvable : {args.input}")
    
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_SHARP{ext}"
        if args.verbose:
            print(f"⚠️  Sortie renommée : {args.output}")
    
    cfg = PreoptConfig(
        dpi=max(150, min(600, args.dpi)),
        keep_text_pages=args.keep_text_pages,
        normalize_background=args.normalize_background,
        denoise_h=max(0, min(15, args.denoise_h)),
        clahe=args.clahe,
        clahe_clip=max(1.0, min(4.0, args.clahe_clip)),
        clahe_tile=8,
        gamma=max(0.5, min(2.0, args.gamma)),
        bilateral=args.bilateral,
        bilateral_d=9,
        bilateral_sigma_color=75,
        bilateral_sigma_space=75,
        unsharp=max(0.0, min(3.0, args.unsharp)),
        unsharp_radius=max(0.3, min(5.0, args.unsharp_radius)),
        high_pass=args.high_pass,
        high_pass_radius=5,
        binarize=args.binarize,
        adaptive_block=max(11, args.adaptive_block),
        adaptive_C=args.adaptive_C,
        remove_speckles_area=15,
        morph_open=1,
        morph_close=1,
        deskew=args.deskew,
        max_skew_deg=10.0,
        crop=args.crop,
        crop_margin=10,
        output_color=args.output_color,
        image_format=args.image_format,
        jpeg_quality=max(40, min(100, args.jpeg_quality)),
        super_sample=max(1.0, min(3.0, args.super_sample)),
        post_unsharp=max(0.0, min(3.0, args.post_unsharp)),
        post_unsharp_radius=max(0.3, min(5.0, args.post_unsharp_radius)),
        edge_enhance=args.edge_enhance,
        contrast_stretch=args.contrast_stretch
    )
    
    return args.input, args.output, cfg, args.pages, args.verbose


def main():
    result = parse_args()
    if result is None:
        return 1
    
    input_path, output_path, cfg, pages_expr, verbose = result
    
    try:
        preoptimize_pdf(input_path, output_path, cfg, pages_expr, verbose)
        
        if verbose:
            print()
            print("╔═══════════════════════════════════════════════════════╗")
            print("║            ✅ TRAITEMENT TERMINÉ AVEC SUCCÈS          ║")
            print("╚═══════════════════════════════════════════════════════╝")
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"📁 Fichier    : {output_path}")
            print(f"📊 Taille     : {file_size:.2f} MB")
            print()
        else:
            print(f"✅ PDF optimisé créé : {output_path}")
        
        return 0
    
    except KeyboardInterrupt:
        print("\n⚠️  Traitement interrompu par l'utilisateur")
        return 130
    
    except Exception as e:
        print(f"\n❌ ERREUR : {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
