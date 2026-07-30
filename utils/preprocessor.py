"""
BankGuard Preprocessor Module
------------------------------
Handles Image Pre-processing (Deskewing, Contrast Enhancement, Denoising)
and Document Image Forensics (Error Level Analysis - ELA for Tampering Detection).
"""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFont
import io
import unicodedata
from typing import Tuple, Union


def load_image(image_input: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
    """Converts various input image types to a standard OpenCV BGR numpy array."""
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise ValueError(f"Could not read image from path: {image_input}")
        return img
    elif isinstance(image_input, bytes):
        nparr = np.frombuffer(image_input, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    elif isinstance(image_input, Image.Image):
        return cv2.cvtColor(np.array(image_input), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Automatically detects image skew angle and rotates the image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 10:
        return image
        
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    if abs(angle) < 0.5 or abs(angle) > 45:
        return image

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def preprocess_for_ocr(image: np.ndarray) -> np.ndarray:
    """Enhances contrast and applies adaptive thresholding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
    return cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)


def detect_ela(image_input: Union[str, bytes, Image.Image, np.ndarray], quality: int = 90) -> Tuple[Image.Image, Image.Image, float, bool]:
    """Performs Error Level Analysis (ELA) to detect image tampering."""
    if isinstance(image_input, str):
        orig_pil = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, bytes):
        orig_pil = Image.open(io.BytesIO(image_input)).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        orig_pil = Image.fromarray(rgb)
    elif isinstance(image_input, Image.Image):
        orig_pil = image_input.convert('RGB')
    else:
        raise TypeError("Invalid image type for ELA")

    buffer = io.BytesIO()
    orig_pil.save(buffer, 'JPEG', quality=quality)
    buffer.seek(0)
    resaved_pil = Image.open(buffer)

    ela_diff = ImageChops.difference(orig_pil, resaved_pil)
    extrema = ela_diff.getextrema()
    max_diff = max([ex[1] for ex in extrema]) or 1
    scale_factor = 255.0 / max_diff
    ela_enhanced = ImageEnhance.Brightness(ela_diff).enhance(scale_factor)

    ela_np = np.array(ela_enhanced)
    gray_ela = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)

    heatmap_np = cv2.applyColorMap(gray_ela, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_np, cv2.COLOR_BGR2RGB)
    heatmap_pil = Image.fromarray(heatmap_rgb)

    std_val = float(np.std(gray_ela))
    max_val = float(np.max(gray_ela))
    high_energy_ratio = float(np.sum(gray_ela > 120) / gray_ela.size) * 100.0
    
    tamper_score = min(100.0, round((std_val * 1.5) + (max_val * 0.2) + (high_energy_ratio * 12.0), 2))
    is_tampered = tamper_score > 30.0

    return orig_pil, heatmap_pil, tamper_score, is_tampered


def _to_ascii_label(text: str) -> str:
    """Converts Vietnamese unicode text to clean ASCII so OpenCV putText doesn't display '??'."""
    if not text:
        return ""
    nfd_form = unicodedata.normalize('NFD', text)
    no_accents = ''.join([c for c in nfd_form if unicodedata.category(c) != 'MN'])
    return no_accents.replace('đ', 'd').replace('Đ', 'D').strip().upper()


def draw_bounding_boxes_on_image(image: np.ndarray, boxes_with_labels: list) -> np.ndarray:
    """Draws colored bounding boxes with clean ASCII labels over an image."""
    annotated = image.copy()
    for item in boxes_with_labels:
        x, y, w, h = item['bbox']
        color = item.get('color', (0, 0, 255) if item.get('is_fraud', False) else (0, 255, 0))
        thickness = 3 if item.get('is_fraud', False) else 1
        
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, thickness)
        
        raw_label = item.get('label', '')
        if raw_label:
            # Convert label to clean ASCII to avoid OpenCV ?? encoding bug
            clean_label = _to_ascii_label(raw_label)
            font_scale = 0.5
            font_thick = 1
            (text_w, text_h), baseline = cv2.getTextSize(clean_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
            
            cv2.rectangle(
                annotated, 
                (x, max(0, y - text_h - 6)), 
                (x + text_w + 4, max(text_h + 6, y)), 
                color, 
                -1
            )
            cv2.putText(
                annotated, 
                clean_label, 
                (x + 2, max(text_h, y - 4)), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                font_scale, 
                (255, 255, 255), 
                font_thick, 
                cv2.LINE_AA
            )
            
    return annotated
