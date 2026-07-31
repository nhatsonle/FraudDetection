"""
BankGuard Preprocessor Module
------------------------------
Handles Image Pre-processing, EXIF Transpose, Auto 4-Way Orientation Detection,
Deskewing, Contrast Enhancement, and Error Level Analysis (ELA).
"""

import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance, ImageDraw, ImageFont, ImageOps
import io
import unicodedata
from typing import Tuple, Union


def load_image(image_input: Union[str, bytes, Image.Image, np.ndarray]) -> np.ndarray:
    """
    Converts input image to a standard OpenCV BGR numpy array,
    automatically applying EXIF orientation transpose for smartphone/scan photos.
    """
    if isinstance(image_input, str):
        pil_img = Image.open(image_input)
        pil_img = ImageOps.exif_transpose(pil_img)
        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, bytes):
        pil_img = Image.open(io.BytesIO(image_input))
        pil_img = ImageOps.exif_transpose(pil_img)
        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, Image.Image):
        pil_img = ImageOps.exif_transpose(image_input)
        return cv2.cvtColor(np.array(pil_img.convert('RGB')), cv2.COLOR_RGB2BGR)
    elif isinstance(image_input, np.ndarray):
        if len(image_input.shape) == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
        return image_input.copy()
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")


def auto_orient_upright(image: np.ndarray, ocr_engine) -> Tuple[np.ndarray, int]:
    """
    Detects if the image is rotated 90, 180, or 270 degrees and rotates it upright.
    Returns: (upright_image, detected_angle)
    """
    rotations = [
        (0, image),
        (90, cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(image, cv2.ROTATE_180)),
        (270, cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    best_angle = 0
    best_img = image
    max_headers = -1

    for angle, rot_img in rotations:
        raw_ocr = ocr_engine.extract_raw_ocr(rot_img)
        header_count = sum(1 for it in raw_ocr if any(h in it['text'].lower() for h in ['stk', 'tên', 'ngân hàng', 'số tiền', 'amount', 'bank']))
        if header_count > max_headers:
            max_headers = header_count
            best_angle = angle
            best_img = rot_img

    return best_img, best_angle


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
        orig_pil = ImageOps.exif_transpose(Image.open(image_input)).convert('RGB')
    elif isinstance(image_input, bytes):
        orig_pil = ImageOps.exif_transpose(Image.open(io.BytesIO(image_input))).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        orig_pil = Image.fromarray(rgb)
    elif isinstance(image_input, Image.Image):
        orig_pil = ImageOps.exif_transpose(image_input).convert('RGB')
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


def draw_bounding_boxes_on_image(image: np.ndarray, boxes_with_labels: list) -> np.ndarray:
    """Renders Bounding Boxes with full Unicode Vietnamese text support using PIL."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_image)
    draw = ImageDraw.Draw(pil_img)

    font_size = max(14, int(image.shape[1] / 70))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()

    for item in boxes_with_labels:
        x, y, w, h = item['bbox']
        is_fraud = item.get('is_fraud', False)
        color = (239, 68, 68) if is_fraud else (16, 185, 129)
        outline_width = max(2, int(image.shape[1] / 300))
        
        draw.rectangle([x, y, x + w, y + h], outline=color, width=outline_width)
        
        raw_label = item.get('label', '')
        if raw_label:
            bbox = draw.textbbox((0, 0), raw_label, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
            
            badge_y1 = max(0, y - text_h - 8)
            badge_y2 = max(text_h + 8, y)
            
            draw.rectangle([x, badge_y1, x + text_w + 10, badge_y2], fill=color)
            draw.text((x + 5, badge_y1 + 2), raw_label, fill=(255, 255, 255), font=font)
            
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
