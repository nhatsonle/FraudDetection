"""
Real-world Scan Test Script for IMG_2839.jpeg and IMG_2840.jpeg
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageOps
import pandas as pd

# Set environment flags to prevent OneDNN C++ bug in Paddle
os.environ['FLAGS_use_onednn'] = '0'
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'

from utils.preprocessor import load_image
from utils.ocr_engine import BankGuardOCREngine
from utils.matcher import AntiFraudMatcher


def test_real_scan(img_path):
    print(f"\n=======================================================")
    print(f"Testing Real-World Scan: {img_path}")
    print(f"=======================================================")
    
    # 1. Load image with EXIF transpose
    pil_img = Image.open(img_path)
    pil_img = ImageOps.exif_transpose(pil_img)
    bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    print(f"Original Transposed Image Size: {bgr_img.shape[1]}x{bgr_img.shape[0]}")

    # Initialize OCR Engine
    engine = BankGuardOCREngine(use_gpu=False)

    # Test 4 rotations (0, 90, 180, 270) to auto-detect upright table orientation
    rotations = [
        (0, bgr_img),
        (90, cv2.rotate(bgr_img, cv2.ROTATE_90_CLOCKWISE)),
        (180, cv2.rotate(bgr_img, cv2.ROTATE_180)),
        (270, cv2.rotate(bgr_img, cv2.ROTATE_90_COUNTERCLOCKWISE)),
    ]

    best_angle = 0
    best_img = bgr_img
    best_items = []
    max_headers = -1

    for angle, rot_img in rotations:
        items = engine.extract_raw_ocr(rot_img)
        headers = [it['text'] for it in items if any(h in it['text'].lower() for h in ['stk', 'tên', 'ngân hàng', 'số tiền', 'amount', 'bank'])]
        print(f"Angle {angle}°: Found {len(headers)} header keywords out of {len(items)} text boxes.")
        if len(headers) > max_headers:
            max_headers = len(headers)
            best_angle = angle
            best_img = rot_img
            best_items = items

    print(f"\n---> Auto-Detected Upright Angle: {best_angle}° ({max_headers} headers found)")

    # Extract Structured Table
    df_paper, cell_bboxes = engine.extract_structured_table(best_img, raw_ocr_items=best_items)
    print(f"\nExtracted {len(df_paper)} rows from Paper Scan:")
    print(df_paper.to_string())

    # Reconcile against CSV
    df_csv = pd.read_csv("demo_data/bank_transactions.csv")
    matcher = AntiFraudMatcher()
    df_rec, summary = matcher.reconcile(df_csv, df_paper)

    print(f"\nReconciliation Summary: {summary}")
    print("\nReconciled Results:")
    print(df_rec[['STK_CSV', 'Tên_CSV', 'STK_Paper', 'Tên_Paper', 'Fraud_Status', 'Fraud_Detail']].to_string())


if __name__ == "__main__":
    test_real_scan("demo_data/IMG_2839.jpeg")
    test_real_scan("demo_data/IMG_2840.jpeg")
