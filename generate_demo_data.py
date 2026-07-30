"""
BankGuard Demo Data Generator
-----------------------------
Generates synthetic CSV files and paper scan document images for testing all 6 fraud cases:
1. Case 1 (Matched): 100% match
2. Case 2 (Amount Fraud): 50,000,000 CSV vs 500,000,000 paper
3. Case 3 (Account Fraud): Altered STK on paper
4. Case 4 (Name Fraud): Altered beneficiary name on paper
5. Case 5 (Extra Row): Ghost transaction on paper (not in CSV)
6. Case 6 (Tampered Image): Photoshop/compression artifact overlay (ELA detectable)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io


def create_demo_dataset(output_dir: str = "demo_data"):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Baseline Original CSV File
    # -------------------------------------------------------------
    csv_data = [
        {"Tên": "NGUYEN VAN AN", "STK": "1012345678", "Số tiền": 50000000, "Ngân hàng": "Vietcombank"},
        {"Tên": "TRAN THI BINH", "STK": "9988776655", "Số tiền": 50000000, "Ngân hàng": "Techcombank"},
        {"Tên": "LE HOANG CUONG", "STK": "0071000123", "Số tiền": 35000000, "Ngân hàng": "BIDV"},
        {"Tên": "PHAM MINH DUC", "STK": "2200155999", "Số tiền": 85000000, "Ngân hàng": "Agribank"},
    ]
    df_csv = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, "bank_transactions.csv")
    df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Generated original CSV dataset: {csv_path}")

    # -------------------------------------------------------------
    # 2. Paper Scan Table Data (Containing All Test Cases)
    # -------------------------------------------------------------
    paper_rows = [
        # Case 1: Matched 100%
        {"Tên": "NGUYEN VAN AN", "STK": "1012345678", "Số tiền": "50,000,000 VND", "Ngân hàng": "VCB"},
        # Case 2 & 6: Amount Fraud (500,000,000 vs 50,000,000) & Tampered Photoshop area
        {"Tên": "TRAN THI BINH", "STK": "9988776655", "Số tiền": "500,000,000 VND", "Ngân hàng": "TCB"},
        # Case 3: Account Fraud (Altered STK: 0071999999 vs 0071000123)
        {"Tên": "LE HOANG CUONG", "STK": "0071999999", "Số tiền": "35,000,000 VND", "Ngân hàng": "BIDV"},
        # Case 4: Name Fraud (Altered Name: HOANG VAN PHUC vs PHAM MINH DUC)
        {"Tên": "HOANG VAN PHUC", "STK": "2200155999", "Số tiền": "85,000,000 VND", "Ngân hàng": "AGRIBANK"},
        # Case 5: Extra Row (Ghost row not in CSV)
        {"Tên": "VO VAN GHOST", "STK": "8888999900", "Số tiền": "250,000,000 VND", "Ngân hàng": "MBBANK"},
    ]

    # Generate Image Document for Full Bank Statement
    full_image_path = os.path.join(output_dir, "bank_statement_demo.png")
    generate_document_image(paper_rows, full_image_path, apply_tamper=True)
    print(f"Generated paper scan document image: {full_image_path}")

    # Generate Individual Case Images for quick testing
    for i, row in enumerate(paper_rows, start=1):
        case_img_path = os.path.join(output_dir, f"test_case_{i}.png")
        generate_document_image([row], case_img_path, apply_tamper=(i == 2))
        print(f"Generated single test case image: {case_img_path}")


def generate_document_image(rows_data: list, output_path: str, apply_tamper: bool = False):
    """
    Renders a high-resolution synthetic bank statement image with headers, gridlines,
    and text, optionally overlaying a compression tampering artifact for ELA testing.
    """
    width, height = 1000, 200 + len(rows_data) * 90
    img = Image.new('RGB', (width, height), color=(250, 252, 255))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 15)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except IOError:
        title_font = header_font = text_font = ImageFont.load_default()

    # Draw Header Section
    draw.rectangle([0, 0, width, 80], fill=(22, 54, 93))
    draw.text((30, 25), "NGÂN HÀNG THƯƠNG MẠI CỔ PHẦN - BẢNG KÊ GIAO DỊCH DỰ THƯỞNG", fill=(255, 255, 255), font=title_font)

    # Column coordinates
    col_x = [30, 300, 520, 750, 970]
    headers = ["HỌ VÀ TÊN THỤ HƯỞNG", "SỐ TÀI KHOẢN (STK)", "SỐ TIỀN GIAO DỊCH", "NGÂN HÀNG"]

    # Draw Table Header Row
    y_start = 100
    draw.rectangle([col_x[0] - 10, y_start, col_x[-1], y_start + 40], fill=(220, 230, 242), outline=(150, 170, 200))
    for idx, header in enumerate(headers):
        draw.text((col_x[idx], y_start + 10), header, fill=(20, 30, 50), font=header_font)

    # Draw Data Rows
    y = y_start + 40
    tamper_bbox = None

    for i, row in enumerate(rows_data):
        fill_color = (255, 255, 255) if i % 2 == 0 else (245, 248, 252)
        draw.rectangle([col_x[0] - 10, y, col_x[-1], y + 60], fill=fill_color, outline=(200, 210, 225))

        draw.text((col_x[0], y + 20), str(row["Tên"]), fill=(10, 10, 10), font=text_font)
        draw.text((col_x[1], y + 20), str(row["STK"]), fill=(10, 10, 10), font=text_font)
        draw.text((col_x[2], y + 20), str(row["Số tiền"]), fill=(10, 10, 10), font=text_font)
        draw.text((col_x[3], y + 20), str(row["Ngân hàng"]), fill=(10, 10, 10), font=text_font)

        if apply_tamper and i == 1:
            tamper_bbox = (col_x[2] - 5, y + 5, col_x[3] - 15, y + 55)

        y += 60

    # Apply digital editing / Photoshop tamper artifact simulation
    if apply_tamper and tamper_bbox:
        crop_box = (int(tamper_bbox[0]), int(tamper_bbox[1]), int(tamper_bbox[2]), int(tamper_bbox[3]))
        
        # Save base image as high quality JPEG first
        base_buf = io.BytesIO()
        img.save(base_buf, 'JPEG', quality=98)
        base_buf.seek(0)
        img = Image.open(base_buf).convert('RGB')
        
        # Create a tampered patch by drawing edited text and saving at low JPEG quality
        patch_w = int(tamper_bbox[2] - tamper_bbox[0])
        patch_h = int(tamper_bbox[3] - tamper_bbox[1])
        patch = Image.new('RGB', (patch_w, patch_h), color=(255, 255, 200))
        p_draw = ImageDraw.Draw(patch)
        p_draw.text((5, 15), "500,000,000 VND", fill=(200, 0, 0), font=text_font)
        
        # Re-compress patch to create ELA error anomaly
        p_buf = io.BytesIO()
        patch.save(p_buf, 'JPEG', quality=10)
        p_buf.seek(0)
        tampered_patch = Image.open(p_buf)
        
        img.paste(tampered_patch, crop_box)

    img.save(output_path, "PNG")


if __name__ == "__main__":
    create_demo_dataset()
