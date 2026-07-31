"""
BankGuard Demo Data Generator
-----------------------------
Generates synthetic CSV files (15 rows) and paper scan document images for testing all 6 fraud cases:
1. Matched rows (100% match)
2. Amount Fraud (Modified amount on paper)
3. Account Fraud (Altered STK on paper)
4. Name Fraud (Altered beneficiary name on paper)
5. Extra Row (Ghost transaction on paper, not in CSV)
6. Missing Row (Transaction in CSV, omitted from paper)
7. Tampered Image (Photoshop/compression artifact overlay)
"""

import os
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io


def create_demo_dataset(output_dir: str = "demo_data"):
    os.makedirs(output_dir, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Baseline Original CSV File (14 Original Transactions)
    # -------------------------------------------------------------
    csv_data = [
        {"Tên": "NGUYEN VAN AN", "STK": "1012345678", "Số tiền": 50000000, "Ngân hàng": "Vietcombank"},
        {"Tên": "TRAN THI BINH", "STK": "9988776655", "Số tiền": 50000000, "Ngân hàng": "Techcombank"},
        {"Tên": "LE HOANG CUONG", "STK": "0071000123", "Số tiền": 35000000, "Ngân hàng": "BIDV"},
        {"Tên": "PHAM MINH DUC", "STK": "2200155999", "Số tiền": 85000000, "Ngân hàng": "Agribank"},
        {"Tên": "HOANG THI EM", "STK": "3300445566", "Số tiền": 12500000, "Ngân hàng": "Vietinbank"},
        {"Tên": "DANG VAN PHONG", "STK": "5544332211", "Số tiền": 45000000, "Ngân hàng": "MBBank"},
        {"Tên": "BUI THI GIANG", "STK": "6677889900", "Số tiền": 150000000, "Ngân hàng": "Sacombank"},
        {"Tên": "NGO QUAN HUNG", "STK": "1122334455", "Số tiền": 95000000, "Ngân hàng": "VPBank"},
        {"Tên": "TRINH THI INH", "STK": "9900112233", "Số tiền": 200000000, "Ngân hàng": "TPBank"},
        {"Tên": "VUONG QUOC KHANH", "STK": "4455667788", "Số tiền": 30000000, "Ngân hàng": "ACB"},
        {"Tên": "LY THI LAM", "STK": "7788990011", "Số tiền": 65000000, "Ngân hàng": "Vietcombank"},
        {"Tên": "MAI VAN NAM", "STK": "8899001122", "Số tiền": 18000000, "Ngân hàng": "Techcombank"},
        {"Tên": "PHAN THI OANH", "STK": "1231231234", "Số tiền": 42000000, "Ngân hàng": "BIDV"},
        {"Tên": "QUACH VAN PHAT", "STK": "3213214321", "Số tiền": 88000000, "Ngân hàng": "Agribank"},
    ]
    df_csv = pd.DataFrame(csv_data)
    csv_path = os.path.join(output_dir, "bank_transactions.csv")
    df_csv.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"Generated original CSV dataset (14 rows): {csv_path}")

    # -------------------------------------------------------------
    # 2. Paper Scan Table Data (14 Rows + 1 Ghost Extra Row)
    # -------------------------------------------------------------
    paper_rows = [
        # 1. Matched 100%
        {"Tên": "NGUYEN VAN AN", "STK": "1012345678", "Số tiền": "50,000,000 VND", "Ngân hàng": "VCB"},
        # 2. Amount Fraud (500,000,000 vs 50,000,000 in CSV) + ELA Photoshop Tamper
        {"Tên": "TRAN THI BINH", "STK": "9988776655", "Số tiền": "500,000,000 VND", "Ngân hàng": "TCB"},
        # 3. Account Fraud (Altered STK: 0071999999 vs 0071000123)
        {"Tên": "LE HOANG CUONG", "STK": "0071999999", "Số tiền": "35,000,000 VND", "Ngân hàng": "BIDV"},
        # 4. Name Fraud (Altered Name: HOANG VAN PHUC vs PHAM MINH DUC)
        {"Tên": "HOANG VAN PHUC", "STK": "2200155999", "Số tiền": "85,000,000 VND", "Ngân hàng": "AGRIBANK"},
        # 5. Matched
        {"Tên": "HOANG THI EM", "STK": "3300445566", "Số tiền": "12,500,000 VND", "Ngân hàng": "CTG"},
        # 6. Matched
        {"Tên": "DANG VAN PHONG", "STK": "5544332211", "Số tiền": "45,000,000 VND", "Ngân hàng": "MB"},
        # 7. Amount Fraud (1,500,000,000 vs 150,000,000 in CSV)
        {"Tên": "BUI THI GIANG", "STK": "6677889900", "Số tiền": "1,500,000,000 VND", "Ngân hàng": "STB"},
        # 8. Matched
        {"Tên": "NGO QUAN HUNG", "STK": "1122334455", "Số tiền": "95,000,000 VND", "Ngân hàng": "VPB"},
        # 9. Name Fraud (LE THI HONG vs TRINH THI INH)
        {"Tên": "LE THI HONG", "STK": "9900112233", "Số tiền": "200,000,000 VND", "Ngân hàng": "TPB"},
        # 10. Matched
        {"Tên": "VUONG QUOC KHANH", "STK": "4455667788", "Số tiền": "30,000,000 VND", "Ngân hàng": "ACB"},
        # 11. Matched
        {"Tên": "LY THI LAM", "STK": "7788990011", "Số tiền": "65,000,000 VND", "Ngân hàng": "VCB"},
        # 12. Matched
        {"Tên": "PHAN THI OANH", "STK": "1231231234", "Số tiền": "42,000,000 VND", "Ngân hàng": "BIDV"},
        # 13. Account Fraud (Altered STK: 3213219999 vs 3213214321)
        {"Tên": "QUACH VAN PHAT", "STK": "3213219999", "Số tiền": "88,000,000 VND", "Ngân hàng": "AGRIBANK"},
        # 14. Extra Row (Ghost row on paper, not in CSV)
        {"Tên": "VO VAN GHOST", "STK": "8888999900", "Số tiền": "250,000,000 VND", "Ngân hàng": "MBBANK"},
    ]

    # Note: MAI VAN NAM (STK: 8899001122) in CSV is omitted from paper scan to test MISSING_ROW detection!

    full_image_path = os.path.join(output_dir, "bank_statement_demo.png")
    generate_document_image(paper_rows, full_image_path, apply_tamper=True)
    print(f"Generated paper scan document image: {full_image_path}")


def generate_document_image(rows_data: list, output_path: str, apply_tamper: bool = False):
    width, height = 1000, 180 + len(rows_data) * 55
    img = Image.new('RGB', (width, height), color=(250, 252, 255))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        header_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        text_font = ImageFont.truetype("DejaVuSans.ttf", 13)
    except IOError:
        title_font = header_font = text_font = ImageFont.load_default()

    # Draw Header Section
    draw.rectangle([0, 0, width, 70], fill=(22, 54, 93))
    draw.text((30, 20), "NGÂN HÀNG THƯƠNG MẠI CỔ PHẦN - BẢNG KÊ GIAO DỊCH DỰ THƯỞNG", fill=(255, 255, 255), font=title_font)

    # Column coordinates
    col_x = [30, 280, 500, 750, 970]
    headers = ["HỌ VÀ TÊN THỤ HƯỞNG", "SỐ TÀI KHOẢN (STK)", "SỐ TIỀN GIAO DỊCH", "NGÂN HÀNG"]

    # Draw Table Header Row
    y_start = 85
    draw.rectangle([col_x[0] - 10, y_start, col_x[-1], y_start + 35], fill=(220, 230, 242), outline=(150, 170, 200))
    for idx, header in enumerate(headers):
        draw.text((col_x[idx], y_start + 8), header, fill=(20, 30, 50), font=header_font)

    # Draw Data Rows
    y = y_start + 35
    tamper_bbox = None

    for i, row in enumerate(rows_data):
        fill_color = (255, 255, 255) if i % 2 == 0 else (245, 248, 252)
        draw.rectangle([col_x[0] - 10, y, col_x[-1], y + 50], fill=fill_color, outline=(210, 220, 230))

        draw.text((col_x[0], y + 15), str(row["Tên"]), fill=(10, 10, 10), font=text_font)
        draw.text((col_x[1], y + 15), str(row["STK"]), fill=(10, 10, 10), font=text_font)
        draw.text((col_x[2], y + 15), str(row["Số tiền"]), fill=(10, 10, 10), font=text_font)
        draw.text((col_x[3], y + 15), str(row["Ngân hàng"]), fill=(10, 10, 10), font=text_font)

        # Track location for Photoshop tamper overlay simulation (on row 2 amount cell)
        if apply_tamper and i == 1:
            tamper_bbox = (col_x[2] - 5, y + 5, col_x[3] - 15, y + 45)

        y += 50

    # Apply digital editing / Photoshop tamper artifact simulation
    if apply_tamper and tamper_bbox:
        crop_box = (int(tamper_bbox[0]), int(tamper_bbox[1]), int(tamper_bbox[2]), int(tamper_bbox[3]))
        patch_w = int(tamper_bbox[2] - tamper_bbox[0])
        patch_h = int(tamper_bbox[3] - tamper_bbox[1])
        
        patch = Image.new('RGB', (patch_w, patch_h), color=(255, 255, 200))
        p_draw = ImageDraw.Draw(patch)
        p_draw.text((5, 12), "500,000,000 VND", fill=(200, 0, 0), font=text_font)
        
        p_buf = io.BytesIO()
        patch.save(p_buf, 'JPEG', quality=10)
        p_buf.seek(0)
        img.paste(Image.open(p_buf), crop_box)

    img.save(output_path, "PNG")


if __name__ == "__main__":
    create_demo_dataset()
