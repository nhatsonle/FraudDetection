"""
BankGuard - CSV vs Paper Fraud Detection MVP
--------------------------------------------
Full-stack Streamlit Dashboard for Automated Reconciliation & Anti-Fraud Detection.
"""

import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import io
import os

from utils.preprocessor import load_image, deskew_image, preprocess_for_ocr, detect_ela, draw_bounding_boxes_on_image
from utils.ocr_engine import BankGuardOCREngine
from utils.matcher import BankGuardNormalizer, AntiFraudMatcher
import generate_demo_data

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="BankGuard - CSV vs Paper Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Inject CSS for Modern Banking Aesthetics & Alert Styles
st.markdown("""
<style>
    /* Main Theme Overrides */
    .main {
        background-color: #0e1117;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #334155;
        margin-bottom: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .header-title {
        color: #38bdf8;
        font-size: 28px;
        font-weight: 700;
        margin: 0;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    
    .header-subtitle {
        color: #94a3b8;
        font-size: 14px;
        margin-top: 6px;
    }
    
    /* Prominent Fraud Alert Banner */
    .fraud-banner-critical {
        background: linear-gradient(90deg, #991b1b 0%, #dc2626 100%);
        color: #ffffff;
        padding: 16px 24px;
        border-radius: 10px;
        font-weight: 800;
        font-size: 20px;
        text-align: center;
        letter-spacing: 1px;
        box-shadow: 0 0 20px rgba(220, 38, 38, 0.6);
        margin: 15px 0 25px 0;
        animation: pulse 2s infinite;
    }
    
    .matched-banner-success {
        background: linear-gradient(90deg, #065f46 0%, #10b981 100%);
        color: #ffffff;
        padding: 16px 24px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 18px;
        text-align: center;
        margin: 15px 0 25px 0;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }

    /* Metric Cards */
    .metric-box {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 16px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-value-total { color: #38bdf8; font-size: 32px; font-weight: 700; }
    .metric-value-matched { color: #34d399; font-size: 32px; font-weight: 700; }
    .metric-value-fraud { color: #f87171; font-size: 32px; font-weight: 700; }
    .metric-label { color: #94a3b8; font-size: 13px; font-weight: 600; text-transform: uppercase; }

    /* Custom Badges for Data Table */
    .badge-matched {
        background-color: #064e3b;
        color: #6ee7b7;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-size: 12px;
    }
    .badge-fraud {
        background-color: #7f1d1d;
        color: #fca5a5;
        padding: 4px 8px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_ocr_engine():
    """Cache OCR engine instance across user interactions."""
    return BankGuardOCREngine(use_gpu=False)


def main():
    # Header Banner
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🛡️ BankGuard - CSV vs Paper Fraud Detection MVP</div>
        <div class="header-subtitle">Hệ thống Đối soát Tự động & Phát hiện Gian lận Tài chính (Sửa tiền, Đổi STK, Đổi tên, Tẩy xóa Physical/Digital)</div>
    </div>
    """, unsafe_allow_html=True)

    ocr_engine = get_ocr_engine()
    matcher = AntiFraudMatcher()

    # -------------------------------------------------------------
    # Sidebar - Control & File Upload
    # -------------------------------------------------------------
    st.sidebar.header("📁 Step 1: Input Data Layer")
    
    # Quick Demo Data Generation Button
    st.sidebar.markdown("---")
    st.sidebar.subheader("⚡ Quick Demo")
    if st.sidebar.button("🚀 Nạp Bộ Dữ Liệu Demo Mẫu", type="primary", use_container_width=True):
        generate_demo_data.create_demo_dataset("demo_data")
        st.session_state["use_demo"] = True
        st.sidebar.success("Đã nạp bộ dữ liệu Demo thành công!")

    st.sidebar.markdown("---")
    st.sidebar.subheader("📤 Tải File Thủ Công")
    uploaded_csv = st.sidebar.file_uploader("1. File CSV gốc từ đối tác", type=["csv"])
    uploaded_img = st.sidebar.file_uploader("2. Ảnh scan / chụp bản cứng", type=["png", "jpg", "jpeg", "pdf"])

    # Determine files to process
    csv_bytes = None
    img_bytes = None

    if uploaded_csv and uploaded_img:
        csv_bytes = uploaded_csv.getvalue()
        img_bytes = uploaded_img.getvalue()
    elif st.session_state.get("use_demo", True):
        # Auto load demo files if exist or generate them
        demo_csv_path = "demo_data/bank_transactions.csv"
        demo_img_path = "demo_data/bank_statement_demo.png"
        if not os.path.exists(demo_csv_path) or not os.path.exists(demo_img_path):
            generate_demo_data.create_demo_dataset("demo_data")

        with open(demo_csv_path, "rb") as f:
            csv_bytes = f.read()
        with open(demo_img_path, "rb") as f:
            img_bytes = f.read()

    if not csv_bytes or not img_bytes:
        st.info("👋 Vui lòng tải lên 2 file CSV & Ảnh bản cứng ở góc trái, hoặc bấm **'🚀 Nạp Bộ Dữ Liệu Demo Mẫu'** để khởi chạy ngay.")
        return

    # Load CSV & Image Data
    df_csv = pd.read_csv(io.BytesIO(csv_bytes))
    orig_bgr = load_image(img_bytes)

    # -------------------------------------------------------------
    # Step 2 & 3: Preprocessing & OCR Extraction
    # -------------------------------------------------------------
    with st.spinner("🔍 Đang chạy Preprocessing, Deskew & OCR Layout Extraction..."):
        # Deskew & Preprocess image for OCR
        deskewed_bgr = deskew_image(orig_bgr)
        ocr_prep = preprocess_for_ocr(deskewed_bgr)
        
        # OCR Text & Table Extraction
        df_paper, cell_bboxes = ocr_engine.extract_structured_table(deskewed_bgr)

    # -------------------------------------------------------------
    # Step 5: Anti-Fraud Matching Engine
    # -------------------------------------------------------------
    df_reconciled, summary = matcher.reconcile(df_csv, df_paper)

    # -------------------------------------------------------------
    # Step 6: ELA Document Tampering Detection
    # -------------------------------------------------------------
    with st.spinner("🔬 Đang phân tích Error Level Analysis (ELA) phát hiện can thiệp Photoshop..."):
        orig_pil, ela_heatmap, tamper_score, is_tampered = detect_ela(img_bytes)

    # Combined Fraud Flag
    has_total_fraud = summary['has_fraud'] or is_tampered

    # -------------------------------------------------------------
    # 1. SUMMARY BAR & CRITICAL FRAUD BANNER
    # -------------------------------------------------------------
    if has_total_fraud:
        st.markdown("""
        <div class="fraud-banner-critical">
            ⚠️ PHÁT HIỆN GIAN LẬN / SAI LỆCH - KHÓA LỆNH DUYỆT CHI TỰ ĐỘNG
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="matched-banner-success">
            ✅ TẤT CẢ DỮ LIỆU KHỚP 100% - AN TOÀN ĐỂ PHÊ DUYỆT CHI
        </div>
        """, unsafe_allow_html=True)

    # Summary Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value-total">{summary['total_rows']}</div>
            <div class="metric-label">Tổng số dòng đối soát</div>
        </div>
        """, unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value-matched">{summary['matched_count']}</div>
            <div class="metric-label">Số dòng khớp (Match)</div>
        </div>
        """, unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value-fraud">{summary['fraud_count']}</div>
            <div class="metric-label">Số dòng Gian lận / Sai lệch</div>
        </div>
        """, unsafe_allow_html=True)
    with m4:
        ela_color = "#f87171" if is_tampered else "#34d399"
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value-total" style="color: {ela_color}">{tamper_score}%</div>
            <div class="metric-label">Chỉ số Nghi ngờ ELA Tamper</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # -------------------------------------------------------------
    # 2. MAIN SECTION (SPLIT-SCREEN VIEWER)
    # -------------------------------------------------------------
    col_left, col_right = st.columns([1.1, 1.4])

    with col_left:
        st.subheader("📸 Split-Screen Image & ELA Forensics")
        
        # Prepare Bounding Box Highlight overlay on the original paper image
        boxes_to_draw = []
        for _, rec_row in df_reconciled.iterrows():
            if rec_row['Is_Fraud'] and rec_row['Paper_Row_Idx'] >= 0:
                p_idx = rec_row['Paper_Row_Idx']
                if p_idx < len(cell_bboxes):
                    bboxes_dict = cell_bboxes[p_idx]['fields']
                    mismatch_fields = rec_row.get('Mismatch_Fields', [])
                    for field in mismatch_fields:
                        if field in bboxes_dict:
                            boxes_to_draw.append({
                                'bbox': bboxes_dict[field],
                                'label': f"FRAUD: {field.upper()}",
                                'color': (0, 0, 255),
                                'is_fraud': True
                            })

        # Draw red boxes on image
        annotated_img = draw_bounding_boxes_on_image(deskewed_bgr, boxes_to_draw)
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        tab_img, tab_ela = st.tabs(["🖼️ Báo Xấu Bounding Box", "🔥 Heatmap ELA Tamper"])
        
        with tab_img:
            st.image(annotated_rgb, caption="Bản Scan gốc với Bounding Box Màu Đỏ tại vị trí Gian lận / Sai lệch", use_container_width=True)
            if boxes_to_draw:
                st.error(f"🚨 Đã vẽ {len(boxes_to_draw)} Bounding Box khoanh vùng trực tiếp trên bản Scan!")

        with tab_ela:
            st.image(ela_heatmap, caption="Error Level Analysis (ELA) Heatmap - Vùng màu Vàng/Đỏ cảnh báo nén bất thường / Photoshop", use_container_width=True)
            if is_tampered:
                st.warning(f"⚠️ Cảnh báo Forensics: Phát hiện chỉ số nén bất thường ({tamper_score}%). Vùng ảnh có khả năng đã bị chỉnh sửa kỹ thuật số.")

    with col_right:
        st.subheader("📊 Bảng Đối Soát Trực Tiếp (CSV vs Paper)")

        # Render clean comparison table
        display_df = df_reconciled[[
            'Tên_CSV', 'STK_CSV', 'Amount_CSV', 'Bank_CSV',
            'Tên_Paper', 'STK_Paper', 'Amount_Paper', 'Bank_Paper',
            'Fraud_Status'
        ]].copy()

        # Format Currency displays
        display_df['Amount_CSV'] = display_df['Amount_CSV'].apply(lambda x: f"{x:,.0f} đ" if x > 0 else "-")
        display_df['Amount_Paper'] = display_df['Amount_Paper'].apply(lambda x: f"{x:,.0f} đ" if x > 0 else "-")

        # Custom styling for status column
        def style_fraud_status(val):
            if 'FRAUD' in str(val):
                return 'background-color: #7f1d1d; color: #fca5a5; font-weight: bold;'
            return 'background-color: #064e3b; color: #6ee7b7; font-weight: bold;'

        styled_table = display_df.style.applymap(style_fraud_status, subset=['Fraud_Status'])
        st.dataframe(styled_table, use_container_width=True, height=450)

    # -------------------------------------------------------------
    # 3. FRAUD AUDIT LOG & EXPORT REPORT
    # -------------------------------------------------------------
    st.markdown("---")
    st.subheader("📋 Fraud Audit Log & Báo Cáo Chi Tiết")

    audit_df = df_reconciled[[
        'STK_CSV', 'Tên_CSV', 'Amount_CSV', 'Bank_CSV',
        'STK_Paper', 'Tên_Paper', 'Amount_Paper', 'Bank_Paper',
        'Fraud_Status', 'Fraud_Detail'
    ]].copy()

    st.dataframe(audit_df, use_container_width=True)

    # Export Report Buttons
    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        csv_buffer = audit_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Tải Báo Cáo Gian Lận (File CSV)",
            data=csv_buffer,
            file_name="BankGuard_Fraud_Audit_Report.csv",
            mime="text/csv",
            use_container_width=True
        )
    with c_exp2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            audit_df.to_excel(writer, index=False, sheet_name='Fraud Audit Report')
        st.download_button(
            label="📊 Tải Báo Cáo Gian Lận (File Excel)",
            data=excel_buffer.getvalue(),
            file_name="BankGuard_Fraud_Audit_Report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )


if __name__ == "__main__":
    main()
