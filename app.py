"""
BankGuard - CSV vs Paper Fraud Detection MVP
--------------------------------------------
Full-stack Streamlit Dashboard for Automated Reconciliation & Anti-Fraud Detection.
"""

import streamlit as st
import pandas as pd
import cv2
import io
import os

from utils.preprocessor import load_image, deskew_image, preprocess_for_ocr, detect_ela, draw_bounding_boxes_on_image
from utils.ocr_engine import BankGuardOCREngine
from utils.matcher import AntiFraudMatcher
import generate_demo_data

st.set_page_config(page_title="BankGuard - CSV vs Paper Fraud Detection", page_icon="🛡️", layout="wide")

# Custom Inject CSS for Modern Banking Aesthetics & Alert Styles
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .header-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 24px; border-radius: 12px; border: 1px solid #334155; margin-bottom: 20px;
    }
    .header-title { color: #38bdf8; font-size: 28px; font-weight: 700; }
    .fraud-banner-critical {
        background: linear-gradient(90deg, #991b1b 0%, #dc2626 100%);
        color: #ffffff; padding: 16px 24px; border-radius: 10px; font-weight: 800; font-size: 20px;
        text-align: center; box-shadow: 0 0 20px rgba(220, 38, 38, 0.6); margin: 15px 0 25px 0;
    }
    .matched-banner-success {
        background: linear-gradient(90deg, #065f46 0%, #10b981 100%);
        color: #ffffff; padding: 16px 24px; border-radius: 10px; font-weight: 700; font-size: 18px; text-align: center;
    }
    .metric-box { background-color: #1e293b; border-radius: 10px; padding: 16px; border: 1px solid #334155; text-align: center; }
    .metric-value-total { color: #38bdf8; font-size: 32px; font-weight: 700; }
    .metric-value-matched { color: #34d399; font-size: 32px; font-weight: 700; }
    .metric-value-fraud { color: #f87171; font-size: 32px; font-weight: 700; }
    .metric-label { color: #94a3b8; font-size: 13px; font-weight: 600; }
    
    /* Custom HTML Table Styling for Tunnel Reliability */
    .styled-table-container {
        overflow-x: auto;
        border-radius: 8px;
        border: 1px solid #334155;
    }
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        color: #f8fafc;
        font-family: sans-serif;
        font-size: 13px;
    }
    .custom-table th {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 10px 12px;
        text-align: left;
        border-bottom: 2px solid #334155;
    }
    .custom-table td {
        padding: 10px 12px;
        border-bottom: 1px solid #1e293b;
    }
    .status-badge-matched {
        background-color: #065f46;
        color: #34d399;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 11px;
    }
    .status-badge-fraud {
        background-color: #991b1b;
        color: #f87171;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
        font-size: 11px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_ocr_engine():
    return BankGuardOCREngine(use_gpu=False)


def main():
    st.markdown("""
    <div class="header-card">
        <div class="header-title">🛡️ BankGuard - CSV vs Paper Fraud Detection MVP</div>
        <div style="color: #94a3b8; margin-top: 5px;">Hệ thống Tự động Đối soát & Phát hiện Gian lận Tài chính (Sửa tiền, Đổi STK, Đổi tên, Tẩy xóa Physical/Digital)</div>
    </div>
    """, unsafe_allow_html=True)

    ocr_engine = get_ocr_engine()
    matcher = AntiFraudMatcher()

    # Sidebar Upload & Quick Demo
    st.sidebar.header("📁 Input Layer")
    if st.sidebar.button("🚀 Nạp Bộ Dữ Liệu Demo Mẫu", type="primary", use_container_width=True):
        generate_demo_data.create_demo_dataset("demo_data")
        st.session_state["use_demo"] = True

    uploaded_csv = st.sidebar.file_uploader("1. File CSV gốc từ đối tác", type=["csv"])
    uploaded_img = st.sidebar.file_uploader("2. Ảnh scan bản cứng", type=["png", "jpg", "jpeg"])

    csv_bytes, img_bytes = None, None
    if uploaded_csv and uploaded_img:
        csv_bytes, img_bytes = uploaded_csv.getvalue(), uploaded_img.getvalue()
    elif st.session_state.get("use_demo", True):
        demo_csv_path, demo_img_path = "demo_data/bank_transactions.csv", "demo_data/bank_statement_demo.png"
        if not os.path.exists(demo_csv_path) or not os.path.exists(demo_img_path):
            generate_demo_data.create_demo_dataset("demo_data")
        with open(demo_csv_path, "rb") as f: csv_bytes = f.read()
        with open(demo_img_path, "rb") as f: img_bytes = f.read()

    if not csv_bytes or not img_bytes:
        st.info("👋 Vui lòng tải file hoặc nhấn **'🚀 Nạp Bộ Dữ Liệu Demo Mẫu'** ở góc trái để khởi chạy.")
        return

    df_csv = pd.read_csv(io.BytesIO(csv_bytes), dtype=str)
    orig_bgr = load_image(img_bytes)

    with st.spinner("🔍 Đang chạy Preprocessing, Deskew & OCR Layout Extraction..."):
        deskewed_bgr = deskew_image(orig_bgr)
        df_paper, cell_bboxes = ocr_engine.extract_structured_table(deskewed_bgr)

    df_reconciled, summary = matcher.reconcile(df_csv, df_paper)

    with st.spinner("🔬 Đang phân tích Error Level Analysis (ELA) phát hiện can thiệp Photoshop..."):
        orig_pil, ela_heatmap, tamper_score, is_tampered = detect_ela(img_bytes)

    has_total_fraud = summary['has_fraud'] or is_tampered

    # 1. SUMMARY BAR & CRITICAL BANNER
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

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(f'<div class="metric-box"><div class="metric-value-total">{summary["total_rows"]}</div><div class="metric-label">Tổng dòng đối soát</div></div>', unsafe_allow_html=True)
    m2.markdown(f'<div class="metric-box"><div class="metric-value-matched">{summary["matched_count"]}</div><div class="metric-label">Dòng Khớp (Green)</div></div>', unsafe_allow_html=True)
    m3.markdown(f'<div class="metric-box"><div class="metric-value-fraud">{summary["fraud_count"]}</div><div class="metric-label">Dòng Gian lận (Red)</div></div>', unsafe_allow_html=True)
    ela_color = "#f87171" if is_tampered else "#34d399"
    m4.markdown(f'<div class="metric-box"><div class="metric-value-total" style="color: {ela_color}">{tamper_score}%</div><div class="metric-label">Nghi ngờ ELA Tamper</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # 2. MAIN SECTION (SPLIT-SCREEN VIEWER)
    col_left, col_right = st.columns([1.1, 1.4])
    with col_left:
        st.subheader("📸 Split-Screen Image & ELA Forensics")
        boxes_to_draw = []
        for _, rec_row in df_reconciled.iterrows():
            if rec_row['Is_Fraud'] and rec_row['Paper_Row_Idx'] >= 0:
                p_idx = rec_row['Paper_Row_Idx']
                if p_idx < len(cell_bboxes):
                    bboxes_dict = cell_bboxes[p_idx]['fields']
                    for field in rec_row.get('Mismatch_Fields', []):
                        if field in bboxes_dict:
                            boxes_to_draw.append({'bbox': bboxes_dict[field], 'label': f"FRAUD: {field.upper()}", 'color': (0, 0, 255), 'is_fraud': True})

        annotated_img = draw_bounding_boxes_on_image(deskewed_bgr, boxes_to_draw)
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        tab_img, tab_ela = st.tabs(["🖼️ Báo Xấu Bounding Box", "🔥 Heatmap ELA Tamper"])
        with tab_img:
            st.image(annotated_rgb, caption="Bản Scan gốc với Bounding Box Màu Đỏ tại vị trí Sai lệch", use_container_width=True)
        with tab_ela:
            st.image(ela_heatmap, caption="ELA Heatmap - Mức nén bất thường / Photoshop", use_container_width=True)

    with col_right:
        st.subheader("📊 Bảng Đối Soát Trực Tiếp (CSV vs Paper)")
        display_df = df_reconciled[['Tên_CSV', 'STK_CSV', 'Amount_CSV', 'Bank_CSV', 'Tên_Paper', 'STK_Paper', 'Amount_Paper', 'Bank_Paper', 'Fraud_Status']].copy()
        
        display_df['STK_CSV'] = display_df['STK_CSV'].astype(str)
        display_df['STK_Paper'] = display_df['STK_Paper'].astype(str)
        display_df['Tên_CSV'] = display_df['Tên_CSV'].astype(str)
        display_df['Tên_Paper'] = display_df['Tên_Paper'].astype(str)
        display_df['Bank_CSV'] = display_df['Bank_CSV'].astype(str)
        display_df['Bank_Paper'] = display_df['Bank_Paper'].astype(str)

        display_df['Amount_CSV'] = display_df['Amount_CSV'].apply(lambda x: f"{int(x):,.0f} đ" if str(x).isdigit() and int(x) > 0 else "-")
        display_df['Amount_Paper'] = display_df['Amount_Paper'].apply(lambda x: f"{int(x):,.0f} đ" if str(x).isdigit() and int(x) > 0 else "-")

        # Custom HTML Table Rendering to avoid Tunnel JS Chunk Loading Errors
        html_table = "<div class='styled-table-container'><table class='custom-table'>"
        html_table += "<thead><tr><th>Tên CSV</th><th>STK CSV</th><th>Số tiền CSV</th><th>Ngân hàng</th><th>Tên Giấy</th><th>STK Giấy</th><th>Số tiền Giấy</th><th>Trạng thái</th></tr></thead><tbody>"
        
        for _, row in display_df.iterrows():
            st_val = str(row['Fraud_Status'])
            badge_cls = "status-badge-fraud" if "FRAUD" in st_val else "status-badge-matched"
            html_table += f"""
            <tr>
                <td>{row['Tên_CSV']}</td>
                <td>{row['STK_CSV']}</td>
                <td>{row['Amount_CSV']}</td>
                <td>{row['Bank_CSV']}</td>
                <td>{row['Tên_Paper']}</td>
                <td>{row['STK_Paper']}</td>
                <td>{row['Amount_Paper']}</td>
                <td><span class='{badge_cls}'>{st_val}</span></td>
            </tr>
            """
        html_table += "</tbody></table></div>"
        
        st.markdown(html_table, unsafe_allow_html=True)

    # 3. FRAUD AUDIT LOG & EXPORT REPORT
    st.markdown("---")
    st.subheader("📋 Fraud Audit Log & Báo Cáo Chi Tiết")
    audit_df = df_reconciled[['STK_CSV', 'Tên_CSV', 'Amount_CSV', 'Bank_CSV', 'STK_Paper', 'Tên_Paper', 'Amount_Paper', 'Bank_Paper', 'Fraud_Status', 'Fraud_Detail']].copy()
    audit_df = audit_df.astype(str)
    
    st.dataframe(audit_df, use_container_width=True)

    c_exp1, c_exp2 = st.columns(2)
    with c_exp1:
        st.download_button("📥 Tải Báo Cáo Gian Lận (CSV)", audit_df.to_csv(index=False).encode('utf-8-sig'), "BankGuard_Fraud_Report.csv", "text/csv", use_container_width=True)
    with c_exp2:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            audit_df.to_excel(writer, index=False, sheet_name='Fraud Audit')
        st.download_button("📊 Tải Báo Cáo Gian Lận (Excel)", excel_buffer.getvalue(), "BankGuard_Fraud_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)


if __name__ == "__main__":
    main()
