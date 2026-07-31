# 🛡️ BankGuard - CSV vs Paper Fraud Detection MVP

**BankGuard** là ứng dụng tự động đối soát tài chính dành cho ngành Ngân hàng, giúp phát hiện tức thì các hành vi lừa đảo và sai lệch giữa file CSV gốc từ đối tác và Ảnh chụp/Scan bản cứng do khách hàng cung cấp.

---

## 🌟 Tính Năng Nổi Bật (Key Features)

1. **Tiền Xử Lý Ảnh (Pre-processing & Deskew)**:
   - Tự động xoay/nắn thẳng góc nghiêng (Deskew) của bản chụp/scan.
   - Tăng tương phản adaptive (CLAHE) & lọc nhiễu giúp tối ưu nét chữ cho OCR.

2. **Bóc Tách Layout & OCR (OCR Engine)**:
   - Tích hợp **PaddleOCR** (uu tiên tiếng Việt) kết hợp cơ chế Fallback linh hoạt (EasyOCR / PyTesseract / Smart Layout Extractor).
   - Tự động bóc tách và xếp nhóm tọa độ Bounding Box `(x, y, w, h)` theo dòng/cột: `[Tên, STK, Số tiền, Ngân hàng]`.

3. **Thuật Toán Chuẩn Hóa & Chống Gian Lận (Anti-Fraud Matching Engine)**:
   - **Chuẩn hóa chống False Alarm**: Tự động sửa lỗi OCR kinh điển (`O/D -> 0`, `I/l -> 1`, `B -> 8`), xóa định dạng tiền tệ (`.`, `,`, `VND`, `đ`), chuyển tên tiếng Việt không dấu & map mã Ngân hàng chuẩn (VCB, TCB, BIDV,...).
   - **So sánh 2 chiều (Primary Key: `STK + BankCode`)**:
     - `FRAUD: AMOUNT_MISMATCH`: Phát hiện sửa số tiền trên giấy.
     - `FRAUD: NAME_MISMATCH`: Phát hiện sửa tên tài khoản (Fuzzy match < 90%).
     - `FRAUD: ACCOUNT_MISMATCH`: Phát hiện sửa số tài khoản.
     - `FRAUD: EXTRA_ROW`: Cảnh báo dòng giao dịch ma xuất hiện trên giấy.
     - `FRAUD: MISSING_ROW`: Cảnh báo dòng giao dịch bị giấu/xóa khỏi giấy.

4. **Kỹ Thuật Hình Sự Số (Image Forensics - ELA)**:
   - Thuật toán **Error Level Analysis (ELA)** phân tích mức nén JPEG để phát hiện vùng bị can thiệp kỹ thuật số (Photoshop, tẩy xóa, chèn số).
   - Xuất ảnh **ELA Heatmap** trực quan với điểm số nguy cơ ELA Tamper Score.

5. **Giao Diện Trực Quan (Streamlit Dashboard)**:
   - **Summary Bar & Warning Banner**: Nút cảnh báo đỏ nổi bật `"⚠️ PHÁT HIỆN GIAN LẬN - KHÓA LỆNH DUYỆT CHI"`.
   - **Split-Screen Viewer**: Vẽ khung màu đỏ (Bounding Box) đè lên trực tiếp ô bị sửa/lừa đảo trên bản scan + Tab xem Heatmap ELA.
   - **Fraud Audit Log**: Cho phép tải Báo cáo Chi tiết dạng CSV và Excel.

---

## 📁 Cấu Trúc Dự Án (Project Structure)

```text
fraud/
├── README.md                 # Hướng dẫn chi tiết sử dụng và cài đặt
├── requirements.txt           # Danh sách các thư viện cần thiết
├── generate_demo_data.py     # Script tự động tạo dữ liệu CSV mẫu & 6 Ảnh Test Cases
├── app.py                    # Giao diện chính Streamlit Dashboard
├── utils/                    # Thư mục chứa các module lõi
│   ├── __init__.py
│   ├── preprocessor.py       # Xử lý ảnh, Deskew, vẽ Bounding Box & ELA Forensics
│   ├── ocr_engine.py         # Engine PaddleOCR & Bóc tách cấu trúc bảng
│   └── matcher.py            # Thuật toán chuẩn hóa & đối soát gian lận
└── demo_data/                # (Tự động tạo) Chứa file CSV & ảnh test cases demo
    ├── bank_transactions.csv
    ├── bank_statement_demo.png
    └── test_case_1..5.png
```

---

## 🚀 Hướng Dẫn Cài Đặt & Khởi Chạy Trên Máy Local

### Bước 1: Yêu cầu môi trường
- Python >= 3.10
- Git (nếu clone từ repository)

### Bước 2: Cài đặt thư viện
Chạy lệnh sau trong Terminal:
```bash
pip install -r requirements.txt
```

### Bước 3: Tạo bộ dữ liệu mẫu (Demo Data)
Chạy script tự động sinh file CSV và các ảnh test case gian lận:
```bash
python3 generate_demo_data.py
```

### Bước 4: Khởi chạy Dashboard Streamlit
```bash
streamlit run app.py
```
Sau khi chạy, ứng dụng sẽ tự động mở tại đường dẫn: `http://localhost:8501`.

---

## ☁️ Hướng Dẫn Chạy Trên Kaggle Notebooks

### Cách A: Chạy Dashboard Web (Khuyên Dùng)

1. Mở một **Kaggle Notebook** mới.
2. Thêm Cell cài đặt dependencies:
   ```python
   !pip install -q streamlit paddlepaddle paddleocr opencv-python-headless easyocr openpyxl
   ```
3. Thêm Cell tạo dữ liệu và chạy ứng dụng qua `cloudflared` (không cần mật khẩu):
   ```python
   !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64.deb
   !dpkg -i cloudflared-linux-amd64.deb
   !python3 generate_demo_data.py
   !streamlit run app.py & cloudflared tunnel --url http://localhost:8501
   ```
4. Bấm vào link `.trycloudflare.com` xuất hiện trong console để xem giao diện Dashboard!

---

## 🧪 Các Test Cases Bắt Buộc Đã Được Xử Lý

| STT | Test Case | Mô Tả | Trạng Thái Xử Lý |
|---|---|---|---|
| **Case 1** | Matched | Dữ liệu trùng khớp 100% | ✅ Flag: `MATCHED` (Xanh) |
| **Case 2** | Amount Fraud | Sửa tiền từ 50,000,000 (CSV) -> 500,000,000 (Giấy) | ✅ Flag: `FRAUD: AMOUNT_MISMATCH` + Red Box |
| **Case 3** | Account Fraud | Sửa Số tài khoản nhận trên giấy | ✅ Flag: `FRAUD: ACCOUNT_MISMATCH` + Red Box |
| **Case 4** | Name Fraud | Đổi tên người thụ hưởng trên giấy | ✅ Flag: `FRAUD: NAME_MISMATCH (<90%)` + Red Box |
| **Case 5** | Extra Row | Thêm dòng giao dịch ma trên giấy | ✅ Flag: `FRAUD: EXTRA_ROW` |
| **Case 6** | Tampered Image | Ảnh bị can thiệp Photoshop / Tẩy xóa | ✅ Flag: `ELA Tamper Alert` + Heatmap |

---

## 🛠️ Xử Lý Lỗi Thường Gặp (Troubleshooting)

- **Lỗi `ArrowTypeError` trên Streamlit**:
  - *Đã fix*: Hệ thống tự động ép kiểu `str` cho tất cả các cột bảng biểu để đảm bảo tương thích tuyệt đối với Apache Arrow.
- **Lỗi hiển thị `??` ở Bounding Box**:
  - *Đã fix*: OpenCV không hỗ trợ Unicode tiếng Việt trực tiếp, hệ thống tự động chuẩn hóa nhãn sang ASCII không dấu (`FRAUD: SO TIEN`, `FRAUD: TEN`).
- **Tùy chọn OCR GPU**:
  - Khi chạy trên máy có Card màn hình NVIDIA hoặc Kaggle GPU, PaddleOCR sẽ tự động nhận diện tăng tốc độ xử lý gấp 5-10 lần.

---

*Hệ thống phát triển bởi Senior Software Engineer & AI Architect - BankGuard Solution.*
