"""
BankGuard Normalization & Anti-Fraud Matching Engine
---------------------------------------------------
Implements:
1. Normalization Engine (STK digit repair, Amount parsing, Name accent removal, Bank mapping)
2. Anti-Fraud Matching Engine (Primary key matching, cell diffs, missing/extra row detection)
"""

import re
import unicodedata
from difflib import SequenceMatcher
import pandas as pd
from typing import List, Dict, Any, Tuple


class BankGuardNormalizer:
    """
    Normalization Engine to clean and standardize CSV & OCR extracted data,
    preventing false alarms caused by OCR noise or format variations.
    """

    # Bank alias dictionary mapping common bank names to standardized short codes
    BANK_ALIAS_MAP = {
        'VIETCOMBANK': 'VCB', 'VCB': 'VCB', 'NGOAI THUONG': 'VCB', 'NH NGOAI THUONG': 'VCB',
        'TECHCOMBANK': 'TCB', 'TCB': 'TCB', 'KTY': 'TCB', 'KY THUONG': 'TCB',
        'BIDV': 'BIDV', 'DT VA PT': 'BIDV', 'DAU TU VA PHAT TRIEN': 'BIDV',
        'AGRIBANK': 'AGRIBANK', 'VBA': 'AGRIBANK', 'NN VA PTNT': 'AGRIBANK',
        'VIETINBANK': 'CTG', 'CTG': 'CTG', 'CONG THUONG': 'CTG',
        'MB': 'MB', 'MBBANK': 'MB', 'QUAN DOI': 'MB',
        'ACB': 'ACB', 'A CHAU': 'ACB',
        'VPBANK': 'VPB', 'VPB': 'VPB', 'THUONG TIN': 'VPB',
        'TPBANK': 'TPB', 'TPB': 'TPB', 'TIEN PHONG': 'TPB',
        'SACOMBANK': 'STB', 'STB': 'STB'
    }

    @staticmethod
    def remove_vietnamese_accents(text: str) -> str:
        """Removes Vietnamese diacritics and returns uppercase ASCII string."""
        if not text or not isinstance(text, str):
            return ""
        # Decomposition & removal of combining marks
        nfd_form = unicodedata.normalize('NFD', text)
        no_accents = ''.join([c for c in nfd_form if unicodedata.category(c) != 'MN'])
        # Handle đ/Đ explicitly
        no_accents = no_accents.replace('đ', 'd').replace('Đ', 'D')
        return no_accents.strip().upper()

    @classmethod
    def normalize_name(cls, name_raw: str) -> str:
        """Normalizes account holder name: removes accents, uppercase, strips extra whitespace."""
        clean_str = cls.remove_vietnamese_accents(str(name_raw))
        # Collapse multiple spaces into single space
        return re.sub(r'\s+', ' ', clean_str)

    @staticmethod
    def normalize_stk(stk_raw: str) -> str:
        """
        Normalizes Account Number (STK):
        - Fixes classic OCR confusion: O/D -> 0, I/l -> 1, B -> 8, S -> 5, Z -> 2
        - Strips all non-digit characters
        """
        if not stk_raw or pd.isna(stk_raw):
            return ""
        s = str(stk_raw).strip()
        # Common OCR character substitutions for account numbers
        ocr_replacements = {
            'O': '0', 'o': '0', 'D': '0', 'Q': '0',
            'I': '1', 'l': '1', '|': '1', '!': '1',
            'B': '8',
            'S': '5',
            'Z': '2', 'z': '2'
        }
        for char, digit in ocr_replacements.items():
            s = s.replace(char, digit)
        # Retain only digits
        return re.sub(r'\D', '', s)

    @staticmethod
    def normalize_amount(amount_raw: Any) -> int:
        """
        Normalizes Amount (Số tiền):
        - Strips separators (., spaces), currency tokens (VND, đ, vnđ)
        - Converts string to integer
        """
        if amount_raw is None or pd.isna(amount_raw):
            return 0
        if isinstance(amount_raw, (int, float)):
            return int(amount_raw)
        
        s = str(amount_raw).upper()
        # Remove currency units & text
        s = re.sub(r'(VND|VNĐ|Đ|DONG|ĐỒNG|\$|\s)', '', s)
        # Remove punctuation separators
        s = re.sub(r'[\.,]', '', s)
        
        try:
            return int(s)
        except ValueError:
            # Fallback extract digits
            digits = re.sub(r'\D', '', s)
            return int(digits) if digits else 0

    @classmethod
    def normalize_bank(cls, bank_raw: str) -> str:
        """Maps bank names to unified standard Bank Code (e.g. Vietcombank -> VCB)."""
        if not bank_raw or pd.isna(bank_raw):
            return "UNKNOWN"
        clean = cls.remove_vietnamese_accents(str(bank_raw))
        
        for alias, code in cls.BANK_ALIAS_MAP.items():
            if alias in clean:
                return code
        return clean.split()[0] if clean else "UNKNOWN"


class AntiFraudMatcher:
    """
    Engine that performs 2-way reconciliation between CSV source records
    and Paper OCR records, generating cell-level fraud flags and warnings.
    """

    @staticmethod
    def fuzzy_name_match(name1: str, name2: str) -> float:
        """Computes similarity ratio (0.0 to 100.0) between two names."""
        if not name1 or not name2:
            return 0.0
        return round(SequenceMatcher(None, name1, name2).ratio() * 100.0, 2)

    def reconcile(
        self, 
        df_csv: pd.DataFrame, 
        df_paper: pd.DataFrame
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Reconciles df_csv vs df_paper.
        
        CSV Expected Columns: ['Tên', 'STK', 'Số tiền', 'Ngân hàng']
        Paper Expected Columns: ['Tên', 'STK', 'Số tiền', 'Ngân hàng']
        """
        # Step 1: Normalize CSV Data
        csv_clean = df_csv.copy()
        csv_clean['stk_norm'] = csv_clean['STK'].apply(BankGuardNormalizer.normalize_stk)
        csv_clean['amount_norm'] = csv_clean['Số tiền'].apply(BankGuardNormalizer.normalize_amount)
        csv_clean['name_norm'] = csv_clean['Tên'].apply(BankGuardNormalizer.normalize_name)
        csv_clean['bank_norm'] = csv_clean['Ngân hàng'].apply(BankGuardNormalizer.normalize_bank)
        csv_clean['key'] = csv_clean['stk_norm'] + "_" + csv_clean['bank_norm']

        # Step 2: Normalize Paper Data
        paper_clean = df_paper.copy()
        paper_clean['stk_norm'] = paper_clean['STK'].apply(BankGuardNormalizer.normalize_stk)
        paper_clean['amount_norm'] = paper_clean['Số tiền'].apply(BankGuardNormalizer.normalize_amount)
        paper_clean['name_norm'] = paper_clean['Tên'].apply(BankGuardNormalizer.normalize_name)
        paper_clean['bank_norm'] = paper_clean['Ngân hàng'].apply(BankGuardNormalizer.normalize_bank)
        paper_clean['key'] = paper_clean['stk_norm'] + "_" + paper_clean['bank_norm']

        reconciled_rows = []
        matched_count = 0
        fraud_count = 0

        processed_csv_keys = set()
        processed_paper_keys = set()

        # Step 3: Match Paper rows against CSV baseline
        for paper_idx, p_row in paper_clean.iterrows():
            p_key = p_row['key']
            p_stk = p_row['stk_norm']
            p_bank = p_row['bank_norm']

            # Attempt exact primary key match (STK + BankCode)
            c_matches = csv_clean[csv_clean['key'] == p_key]
            
            # If no exact bank key match, fallback match on STK only
            if c_matches.empty:
                c_matches = csv_clean[csv_clean['stk_norm'] == p_stk]

            if c_matches.empty:
                # Extra Row on Paper (Ghost transaction not in CSV)
                fraud_count += 1
                reconciled_rows.append({
                    'STK_CSV': '-',
                    'Tên_CSV': '-',
                    'Amount_CSV': 0,
                    'Bank_CSV': '-',
                    'STK_Paper': p_row['STK'],
                    'Tên_Paper': p_row['Tên'],
                    'Amount_Paper': p_row['amount_norm'],
                    'Bank_Paper': p_row['bank_norm'],
                    'Fraud_Status': 'FRAUD: EXTRA_ROW',
                    'Fraud_Detail': 'Dòng giao dịch ma trên bản cứng (Không có trong CSV gốc)',
                    'Paper_Row_Idx': paper_idx,
                    'Is_Fraud': True,
                    'Mismatch_Fields': ['STK', 'Tên', 'Số tiền', 'Ngân hàng']
                })
                processed_paper_keys.add(p_key)
            else:
                c_row = c_matches.iloc[0]
                c_key = c_row['key']
                processed_csv_keys.add(c_key)
                processed_paper_keys.add(p_key)

                mismatches = []
                flags = []

                # Cell 1: Check Amount Mismatch
                if c_row['amount_norm'] != p_row['amount_norm']:
                    mismatches.append('Số tiền')
                    flags.append('FRAUD: AMOUNT_MISMATCH')

                # Cell 2: Check Name Mismatch (Fuzzy < 90%)
                similarity = self.fuzzy_name_match(c_row['name_norm'], p_row['name_norm'])
                if similarity < 90.0:
                    mismatches.append('Tên')
                    flags.append(f'FRAUD: NAME_MISMATCH ({similarity}%)')

                # Cell 3: Check Account / Bank mismatch if fallback used
                if c_row['stk_norm'] != p_row['stk_norm']:
                    mismatches.append('STK')
                    flags.append('FRAUD: ACCOUNT_MISMATCH')

                if flags:
                    fraud_count += 1
                    status_str = " | ".join(flags)
                    detail_str = f"Sai lệch ô: {', '.join(mismatches)}"
                    is_fraud = True
                else:
                    matched_count += 1
                    status_str = "MATCHED"
                    detail_str = "Khớp 100% dữ liệu"
                    is_fraud = False

                reconciled_rows.append({
                    'STK_CSV': c_row['STK'],
                    'Tên_CSV': c_row['Tên'],
                    'Amount_CSV': c_row['amount_norm'],
                    'Bank_CSV': c_row['bank_norm'],
                    'STK_Paper': p_row['STK'],
                    'Tên_Paper': p_row['Tên'],
                    'Amount_Paper': p_row['amount_norm'],
                    'Bank_Paper': p_row['bank_norm'],
                    'Fraud_Status': status_str,
                    'Fraud_Detail': detail_str,
                    'Paper_Row_Idx': paper_idx,
                    'Is_Fraud': is_fraud,
                    'Mismatch_Fields': mismatches
                })

        # Step 4: Identify Missing Rows (In CSV but omitted from Paper scan)
        for _, c_row in csv_clean.iterrows():
            c_key = c_row['key']
            if c_key not in processed_csv_keys and c_row['stk_norm'] not in [r['STK_Paper'] for r in reconciled_rows]:
                fraud_count += 1
                reconciled_rows.append({
                    'STK_CSV': c_row['STK'],
                    'Tên_CSV': c_row['Tên'],
                    'Amount_CSV': c_row['amount_norm'],
                    'Bank_CSV': c_row['bank_norm'],
                    'STK_Paper': '-',
                    'Tên_Paper': '-',
                    'Amount_Paper': 0,
                    'Bank_Paper': '-',
                    'Fraud_Status': 'FRAUD: MISSING_ROW',
                    'Fraud_Detail': 'Dòng trong CSV bị giấu / tẩy xóa khỏi bản cứng',
                    'Paper_Row_Idx': -1,
                    'Is_Fraud': True,
                    'Mismatch_Fields': ['STK', 'Tên', 'Số tiền', 'Ngân hàng']
                })

        df_result = pd.DataFrame(reconciled_rows)

        summary = {
            'total_rows': len(reconciled_rows),
            'matched_count': matched_count,
            'fraud_count': fraud_count,
            'has_fraud': fraud_count > 0
        }

        return df_result, summary
