"""
BankGuard Normalization & Anti-Fraud Matching Engine
---------------------------------------------------
"""

import re
import unicodedata
from difflib import SequenceMatcher
import pandas as pd
from typing import List, Dict, Any, Tuple


class BankGuardNormalizer:
    BANK_ALIAS_MAP = {
        'VIETCOMBANK': 'VCB', 'VCB': 'VCB', 'NGOAI THUONG': 'VCB',
        'TECHCOMBANK': 'TCB', 'TCB': 'TCB', 'KY THUONG': 'TCB',
        'BIDV': 'BIDV', 'DAU TU VA PHAT TRIEN': 'BIDV',
        'AGRIBANK': 'AGRIBANK', 'VBA': 'AGRIBANK',
        'VIETINBANK': 'CTG', 'CTG': 'CTG',
        'MB': 'MB', 'MBBANK': 'MB',
        'ACB': 'ACB', 'VPBANK': 'VPB', 'TPBANK': 'TPB'
    }

    @staticmethod
    def remove_vietnamese_accents(text: str) -> str:
        if not text or not isinstance(text, str):
            return ""
        nfd_form = unicodedata.normalize('NFD', text)
        no_accents = ''.join([c for c in nfd_form if unicodedata.category(c) != 'MN'])
        return no_accents.replace('đ', 'd').replace('Đ', 'D').strip().upper()

    @classmethod
    def normalize_name(cls, name_raw: str) -> str:
        return re.sub(r'\s+', ' ', cls.remove_vietnamese_accents(str(name_raw)))

    @staticmethod
    def normalize_stk(stk_raw: str) -> str:
        if not stk_raw or pd.isna(stk_raw):
            return ""
        s = str(stk_raw).strip()
        replacements = {'O': '0', 'o': '0', 'D': '0', 'I': '1', 'l': '1', '|': '1', 'B': '8', 'S': '5', 'Z': '2'}
        for k, v in replacements.items():
            s = s.replace(k, v)
        return re.sub(r'\D', '', s)

    @staticmethod
    def normalize_amount(amount_raw: Any) -> int:
        if amount_raw is None or pd.isna(amount_raw):
            return 0
        if isinstance(amount_raw, (int, float)):
            return int(amount_raw)
        s = re.sub(r'(VND|VNĐ|Đ|\$|\s)', '', str(amount_raw).upper())
        s = re.sub(r'[\.,]', '', s)
        try:
            return int(s)
        except ValueError:
            digits = re.sub(r'\D', '', s)
            return int(digits) if digits else 0

    @classmethod
    def normalize_bank(cls, bank_raw: str) -> str:
        if not bank_raw or pd.isna(bank_raw):
            return "UNKNOWN"
        clean = cls.remove_vietnamese_accents(str(bank_raw))
        for alias, code in cls.BANK_ALIAS_MAP.items():
            if alias in clean:
                return code
        return clean.split()[0] if clean else "UNKNOWN"


class AntiFraudMatcher:
    @staticmethod
    def fuzzy_name_match(name1: str, name2: str) -> float:
        if not name1 or not name2:
            return 0.0
        return round(SequenceMatcher(None, name1, name2).ratio() * 100.0, 2)

    def reconcile(self, df_csv: pd.DataFrame, df_paper: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        csv_clean = df_csv.copy()
        csv_clean['stk_norm'] = csv_clean['STK'].apply(BankGuardNormalizer.normalize_stk)
        csv_clean['amount_norm'] = csv_clean['Số tiền'].apply(BankGuardNormalizer.normalize_amount)
        csv_clean['name_norm'] = csv_clean['Tên'].apply(BankGuardNormalizer.normalize_name)
        csv_clean['bank_norm'] = csv_clean['Ngân hàng'].apply(BankGuardNormalizer.normalize_bank)
        csv_clean['key'] = csv_clean['stk_norm'] + "_" + csv_clean['bank_norm']

        paper_clean = df_paper.copy()
        paper_clean['stk_norm'] = paper_clean['STK'].apply(BankGuardNormalizer.normalize_stk)
        paper_clean['amount_norm'] = paper_clean['Số tiền'].apply(BankGuardNormalizer.normalize_amount)
        paper_clean['name_norm'] = paper_clean['Tên'].apply(BankGuardNormalizer.normalize_name)
        paper_clean['bank_norm'] = paper_clean['Ngân hàng'].apply(BankGuardNormalizer.normalize_bank)
        paper_clean['key'] = paper_clean['stk_norm'] + "_" + paper_clean['bank_norm']

        reconciled_rows = []
        matched_count, fraud_count = 0, 0
        processed_csv_keys, processed_paper_keys = set(), set()

        for paper_idx, p_row in paper_clean.iterrows():
            p_key = p_row['key']
            p_stk = p_row['stk_norm']
            c_matches = csv_clean[csv_clean['key'] == p_key]
            if c_matches.empty:
                c_matches = csv_clean[csv_clean['stk_norm'] == p_stk]

            if c_matches.empty:
                # Extra Row (Dòng giao dịch ma)
                fraud_count += 1
                reconciled_rows.append({
                    'STK_CSV': '-', 
                    'Tên_CSV': '-', 
                    'Amount_CSV': 0, 
                    'Bank_CSV': '-',
                    'STK_Paper': str(p_row['STK']), 
                    'Tên_Paper': str(p_row['Tên']), 
                    'Amount_Paper': int(p_row['amount_norm']), 
                    'Bank_Paper': str(p_row['bank_norm']),
                    'Fraud_Status': 'FRAUD: EXTRA_ROW',
                    'Fraud_Detail': 'Dòng giao dịch ma trên bản cứng (Không có trong CSV)',
                    'Paper_Row_Idx': int(paper_idx), 
                    'Is_Fraud': True,
                    'Mismatch_Fields': ['STK', 'Tên', 'Số tiền', 'Ngân hàng']
                })
                processed_paper_keys.add(p_key)
            else:
                c_row = c_matches.iloc[0]
                processed_csv_keys.add(c_row['key'])
                processed_paper_keys.add(p_key)
                mismatches, flags = [], []

                if c_row['amount_norm'] != p_row['amount_norm']:
                    mismatches.append('Số tiền')
                    flags.append('FRAUD: AMOUNT_MISMATCH')

                similarity = self.fuzzy_name_match(c_row['name_norm'], p_row['name_norm'])
                if similarity < 90.0:
                    mismatches.append('Tên')
                    flags.append(f'FRAUD: NAME_MISMATCH ({similarity}%)')

                if c_row['stk_norm'] != p_row['stk_norm']:
                    mismatches.append('STK')
                    flags.append('FRAUD: ACCOUNT_MISMATCH')

                if flags:
                    fraud_count += 1
                    status_str, is_fraud = " | ".join(flags), True
                else:
                    matched_count += 1
                    status_str, is_fraud = "MATCHED", False

                reconciled_rows.append({
                    'STK_CSV': str(c_row['STK']), 
                    'Tên_CSV': str(c_row['Tên']), 
                    'Amount_CSV': int(c_row['amount_norm']), 
                    'Bank_CSV': str(c_row['bank_norm']),
                    'STK_Paper': str(p_row['STK']), 
                    'Tên_Paper': str(p_row['Tên']), 
                    'Amount_Paper': int(p_row['amount_norm']), 
                    'Bank_Paper': str(p_row['bank_norm']),
                    'Fraud_Status': str(status_str), 
                    'Fraud_Detail': f"Sai lệch: {', '.join(mismatches)}" if mismatches else "Khớp 100%",
                    'Paper_Row_Idx': int(paper_idx), 
                    'Is_Fraud': bool(is_fraud), 
                    'Mismatch_Fields': mismatches
                })

        # Missing Rows
        for _, c_row in csv_clean.iterrows():
            if c_row['key'] not in processed_csv_keys and str(c_row['STK']) not in [r['STK_Paper'] for r in reconciled_rows]:
                fraud_count += 1
                reconciled_rows.append({
                    'STK_CSV': str(c_row['STK']), 
                    'Tên_CSV': str(c_row['Tên']), 
                    'Amount_CSV': int(c_row['amount_norm']), 
                    'Bank_CSV': str(c_row['bank_norm']),
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
        return df_result, {'total_rows': len(reconciled_rows), 'matched_count': matched_count, 'fraud_count': fraud_count, 'has_fraud': fraud_count > 0}
