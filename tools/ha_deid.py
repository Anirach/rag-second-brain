#!/usr/bin/env python3
"""
HA Database De-Identification Tool
Strips/masks identifiable fields from query results before they enter LLM context.

Usage:
  # Pipe psql output through de-ID
  PGPASSWORD='...' psql -h ... -c "SELECT * FROM ..." | python3 tools/ha_deid.py

  # Or de-ID a saved file
  python3 tools/ha_deid.py < raw_output.txt > safe_output.txt

  # Or use in Python
  from tools.ha_deid import deid_text, deid_rows
"""
import sys
import re
import hashlib
import json
import csv
import io

# Fields that contain identifiable information — ALWAYS mask these
SENSITIVE_FIELDS = {
    'hospital_name', 'old_hospital_name', 'hospital_english_name',
    'hospital_code', 'internal_code',
    'leader_position', 'organization_name',
    'notes', 'alro_note',
    # Person-related
    'name', 'employee_name', 'surveyor_name', 'speaker_name',
    'full_name', 'first_name', 'last_name', 'nickname',
    'email', 'phone', 'mobile', 'address',
    # Contact & communication
    'phone_number', 'telephone', 'tel', 'fax', 'contact',
    'contact_name', 'contact_person', 'contact_phone', 'contact_email',
    'contact_number', 'contact_info', 'contact_detail',
    'line_id', 'social_media',
    # ID numbers
    'citizen_id', 'national_id', 'id_card', 'passport',
    'license_number', 'registration_number',
    # Any field with "name", "phone", "contact", "email" in it
}

# Fields that are safe (aggregated/categorical)
SAFE_FIELDS = {
    'region', 'province', 'district', 'affiliation', 'hospital_type',
    'beds_by_framework', 'actual_beds', 'pct_count',
    'publication_level', 'current_ha_level', 'current_dhsa_level',
    'ha_accreditation_date', 'ha_expiry_date',
    'dhsa_accreditation_date', 'dhsa_expiry_date',
    'count', 'cnt', 'total', 'avg', 'sum', 'min', 'max',
    'membership', 'uc', 'th_life', 'spk',
    'score_53', 'score_64', 'score_75',  # scores OK without hospital name
}


class DeIdentifier:
    """Session-consistent de-identification. Same input always maps to same pseudonym."""

    def __init__(self, seed="ha-deid-session"):
        self.seed = seed
        self.hospital_map = {}
        self.person_map = {}
        self.code_map = {}
        self.counters = {'hospital': 0, 'person': 0, 'code': 0}

    def _hash_key(self, value):
        return hashlib.md5(f"{self.seed}:{value}".encode()).hexdigest()[:8]

    def mask_hospital(self, name):
        if not name or name.strip() == '':
            return name
        name = name.strip()
        if name not in self.hospital_map:
            self.counters['hospital'] += 1
            self.hospital_map[name] = f"Hospital_{self.counters['hospital']:03d}"
        return self.hospital_map[name]

    def mask_person(self, name):
        if not name or name.strip() == '':
            return name
        name = name.strip()
        if name not in self.person_map:
            self.counters['person'] += 1
            self.person_map[name] = f"Person_{self.counters['person']:03d}"
        return self.person_map[name]

    def mask_code(self, code):
        if not code or code.strip() == '':
            return code
        code = code.strip()
        if code not in self.code_map:
            self.counters['code'] += 1
            self.code_map[code] = f"CODE_{self.counters['code']:03d}"
        return self.code_map[code]

    def mask_phone(self, phone):
        if not phone or str(phone).strip() == '':
            return phone
        phone = str(phone).strip()
        if phone not in self.person_map:
            self.counters['person'] += 1
            self.person_map[phone] = f"PHONE_{self.counters['person']:03d}"
        return self.person_map[phone]

    def mask_email(self, email):
        if not email or str(email).strip() == '':
            return email
        email = str(email).strip()
        if email not in self.person_map:
            self.counters['person'] += 1
            self.person_map[email] = f"EMAIL_{self.counters['person']:03d}"
        return self.person_map[email]

    def mask_id(self, id_val):
        if not id_val or str(id_val).strip() == '':
            return id_val
        id_val = str(id_val).strip()
        if id_val not in self.code_map:
            self.counters['code'] += 1
            self.code_map[id_val] = f"ID_{self.counters['code']:03d}"
        return self.code_map[id_val]

    def mask_field(self, field_name, value):
        """Mask a value based on its field name."""
        if value is None or str(value).strip() == '':
            return value

        fn = field_name.lower().strip()

        # Check if it's a sensitive field
        if fn in SENSITIVE_FIELDS or any(s in fn for s in ['name', 'note', 'email', 'phone', 'address', 'contact', 'mobile', 'tel', 'fax']):
            if 'hospital' in fn or fn == 'organization_name':
                return self.mask_hospital(str(value))
            elif 'code' in fn or 'license' in fn or 'registration' in fn or 'citizen' in fn or 'national_id' in fn or 'passport' in fn or 'id_card' in fn:
                return self.mask_id(str(value))
            elif 'email' in fn:
                return self.mask_email(str(value))
            elif any(p in fn for p in ['phone', 'mobile', 'tel', 'fax', 'contact_number']):
                return self.mask_phone(str(value))
            elif 'note' in fn:
                return '[REDACTED]'
            elif 'contact' in fn or 'address' in fn or 'line_id' in fn or 'social' in fn:
                return '[REDACTED]'
            else:
                return self.mask_person(str(value))

        # Safe fields pass through
        return value

    def deid_rows(self, headers, rows):
        """De-identify a list of rows given headers."""
        safe_rows = []
        for row in rows:
            safe_row = []
            for i, val in enumerate(row):
                if i < len(headers):
                    safe_row.append(self.mask_field(headers[i], val))
                else:
                    safe_row.append(val)
            safe_rows.append(safe_row)
        return safe_rows

    def deid_psql_output(self, text):
        """De-identify psql tabular output format."""
        lines = text.split('\n')
        if len(lines) < 3:
            return text

        # Find header line (contains | separators)
        header_idx = None
        for i, line in enumerate(lines):
            if '|' in line and i + 1 < len(lines) and set(lines[i + 1].strip()).issubset({'-', '+', ' '}):
                header_idx = i
                break

        if header_idx is None:
            # No tabular output detected, redact Thai hospital names as fallback
            return self._redact_thai_names(text)

        headers = [h.strip() for h in lines[header_idx].split('|')]

        result_lines = []
        for i, line in enumerate(lines):
            if i <= header_idx + 1:  # Header + separator
                result_lines.append(line)
                continue

            if '|' not in line:
                result_lines.append(line)
                continue

            # Data row
            values = line.split('|')
            masked_values = []
            for j, val in enumerate(values):
                if j < len(headers):
                    masked = self.mask_field(headers[j], val.strip())
                    masked_values.append(f" {masked} ")
                else:
                    masked_values.append(val)
            result_lines.append('|'.join(masked_values))

        return '\n'.join(result_lines)

    def _redact_thai_names(self, text):
        """Fallback: redact patterns that look like Thai hospital names, phones, emails."""
        # Redact Thai text that might be hospital names (sequences of Thai chars > 5)
        result = re.sub(r'[ก-๙]{5,}(?:\s+[ก-๙]{2,})*', '[THAI_REDACTED]', text)
        # Redact hospital codes like DS 02003, PS 25001
        result = re.sub(r'\b[A-Z]{2}\s?\d{4,5}\b', '[CODE_REDACTED]', result)
        # Redact Thai phone numbers (0x-xxx-xxxx, 0xx-xxx-xxxx, +66...)
        result = re.sub(r'(?:\+66|0)\d[\d\s\-]{7,12}', '[PHONE_REDACTED]', result)
        # Redact email addresses
        result = re.sub(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}', '[EMAIL_REDACTED]', result)
        # Redact Thai citizen IDs (13 digits, sometimes with dashes)
        result = re.sub(r'\b\d[\-\s]?\d{4}[\-\s]?\d{5}[\-\s]?\d{2}[\-\s]?\d\b', '[CITIZEN_ID_REDACTED]', result)
        return result

    def stats(self):
        return {
            'hospitals_masked': self.counters['hospital'],
            'persons_masked': self.counters['person'],
            'codes_masked': self.counters['code'],
        }

    def save_mapping(self, path='/home/clawdbot/clawd/tmp/ha_deid_mapping.json'):
        """Save mapping to local JSON file (NEVER send this to LLM/API)."""
        import json
        mapping = {
            'hospitals': self.hospital_map,
            'persons': self.person_map,
            'codes': self.code_map,
            'reverse_hospitals': {v: k for k, v in self.hospital_map.items()},
            'reverse_persons': {v: k for k, v in self.person_map.items()},
            'reverse_codes': {v: k for k, v in self.code_map.items()},
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
        return path

    @staticmethod
    def load_mapping(path='/home/clawdbot/clawd/tmp/ha_deid_mapping.json'):
        """Load saved mapping from JSON file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def reverse_text(self, text):
        """Reverse de-identification: replace pseudonyms back to real names.
        Use this ONLY for local report generation — never send reversed text to LLM."""
        result = text
        # Reverse in order: longest pseudonyms first to avoid partial matches
        all_mappings = {}
        all_mappings.update({v: k for k, v in self.hospital_map.items()})
        all_mappings.update({v: k for k, v in self.person_map.items()})
        all_mappings.update({v: k for k, v in self.code_map.items()})
        # Sort by pseudonym length descending (avoid Hospital_1 matching before Hospital_10)
        for pseudo, real in sorted(all_mappings.items(), key=lambda x: -len(x[0])):
            result = result.replace(pseudo, real)
        # Restore [REDACTED] notes if we have them cached
        return result

    def reverse_docx(self, input_path, output_path):
        """Reverse de-identification in a DOCX file.
        Reads the de-identified DOCX, replaces all pseudonyms with real names,
        saves to output_path. Use for final deliverable reports ONLY."""
        from docx import Document
        doc = Document(input_path)
        all_mappings = {}
        all_mappings.update({v: k for k, v in self.hospital_map.items()})
        all_mappings.update({v: k for k, v in self.person_map.items()})
        all_mappings.update({v: k for k, v in self.code_map.items()})
        sorted_maps = sorted(all_mappings.items(), key=lambda x: -len(x[0]))

        def _replace(text):
            for pseudo, real in sorted_maps:
                text = text.replace(pseudo, real)
            return text

        # Replace in paragraphs
        for para in doc.paragraphs:
            for run in para.runs:
                if any(pseudo in run.text for pseudo, _ in sorted_maps):
                    run.text = _replace(run.text)

        # Replace in tables
        for table in doc.tables:
            for row in table.rows:
                for cell in row.cells:
                    for para in cell.paragraphs:
                        for run in para.runs:
                            if any(pseudo in run.text for pseudo, _ in sorted_maps):
                                run.text = _replace(run.text)

        doc.save(output_path)
        return output_path


# Module-level instance for easy use
_deid = DeIdentifier()

def deid_text(text):
    """Quick de-identify psql output text."""
    return _deid.deid_psql_output(text)

def deid_rows(headers, rows):
    """Quick de-identify structured rows."""
    return _deid.deid_rows(headers, rows)

def reset():
    """Reset mappings (new session)."""
    global _deid
    _deid = DeIdentifier()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='De-identify HA database output')
    parser.add_argument('--stats', action='store_true', help='Print masking stats')
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    args = parser.parse_args()

    text = args.infile.read()
    deid = DeIdentifier()
    result = deid.deid_psql_output(text)
    print(result)

    if args.stats:
        print(f"\n--- De-ID Stats ---", file=sys.stderr)
        for k, v in deid.stats().items():
            print(f"  {k}: {v}", file=sys.stderr)
