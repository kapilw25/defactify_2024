#!/usr/bin/env python3
"""
BibTeX Citation Validation Script
Validates newly added citations in references.bib (lines 490+)
"""

import re
import requests
from typing import Dict, List, Tuple
from dataclasses import dataclass
import sys

@dataclass
class ValidationResult:
    citation_key: str
    title: str
    status: str  # 'valid', 'warning', 'error'
    messages: List[str]
    arxiv_id: str = None
    url: str = None

class BibTeXValidator:
    def __init__(self, bib_file_path: str):
        self.bib_file_path = bib_file_path
        self.results = []

    def parse_bibtex_entry(self, entry_text: str) -> Dict[str, str]:
        """Parse a single BibTeX entry"""
        entry_dict = {}

        # Extract citation key
        key_match = re.search(r'@\w+\{([^,]+),', entry_text)
        if key_match:
            entry_dict['citation_key'] = key_match.group(1).strip()

        # Extract fields
        field_pattern = r'(\w+)\s*=\s*\{([^}]*)\}'
        for match in re.finditer(field_pattern, entry_text):
            field_name = match.group(1).strip()
            field_value = match.group(2).strip()
            entry_dict[field_name] = field_value

        return entry_dict

    def extract_arxiv_id(self, entry: Dict[str, str]) -> str:
        """Extract arXiv ID from entry"""
        if 'eprint' in entry:
            return entry['eprint']
        if 'url' in entry and 'arxiv' in entry['url'].lower():
            match = re.search(r'(\d{4}\.\d{4,5})', entry['url'])
            if match:
                return match.group(1)
        return None

    def validate_arxiv_id(self, arxiv_id: str) -> Tuple[bool, str]:
        """Validate arXiv ID format and check if it exists"""
        if not arxiv_id:
            return False, "No arXiv ID found"

        # Check format (fixed syntax error)
        if not re.match(r'^\d{4}\.\d{4,5}(v\d+)?$', arxiv_id):
            return False, f"Invalid arXiv ID format: {arxiv_id}"

        # Try to verify it exists (with timeout)
        try:
            url = f"https://arxiv.org/abs/{arxiv_id}"
            response = requests.head(url, timeout=5, allow_redirects=True)
            if response.status_code == 200:
                return True, f"arXiv ID verified: {arxiv_id}"
            else:
                return False, f"arXiv ID not found (HTTP {response.status_code}): {arxiv_id}"
        except requests.RequestException as e:
            return None, f"Could not verify arXiv ID (network error): {arxiv_id}"

    def validate_url(self, url: str) -> Tuple[bool, str]:
        """Validate URL format"""
        if not url:
            return False, "No URL provided"

        # Check for valid URL format
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        if not re.match(url_pattern, url):
            return False, f"Invalid URL format: {url}"

        return True, "URL format valid"

    def validate_required_fields(self, entry: Dict[str, str]) -> List[str]:
        """Check for required BibTeX fields"""
        issues = []
        required_fields = ['title', 'author', 'year']

        for field in required_fields:
            if field not in entry or not entry[field]:
                issues.append(f"Missing required field: {field}")

        return issues

    def validate_entry(self, entry_text: str) -> ValidationResult:
        """Validate a single BibTeX entry"""
        entry = self.parse_bibtex_entry(entry_text)

        if not entry:
            return ValidationResult(
                citation_key="UNKNOWN",
                title="UNKNOWN",
                status="error",
                messages=["Could not parse BibTeX entry"]
            )

        citation_key = entry.get('citation_key', 'UNKNOWN')
        title = entry.get('title', 'UNKNOWN')
        messages = []
        status = 'valid'

        # Check required fields
        field_issues = self.validate_required_fields(entry)
        if field_issues:
            messages.extend(field_issues)
            status = 'error'

        # Validate arXiv ID if present
        arxiv_id = self.extract_arxiv_id(entry)
        if arxiv_id:
            is_valid, msg = self.validate_arxiv_id(arxiv_id)
            if is_valid:
                messages.append(msg)
            elif is_valid is False:
                messages.append(f"⚠️ {msg}")
                if status == 'valid':
                    status = 'warning'
            else:  # None (network error)
                messages.append(f"⚠️ {msg}")

        # Validate URL if present
        if 'url' in entry:
            is_valid, msg = self.validate_url(entry['url'])
            if is_valid:
                messages.append(f"✓ {msg}")
            else:
                messages.append(f"✗ {msg}")
                if status == 'valid':
                    status = 'warning'

        return ValidationResult(
            citation_key=citation_key,
            title=title,
            status=status,
            messages=messages,
            arxiv_id=arxiv_id,
            url=entry.get('url')
        )

    def validate_all_new_entries(self):
        """Validate all newly added entries (after line 490)"""
        with open(self.bib_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find where new entries start
        new_entries_start = None
        for i, line in enumerate(lines):
            if '% ==== NEWLY ADDED REFERENCES FOR RELATED WORK SECTION ====' in line:
                new_entries_start = i + 1
                break

        if new_entries_start is None:
            print("❌ Could not find new entries marker in BibTeX file")
            return

        # Extract individual entries
        content = ''.join(lines[new_entries_start:])

        # Split by @misc, @article, @inproceedings, etc.
        entry_pattern = r'(@(?:misc|article|inproceedings|book|incollection)\{[^@]+)'
        entries = re.findall(entry_pattern, content, re.DOTALL)

        print(f"\n🔍 Found {len(entries)} newly added citations to validate\n")
        print("="*80)

        # Validate each entry
        for i, entry_text in enumerate(entries, 1):
            result = self.validate_entry(entry_text)
            self.results.append(result)

            # Print result
            status_icon = {
                'valid': '✅',
                'warning': '⚠️',
                'error': '❌'
            }[result.status]

            print(f"\n{i}. {status_icon} [{result.citation_key}]")
            print(f"   Title: {result.title[:70]}..." if len(result.title) > 70 else f"   Title: {result.title}")
            if result.arxiv_id:
                print(f"   arXiv: {result.arxiv_id}")
            if result.url:
                print(f"   URL: {result.url[:60]}..." if len(result.url) > 60 else f"   URL: {result.url}")
            for msg in result.messages:
                print(f"   • {msg}")

        print("\n" + "="*80)
        self.print_summary()

    def print_summary(self):
        """Print validation summary"""
        valid_count = sum(1 for r in self.results if r.status == 'valid')
        warning_count = sum(1 for r in self.results if r.status == 'warning')
        error_count = sum(1 for r in self.results if r.status == 'error')

        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   ✅ Valid:    {valid_count}")
        print(f"   ⚠️  Warnings: {warning_count}")
        print(f"   ❌ Errors:   {error_count}")
        print(f"   📝 Total:    {len(self.results)}\n")

        if error_count > 0:
            print("⚠️  Errors found! Please review the entries marked with ❌")
            print("   Missing required fields (title, author, year) should be corrected.\n")
        elif warning_count > 0:
            print("⚠️  Some warnings found. Review entries marked with ⚠️")
            print("   These are typically network issues or URL format warnings.\n")
        else:
            print("✅ All citations validated successfully!\n")

def main():
    bib_file = "/Users/kapilwanaskar/Downloads/word_cloud/word_cloud_project/overleaf/Defactify_Text_Shared_Task_Dataset_paper/references.bib"

    print("🔍 BibTeX Citation Validator")
    print("="*80)
    print(f"📄 Validating: {bib_file}\n")

    validator = BibTeXValidator(bib_file)
    validator.validate_all_new_entries()

if __name__ == "__main__":
    main()
