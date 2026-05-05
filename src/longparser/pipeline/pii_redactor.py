"""PII redaction using fast regex patterns and Luhn validation."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Tuple

from ..schemas import Document

logger = logging.getLogger(__name__)


@dataclass
class RedactionReport:
    """Report summarizing PII redactions."""
    emails: int = 0
    phones: int = 0
    ssns: int = 0
    credit_cards: int = 0
    ip_addresses: int = 0
    names: int = 0
    organizations: int = 0
    locations: int = 0

    def summary(self) -> str:
        parts = []
        if self.emails:
            parts.append(f"{self.emails} emails")
        if self.phones:
            parts.append(f"{self.phones} phones")
        if self.ssns:
            parts.append(f"{self.ssns} SSNs")
        if self.credit_cards:
            parts.append(f"{self.credit_cards} CCs")
        if self.ip_addresses:
            parts.append(f"{self.ip_addresses} IPs")
        if self.names:
            parts.append(f"{self.names} names")
        if self.organizations:
            parts.append(f"{self.organizations} orgs")
        if self.locations:
            parts.append(f"{self.locations} locs")
        return ", ".join(parts) if parts else "No PII found"

_nlp_models: dict = {}

def _get_nlp(model_name: str):
    """Lazily load spaCy NER model."""
    if model_name not in _nlp_models:
        try:
            import spacy
            logger.info(f"Loading spaCy NER model: {model_name}")
            _nlp_models[model_name] = spacy.load(model_name)
        except ImportError:
            logger.warning("spaCy not installed. NER redaction disabled. Run: uv add spacy")
            return None
        except Exception:
            logger.warning(f"Failed to load spaCy model '{model_name}'. Run: python -m spacy download {model_name}")
            return None
    return _nlp_models[model_name]


# Pre-compiled regex patterns
_EMAIL_RE = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b', re.IGNORECASE)
_PHONE_US_RE = re.compile(r'\b(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
_PHONE_INTL_RE = re.compile(r'\+\d{1,3}[-.\s]?\d{6,14}\b')
_SSN_RE = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
_CC_RE = re.compile(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b')
_IP_RE = re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b')


def _luhn_check(card_number: str) -> bool:
    """Validate credit card number using Luhn algorithm."""
    # Remove non-digits
    digits = [int(c) for c in card_number if c.isdigit()]
    if not digits or len(digits) != 16:
        return False
        
    # Double every second digit from right to left
    for i in range(len(digits) - 2, -1, -2):
        digits[i] *= 2
        if digits[i] > 9:
            digits[i] -= 9
            
    return sum(digits) % 10 == 0


def redact_document(
    document: Document, 
    use_ner: bool = False, 
    ner_model: str = "en_core_web_sm"
) -> Tuple[Document, RedactionReport]:
    """Redact PII from document blocks, preserving originals in metadata."""
    report = RedactionReport()
    nlp = _get_nlp(ner_model) if use_ner else None
    
    for page in document.pages:
        for block in page.blocks:
            if not block.text.strip():
                continue
                
            original_text = block.text
            pii_map = {}
            new_text = original_text
            
            # 1. Emails
            for match in _EMAIL_RE.finditer(new_text):
                val = match.group(0)
                placeholder = f"[REDACTED_EMAIL_{report.emails + 1}]"
                pii_map[placeholder] = val
                new_text = new_text.replace(val, placeholder)
                report.emails += 1
                
            # 2. SSNs
            for match in _SSN_RE.finditer(new_text):
                val = match.group(0)
                placeholder = f"[REDACTED_SSN_{report.ssns + 1}]"
                pii_map[placeholder] = val
                new_text = new_text.replace(val, placeholder)
                report.ssns += 1
                
            # 3. Credit Cards
            for match in _CC_RE.finditer(new_text):
                val = match.group(0)
                if _luhn_check(val):
                    placeholder = f"[REDACTED_CC_{report.credit_cards + 1}]"
                    pii_map[placeholder] = val
                    new_text = new_text.replace(val, placeholder)
                    report.credit_cards += 1
                    
            # 4. IP Addresses
            for match in _IP_RE.finditer(new_text):
                val = match.group(0)
                # Verify IP format (octets <= 255)
                try:
                    octets = [int(x) for x in val.split('.')]
                    if all(0 <= x <= 255 for x in octets):
                        placeholder = f"[REDACTED_IP_{report.ip_addresses + 1}]"
                        pii_map[placeholder] = val
                        new_text = new_text.replace(val, placeholder)
                        report.ip_addresses += 1
                except ValueError:
                    pass

            # 5. Phones (US & Intl)
            for match in _PHONE_US_RE.finditer(new_text):
                val = match.group(0)
                if val not in pii_map.values(): # Don't double-count
                    placeholder = f"[REDACTED_PHONE_{report.phones + 1}]"
                    pii_map[placeholder] = val
                    new_text = new_text.replace(val, placeholder)
                    report.phones += 1
                    
            for match in _PHONE_INTL_RE.finditer(new_text):
                val = match.group(0)
                if val not in pii_map.values():
                    placeholder = f"[REDACTED_PHONE_{report.phones + 1}]"
                    pii_map[placeholder] = val
                    new_text = new_text.replace(val, placeholder)
                    report.phones += 1

            # 6. NER Pass (Names, Orgs, Locations)
            if nlp and new_text.strip():
                doc = nlp(new_text)
                ents_to_redact = []
                for ent in doc.ents:
                    if ent.label_ == "PERSON":
                        ents_to_redact.append((ent.text, "PERSON", ent.start_char, ent.end_char))
                    elif ent.label_ == "ORG":
                        ents_to_redact.append((ent.text, "ORG", ent.start_char, ent.end_char))
                    elif ent.label_ in ("GPE", "LOC"):
                        ents_to_redact.append((ent.text, "LOC", ent.start_char, ent.end_char))
                
                # Sort descending by char offset so string replacement doesn't shift remaining offsets
                ents_to_redact.sort(key=lambda x: x[2], reverse=True)
                
                for ent_text, ent_label, start_char, end_char in ents_to_redact:
                    if ent_label == "PERSON":
                        report.names += 1
                        placeholder = f"[REDACTED_NAME_{report.names}]"
                    elif ent_label == "ORG":
                        report.organizations += 1
                        placeholder = f"[REDACTED_ORG_{report.organizations}]"
                    else:
                        report.locations += 1
                        placeholder = f"[REDACTED_LOC_{report.locations}]"
                    
                    pii_map[placeholder] = ent_text
                    new_text = new_text[:start_char] + placeholder + new_text[end_char:]

            # Update block if changed
            if new_text != original_text:
                block.text = new_text
                block.pii_redactions = pii_map
                
    return document, report
