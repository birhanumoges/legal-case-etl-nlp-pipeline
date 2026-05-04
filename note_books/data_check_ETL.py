"""
COMPLETE ETL PIPELINE FOR LEGAL NLP - FIXED VERSION
Processes nested folder structure: allcase/file1/html/*.html + json/*.json + metadata/all_metadata.json
Outputs: CSV, JSON, Parquet with columns: Case_ID, Year, Court, Case_Text, Verdict, Legal_Citations, Case_Type, Sub_Type
FIX: Improved Sub-Type extraction for Criminal Law cases (Larceny, Homicide, Assault, etc.)
"""

import os
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
from bs4 import BeautifulSoup
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from collections import defaultdict

# ============================================================
# CONFIGURATION
# ============================================================

ROOT_PATH = "C:/Users/DELL/Downloads/Legal Case"  # CHANGE THIS TO YOUR PATH
OUTPUT_DIR = "C:/Users/DELL/Downloads/Legal Case/processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

REQUIRED_COLUMNS = ['Case_ID', 'Year', 'Court', 'Case_Text', 'Verdict', 'Legal_Citations', 'Case_Type', 'Sub_Type']

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================
# STEP 1: FILE DISCOVERY AND MATCHING
# ============================================================

def discover_and_match_files(root_path: str) -> Dict:
    """Discover all HTML and JSON files and match them by base name"""
    structure = {}
    
    for source_folder in os.listdir(root_path):
        source_path = Path(root_path) / source_folder
        
        if not source_path.is_dir() or source_folder.startswith('.') or source_folder == "processed_data":
            continue
        
        logger.info(f"\n📁 Scanning source: {source_folder}")
        
        html_folder = source_path / "html"
        json_folder = source_path / "json"
        metadata_file = source_path / "metadata" / "all_metadata.json"
        
        if not html_folder.exists():
            logger.warning(f"  ⚠️ html folder not found in {source_folder}")
            continue
        if not json_folder.exists():
            logger.warning(f"  ⚠️ json folder not found in {source_folder}")
            continue
        
        html_files = list(html_folder.glob("*.html"))
        json_files = list(json_folder.glob("*.json"))
        
        logger.info(f"  Found {len(html_files)} HTML files, {len(json_files)} JSON files")
        
        # Match by base name (without extension)
        html_map = {f.stem: f for f in html_files}
        json_map = {f.stem: f for f in json_files}
        common_names = set(html_map.keys()) & set(json_map.keys())
        
        # Load master metadata
        master_metadata = {}
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    metadata_list = json.load(f)
                    for meta in metadata_list:
                        case_id = str(meta.get('id', ''))
                        case_name = meta.get('name', '')
                        if case_id:
                            master_metadata[case_id] = meta
                        if case_name:
                            master_metadata[case_name] = meta
                logger.info(f"  Loaded {len(master_metadata)} metadata entries")
            except Exception as e:
                logger.warning(f"  Could not load metadata: {e}")
        
        # Create case entries
        cases = []
        for name in common_names:
            case_id_match = re.search(r'(\d{4})', name)
            case_id = case_id_match.group(1) if case_id_match else name
            
            case_metadata = None
            for key in [case_id, name, case_id_match.group(0) if case_id_match else None]:
                if key and key in master_metadata:
                    case_metadata = master_metadata[key]
                    break
            
            cases.append({
                'source': source_folder,
                'case_name': name,
                'case_id': case_id,
                'html_file': html_map[name],
                'json_file': json_map[name],
                'metadata': case_metadata
            })
        
        structure[source_folder] = {
            'path': source_path,
            'cases': cases,
            'total_cases': len(cases)
        }
        
        logger.info(f"  ✅ Matched {len(cases)} cases")
        
        unmatched_html = set(html_map.keys()) - common_names
        unmatched_json = set(json_map.keys()) - common_names
        if unmatched_html:
            logger.warning(f"  ⚠️ Unmatched HTML: {list(unmatched_html)[:3]}...")
        if unmatched_json:
            logger.warning(f"  ⚠️ Unmatched JSON: {list(unmatched_json)[:3]}...")
    
    return structure

# ============================================================
# STEP 2: TEXT EXTRACTION
# ============================================================

def extract_text_from_html(html_file: Path) -> Tuple[str, int]:
    """Extract and clean text from HTML file"""
    try:
        with open(html_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        try:
            soup = BeautifulSoup(content, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            text = soup.get_text(separator=' ')
        except:
            text = content
        
        original_length = len(text)
        
        # Clean text
        text = re.sub(r'\*\d+', '', text)           # Remove page markers
        text = re.sub(r'\s+', ' ', text)             # Normalize whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)      # Normalize paragraphs
        text = text.strip()
        
        return text, original_length
    except Exception as e:
        logger.error(f"Error extracting text from {html_file}: {e}")
        return "", 0

# ============================================================
# STEP 3: JSON METADATA EXTRACTION
# ============================================================

def extract_metadata_from_json(json_file: Path) -> Dict:
    """Extract metadata from JSON file"""
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        metadata = {
            'case_name': data.get('name', ''),
            'decision_date': data.get('decision_date', ''),
            'docket_number': data.get('docket_number', ''),
            'first_page': data.get('first_page', ''),
            'last_page': data.get('last_page', ''),
            'court_name': '',
            'citations': []
        }
        
        if 'court' in data:
            court = data['court']
            metadata['court_name'] = court.get('name', '') if isinstance(court, dict) else str(court)
        
        if 'citations' in data:
            for cite in data['citations']:
                if isinstance(cite, dict) and 'cite' in cite:
                    metadata['citations'].append(cite['cite'])
        
        if 'cites_to' in data:
            for cite in data['cites_to']:
                if isinstance(cite, dict) and 'cite' in cite:
                    metadata['citations'].append(cite['cite'])
                elif isinstance(cite, str):
                    metadata['citations'].append(cite)
        
        return metadata
    except Exception as e:
        logger.error(f"Error extracting metadata from {json_file}: {e}")
        return {}

# ============================================================
# STEP 4: VERDICT EXTRACTION (IMPROVED)
# ============================================================

def extract_verdict(text: str) -> str:
    """
    Enhanced verdict extraction for legal cases.
    Handles multiple verdict locations and formats.
    """
    if not text:
        return "Verdict Unknown"
    
    text_lower = text.lower()
    
    # ============================================================
    # METHOD 1: Look for explicit judgment statements (anywhere)
    # ============================================================
    
    # Priority patterns (highest confidence)
    priority_patterns = [
        # Per Curiam (most authoritative)
        (r'per curiam[^.]*\.\s*(?:the\s+)?(?:judgment|decree|order|verdict)\s+is\s+(reversed|affirmed|remanded)', 
         lambda m: f"Per Curiam: Judgment {m.group(1).upper()}"),
        (r'per curiam[^.]*\.\s*the\s+arrest\s+of\s+judgment\s+is\s+(reversed|affirmed)', 
         lambda m: f"Per Curiam: Arrest of Judgment {m.group(1).upper()}"),
        (r'per curiam[^.]*\.\s*the\s+(?:judgment|decree)\s+is\s+reversed', 
         "Per Curiam: Judgment REVERSED"),
        (r'per curiam[^.]*\.\s*the\s+(?:judgment|decree)\s+is\s+affirmed', 
         "Per Curiam: Judgment AFFIRMED"),
        
        # Direct judgment statements
        (r'(?:the\s+)?judgment\s+of\s+the\s+court\s+below\s+is\s+(reversed|affirmed|remanded)', 
         lambda m: f"Judgment {m.group(1).upper()}"),
        (r'(?:the\s+)?(?:judgment|decree)\s+is\s+(reversed|affirmed|remanded)\s+(?:with\s+costs)?',
         lambda m: f"Judgment {m.group(1).upper()}"),
        (r'(?:the\s+)?(?:judgment|decree)\s+is\s+(affirmed|reversed)', 
         lambda m: f"Judgment {m.group(1).upper()}"),
        
        # Compound judgments (partial affirmance/reversal)
        (r'judgment\s+as\s+to\s+the\s+debt[^.]*is\s+affirmed[^.]*as\s+to\s+the\s+damages\s+reversed',
         "Judgment AFFIRMED in part, REVERSED in part"),
        (r'judgment\s+(?:as\s+to\s+[^,]+)?\s*affirmed[^.]*reversed', 
         "Judgment AFFIRMED in part, REVERSED in part"),
    ]
    
    for pattern, result in priority_patterns:
        match = re.search(pattern, text_lower)
        if match:
            if callable(result):
                return result(match)
            return result

# ============================================================
# STEP 5: CASE TYPE CLASSIFICATION (IMPROVED)
# ============================================================

def extract_case_type(text: str) -> str:
    """Classify case type using weighted keyword matching"""
    t = text.lower()
    
    type_keywords = {
        "Criminal Law": {
            'keywords': ['indictment', 'larceny', 'murder', 'felony', 'misdemeanor', 'guilty', 
                        'theft', 'robbery', 'burglary', 'homicide', 'manslaughter', 'assault',
                        'battery', 'crime', 'criminal', 'state', 'prosecution', 'grand jury'],
            'weight': 3
        },
        "Civil Procedure": {
            'keywords': ['demurrer', 'plea', 'pleading', 'writ', 'jurisdiction', 'venue', 
                        'continuance', 'appeal', 'error', 'scire facias', 'replevin', 'assumpsit',
                        'verification', 'replication', 'joinder', 'nul tiel record'],
            'weight': 2
        },
        "Contract Law - Debt": {
            'keywords': ['debt', 'bond', 'promissory note', 'covenant', 'contract', 'payment', 
                        'interest', 'usury', 'consideration', 'obligation', 'surety', 'bail'],
            'weight': 2
        },
        "Property Law - Ejectment": {
            'keywords': ['ejectment', 'possession', 'land', 'real estate', 'title', 'deed', 
                        'conveyance', 'mortgage', 'foreclosure', 'equity of redemption', 'tenement'],
            'weight': 2
        },
        "Property Law - Execution Sale": {
            'keywords': ['execution', 'sheriff', 'sale', 'levy', 'fi. fa.', 'ca. sa.', 
                        'fieri facias', 'capias', 'venditioni exponas', 'goods'],
            'weight': 2
        },
        "Family Law - Dower": {
            'keywords': ['dower', 'dowable', 'widow', 'marriage', 'coverture', 'cestui que trust', 
                        'heir', 'inheritance', 'dower rights'],
            'weight': 2
        },
        "Torts": {
            'keywords': ['trespass', 'negligence', 'damages', 'assault', 'battery', 'slander', 
                        'libel', 'injury', 'conversion', 'nuisance'],
            'weight': 2
        },
    }
    
    scores = defaultdict(int)
    for case_type, info in type_keywords.items():
        for keyword in info['keywords']:
            if keyword in t:
                scores[case_type] += info['weight']
    
    if scores:
        return max(scores, key=scores.get)
    return "Unclassified"


