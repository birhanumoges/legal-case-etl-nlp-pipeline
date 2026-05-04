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

# ============================================================
# STEP 6: SUB-TYPE EXTRACTION (FIXED - NOW HANDLES CRIMINAL CASES)
# ============================================================

def extract_sub_type(text: str, case_type: str) -> str:
    """Extract fine-grained sub-type with improved criminal law detection"""
    t = text.lower()
    
    # Criminal Law sub-types (FIXED - Now properly detects Larceny, etc.)
    if case_type == "Criminal Law":
        if any(kw in t for kw in ['larceny', 'steal', 'stolen', 'theft', 'petit larceny', 'grand larceny']):
            return "Larceny"
        if any(kw in t for kw in ['murder', 'homicide', 'kill', 'manslaughter', 'slay']):
            return "Homicide"
        if any(kw in t for kw in ['assault', 'battery', 'assault and battery']):
            return "Assault & Battery"
        if any(kw in t for kw in ['burglary', 'robbery', 'break']):
            return "Burglary/Robbery"
        if any(kw in t for kw in ['indictment', 'grand jury']):
            return "Indictment"
        return "General Criminal"
    
    # Civil Procedure sub-types
    if case_type == "Civil Procedure":
        if any(kw in t for kw in ['demurrer', 'demur']):
            return "Demurrer"
        if any(kw in t for kw in ['appeal', 'writ of error', 'error']):
            return "Appeal"
        if any(kw in t for kw in ['default', 'defaulted', 'nil dicit']):
            return "Default Judgment"
        if any(kw in t for kw in ['plea', 'pleading', 'replication', 'rejoinder']):
            return "Pleading"
        if any(kw in t for kw in ['writ', 'scire facias', 'certiorari', 'mandamus']):
            return "Writ"
        if any(kw in t for kw in ['continuance', 'adjournment']):
            return "Continuance"
        if any(kw in t for kw in ['habeas corpus']):
            return "Habeas Corpus"
        return "General Civil Procedure"
    
    # Contract Law sub-types
    if case_type == "Contract Law - Debt":
        if any(kw in t for kw in ['promissory note', 'note', 'negotiable instrument']):
            return "Promissory Note"
        if any(kw in t for kw in ['bond', 'obligation', 'specialty']):
            return "Bond"
        if any(kw in t for kw in ['debt', 'indebtedness', 'recovery']):
            return "Debt Collection"
        if any(kw in t for kw in ['usury', 'usurious']):
            return "Usury"
        if any(kw in t for kw in ['breach', 'non-performance']):
            return "Breach of Contract"
        return "General Contract"
    
    # Property Law - Ejectment sub-types
    if case_type == "Property Law - Ejectment":
        if any(kw in t for kw in ['ejectment', 'recovery of land', 'possession']):
            return "Ejectment"
        if any(kw in t for kw in ['foreclosure', 'mortgage', 'equity of redemption']):
            return "Mortgage Foreclosure"
        if any(kw in t for kw in ['title', 'quiet title', 'cloud on title']):
            return "Title Dispute"
        if any(kw in t for kw in ['adverse possession', 'disseisin']):
            return "Adverse Possession"
        return "General Property"
    
    # Property Law - Execution Sale sub-types
    if case_type == "Property Law - Execution Sale":
        if any(kw in t for kw in ['sheriff sale', 'execution sale']):
            return "Sheriff Sale"
        if any(kw in t for kw in ['fi. fa.', 'fieri facias']):
            return "Fieri Facias"
        if any(kw in t for kw in ['ca. sa.', 'capias']):
            return "Capias"
        return "General Execution"
    
    # Family Law - Dower sub-types
    if case_type == "Family Law - Dower":
        if any(kw in t for kw in ['dower', 'dowable', 'dower rights']):
            return "Dower Rights"
        if any(kw in t for kw in ['heir', 'inheritance', 'devise', 'legatee']):
            return "Inheritance"
        return "General Family Law"
    
    # Torts sub-types
    if case_type == "Torts":
        if any(kw in t for kw in ['slander', 'defamation', 'libel']):
            return "Defamation"
        if any(kw in t for kw in ['trespass']):
            return "Trespass"
        if any(kw in t for kw in ['assault', 'battery']):
            return "Assault & Battery"
        if any(kw in t for kw in ['negligence']):
            return "Negligence"
        return "General Torts"
    
    # Fallback for Unclassified
    general = {
        "Foreclosure": ['foreclosure'],
        "Appeal": ['appeal', 'writ of error'],
        "Demurrer": ['demurrer', 'demur'],
        "Default": ['default', 'default judgment'],
        "Larceny": ['larceny', 'steal', 'theft'],
        "Homicide": ['murder', 'homicide', 'kill']
    }
    
    for sub_type, keywords in general.items():
        if any(kw in t for kw in keywords):
            return sub_type
    
    return "General"

# ============================================================
# STEP 7: CITATION EXTRACTION
# ============================================================

def extract_citations(text: str, json_metadata: Dict) -> List[str]:
    """Extract legal citations"""
    citations = set()
    
    patterns = [
        r'(\d+)\s+([A-Z][a-z]+\.?)\s+(\d+)',
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+v\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s+(\d+)\s+([A-Za-z\.]+)\s+(\d+)',
        r'(\d+)\s+([A-Z][a-z]+\.?\s+[A-Z]?\.?)\s+(\d+)',
        r'(\d+)\s+U\.S\.\s+(\d+)',
        r'(\d+)\s+Blackf\.\s+(\d+)',
        r'(\d+)\s+Binn\.\s+(\d+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            if len(match) == 3:
                citations.add(f"{match[0]} {match[1]} {match[2]}")
            elif len(match) == 5:
                citations.add(f"{match[0]} v. {match[1]}, {match[2]} {match[3]} {match[4]}")
            elif len(match) == 2:
                citations.add(f"{match[0]} U.S. {match[1]}")
    
    if json_metadata and 'citations' in json_metadata:
        for cite in json_metadata['citations']:
            if cite:
                citations.add(cite)
    
    return sorted(list(citations))


# ============================================================
# STEP 8: YEAR & COURT EXTRACTION
# ============================================================

def extract_year(text: str, json_metadata: Dict) -> str:
    if json_metadata and 'decision_date' in json_metadata:
        year_match = re.search(r'(\d{4})', str(json_metadata['decision_date']))
        if year_match:
            return year_match.group(1)
    year_match = re.search(r'\b(18|19|20)\d{2}\b', text)
    return year_match.group(0) if year_match else "Unknown"

def extract_court(json_metadata: Dict, text: str) -> str:
    if json_metadata and json_metadata.get('court_name'):
        return json_metadata['court_name']
    match = re.search(r'Supreme Court of ([A-Za-z]+)', text)
    if match:
        return f"Supreme Court of {match.group(1)}"
    return "Supreme Court of Indiana"

def extract_case_name(json_metadata: Dict, text: str, case_name_from_file: str) -> str:
    if json_metadata and json_metadata.get('case_name'):
        return json_metadata['case_name']
    match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?(?:\s+and\s+[A-Z][a-z]+)?)\s+v\.\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', text[:500])
    if match:
        return f"{match.group(1)} v. {match.group(2)}"
    return case_name_from_file


