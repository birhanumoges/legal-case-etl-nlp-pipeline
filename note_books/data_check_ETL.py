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
# STEP 5: CASE TYPE CLASSIFICATION (FULLY FIXED)
# ============================================================

def extract_case_type(text: str) -> str:
    """
    Classify case type using weighted keyword matching - FULLY FIXED.
    PRIORITY 1: Check for slander/defamation FIRST (these are TORTS, not criminal)
    PRIORITY 2: Check for criminal cases
    PRIORITY 3: Check for other case types
    """
    t = text.lower()
    
    # ============================================================
    # PRIORITY 1: Defamation/Slander (MUST come before Criminal check)
    # ============================================================
    # If it's a slander case, it should NEVER be Criminal Law
    slander_keywords = [
        'slander', 'defamation', 'libel', 'scandalous words', 
        'action on the case', 'words spoken', 'actionable words',
        'for slanderous words', 'charged with adultery', 'fornication',
        'action on the case for slander', 'slanderous words'
    ]
    if any(kw in t for kw in slander_keywords):
        return "Torts - Defamation"
    
    # ============================================================
    # PRIORITY 2: Criminal Law
    # ============================================================
    criminal_keywords = [
        'indictment', 'larceny', 'murder', 'felony', 'misdemeanor', 'guilty', 
        'theft', 'robbery', 'burglary', 'homicide', 'manslaughter', 'assault',
        'battery', 'crime', 'criminal', 'state', 'prosecution', 'grand jury',
        'counterfeiting', 'perjury', 'quo warranto', 'contra formam statuti'
    ]
    
    criminal_score = 0
    for keyword in criminal_keywords:
        if keyword in t:
            criminal_score += 3
    
    # ============================================================
    # PRIORITY 3: Civil Procedure
    # ============================================================
    civil_keywords = [
        'demurrer', 'plea', 'pleading', 'writ', 'jurisdiction', 'venue', 
        'continuance', 'appeal', 'error', 'scire facias', 'replevin', 'assumpsit',
        'verification', 'replication', 'joinder', 'nul tiel record', 'habeas corpus',
        'certiorari', 'mandamus', 'injunction', 'bill of exceptions'
    ]
    
    civil_score = 0
    for keyword in civil_keywords:
        if keyword in t:
            civil_score += 2
    
    # ============================================================
    # PRIORITY 4: Contract Law
    # ============================================================
    contract_keywords = [
        'debt', 'bond', 'promissory note', 'covenant', 'contract', 'payment', 
        'interest', 'usury', 'consideration', 'obligation', 'surety', 'bail',
        'negotiable instrument', 'specialty', 'sealed note'
    ]
    
    contract_score = 0
    for keyword in contract_keywords:
        if keyword in t:
            contract_score += 2
    
    # ============================================================
    # PRIORITY 5: Property - Ejectment
    # ============================================================
    property_eject_keywords = [
        'ejectment', 'possession', 'land', 'real estate', 'title', 'deed', 
        'conveyance', 'mortgage', 'foreclosure', 'equity of redemption', 'tenement',
        'adverse possession', 'disseisin', 'lease', 'demise'
    ]
    
    property_eject_score = 0
    for keyword in property_eject_keywords:
        if keyword in t:
            property_eject_score += 2
    
    # ============================================================
    # PRIORITY 6: Property - Execution Sale
    # ============================================================
    property_exec_keywords = [
        'execution', 'sheriff', 'sale', 'levy', 'fi. fa.', 'ca. sa.', 
        'fieri facias', 'capias', 'venditioni exponas', 'goods',
        'replevin bond', 'distress'
    ]
    
    property_exec_score = 0
    for keyword in property_exec_keywords:
        if keyword in t:
            property_exec_score += 2
    
    # ============================================================
    # PRIORITY 7: Torts (non-defamation)
    # ============================================================
    torts_keywords = [
        'trespass', 'negligence', 'damages', 'injury', 'conversion', 'nuisance',
        'trover', 'case', 'mesne profits'
    ]
    
    torts_score = 0
    for keyword in torts_keywords:
        if keyword in t:
            torts_score += 2
    
    # ============================================================
    # DECISION LOGIC - Highest score wins
    # ============================================================
    scores = {
        "Criminal Law": criminal_score,
        "Civil Procedure": civil_score,
        "Contract Law - Debt": contract_score,
        "Property Law - Ejectment": property_eject_score,
        "Property Law - Execution Sale": property_exec_score,
        "Torts": torts_score
    }
    
    # Find max score
    max_score = max(scores.values())
    
    if max_score > 0:
        for case_type, score in scores.items():
            if score == max_score:
                return case_type
    
    return "Unclassified"


# ============================================================
# STEP 6: SUB-TYPE EXTRACTION (FULLY FIXED)
# ============================================================

def extract_sub_type(text: str, case_type: str) -> str:
    """
    Extract fine-grained sub-type - FULLY FIXED.
    Uses ordered keyword-label pairs per case type.
    Prevents "General" from firing too easily - checks all specific patterns first.
    """
    t = text.lower()
    
    # ============================================================
    # TORTS - DEFAMATION SUB-TYPES (FIXES CASE 0086)
    # ============================================================
    if case_type == "Torts - Defamation":
        if any(kw in t for kw in ['slander', 'words spoken', 'scandalous words', 'slanderous']):
            return "Slander"
        if 'libel' in t:
            return "Libel"
        return "Defamation"
    
    # ============================================================
    # CRIMINAL LAW SUB-TYPES (ORDERED - MOST SPECIFIC FIRST)
    # ============================================================
    if case_type == "Criminal Law":
        # Check for Larceny (most specific first)
        if any(kw in t for kw in ['larceny', 'grand larceny', 'petit larceny']):
            return "Larceny"
        if any(kw in t for kw in ['steal', 'stolen goods', 'receiving stolen', 'stole', 'theft']):
            return "Larceny"
        
        # Check for Homicide
        if any(kw in t for kw in ['murder', 'homicide', 'manslaughter', 'slay', 'mortal wound']):
            return "Homicide"
        if 'kill' in t and ('person' in t or 'man' in t or 'woman' in t):
            return "Homicide"
        
        # Check for Assault & Battery
        if 'assault and battery' in t or 'assault & battery' in t:
            return "Assault & Battery"
        if 'assault' in t and 'battery' in t:
            return "Assault & Battery"
        if any(kw in t for kw in ['assault', 'battery']):
            return "Assault & Battery"
        
        # Check for Burglary/Robbery
        if any(kw in t for kw in ['burglary', 'robbery', 'housebreaking', 'breaking and entering']):
            return "Burglary/Robbery"
        
        # Check for Counterfeiting
        if any(kw in t for kw in ['counterfeiting', 'forgery', 'forged', 'counterfeit']):
            return "Counterfeiting"
        
        # Check for Perjury
        if any(kw in t for kw in ['perjury', 'false oath', 'forsworn', 'swore falsely']):
            return "Perjury"
        
        # Check for Affray
        if 'affray' in t:
            return "Affray"
        
        # Check for Riot
        if 'riot' in t:
            return "Riot"
        
        # Check for Indictment (general criminal proceeding)
        if 'indictment' in t and 'found' in t:
            return "Indictment"
        
        # Only now fall back to General Criminal
        return "General Criminal"
    
    # ============================================================
    # CIVIL PROCEDURE SUB-TYPES
    # ============================================================
    if case_type == "Civil Procedure":
        # Demurrer
        if any(kw in t for kw in ['demurrer', 'demur', 'demurred']):
            return "Demurrer"
        
        # Appeal / Error
        if any(kw in t for kw in ['appeal', 'writ of error', 'error', 'appellate']):
            return "Appeal"
        
        # Scire Facias
        if 'scire facias' in t:
            return "Scire Facias"
        
        # Attachment
        if 'attachment' in t and ('foreign' in t or 'domestic' in t or 'property' in t):
            return "Attachment"
        
        # Habeas Corpus
        if 'habeas corpus' in t:
            return "Habeas Corpus"
        
        # Default Judgment
        if any(kw in t for kw in ['default', 'defaulted', 'nil dicit']):
            return "Default Judgment"
        
        # Pleading
        if any(kw in t for kw in ['plea', 'pleading', 'replication', 'rejoinder', 'surrejoinder']):
            return "Pleading"
        
        # Writ
        if any(kw in t for kw in ['writ', 'certiorari', 'mandamus', 'prohibition']):
            return "Writ"
        
        # Continuance
        if any(kw in t for kw in ['continuance', 'adjournment', 'postponement']):
            return "Continuance"
        
        return "General Civil Procedure"
    
    # ============================================================
    # CONTRACT LAW SUB-TYPES
    # ============================================================
    if case_type == "Contract Law - Debt":
        # Promissory Note
        if any(kw in t for kw in ['promissory note', 'note payable', 'negotiable instrument']):
            return "Promissory Note"
        if 'note' in t and ('promise' in t or 'pay' in t):
            return "Promissory Note"
        
        # Bond
        if any(kw in t for kw in ['bond', 'obligation', 'writing obligatory', 'specialty', 'sealed']):
            return "Bond"
        
        # Debt Collection
        if any(kw in t for kw in ['debt', 'indebtedness', 'recovery', 'collect']):
            return "Debt Collection"
        
        # Usury
        if any(kw in t for kw in ['usury', 'usurious', 'interest']):
            return "Usury"
        
        # Breach of Contract
        if any(kw in t for kw in ['breach', 'non-performance', 'failed to perform']):
            return "Breach of Contract"
        
        return "General Contract"
    
    # ============================================================
    # PROPERTY LAW - EJECTMENT SUB-TYPES
    # ============================================================
    if case_type == "Property Law - Ejectment":
        # Ejectment
        if any(kw in t for kw in ['ejectment', 'recovery of land', 'possession of land']):
            return "Ejectment"
        
        # Mortgage Foreclosure
        if any(kw in t for kw in ['foreclosure', 'mortgage', 'equity of redemption', 'scire facias']):
            return "Mortgage Foreclosure"
        
        # Title Dispute
        if any(kw in t for kw in ['title', 'quiet title', 'cloud on title', 'legal title', 'equitable title']):
            return "Title Dispute"
        
        # Adverse Possession
        if any(kw in t for kw in ['adverse possession', 'disseisin', 'peaceable possession']):
            return "Adverse Possession"
        
        return "General Property"
    
    # ============================================================
    # PROPERTY LAW - EXECUTION SALE SUB-TYPES
    # ============================================================
    if case_type == "Property Law - Execution Sale":
        # Sheriff Sale
        if any(kw in t for kw in ['sheriff sale', 'execution sale', 'sheriff\'s deed']):
            return "Sheriff Sale"
        
        # Fieri Facias
        if any(kw in t for kw in ['fi. fa.', 'fieri facias']):
            return "Fieri Facias"
        
        # Capias
        if any(kw in t for kw in ['ca. sa.', 'capias', 'capias ad satisfaciendum']):
            return "Capias"
        
        return "General Execution"
    
    # ============================================================
    # FAMILY LAW - DOWER SUB-TYPES
    # ============================================================
    if case_type == "Family Law - Dower":
        if any(kw in t for kw in ['dower', 'dowable', 'dower rights']):
            return "Dower Rights"
        
        if any(kw in t for kw in ['heir', 'inheritance', 'devise', 'legatee', 'will']):
            return "Inheritance"
        
        return "General Family Law"
    
    # ============================================================
    # TORTS (NON-DEFAMATION) SUB-TYPES
    # ============================================================
    if case_type == "Torts":
        if 'trespass' in t:
            return "Trespass"
        
        if any(kw in t for kw in ['assault', 'battery']):
            return "Assault & Battery"
        
        if 'negligence' in t:
            return "Negligence"
        
        if 'conversion' in t or 'trover' in t:
            return "Conversion"
        
        return "General Torts"
    
    # ============================================================
    # FALLBACK FOR UNCLASSIFIED - Check common patterns
    # ============================================================
    common_patterns = [
        ('larceny', 'Larceny'),
        ('steal', 'Larceny'),
        ('theft', 'Larceny'),
        ('stolen', 'Larceny'),
        ('murder', 'Homicide'),
        ('homicide', 'Homicide'),
        ('kill', 'Homicide'),
        ('assault', 'Assault & Battery'),
        ('battery', 'Assault & Battery'),
        ('appeal', 'Appeal'),
        ('writ of error', 'Appeal'),
        ('demurrer', 'Demurrer'),
        ('ejectment', 'Ejectment'),
        ('foreclosure', 'Mortgage Foreclosure'),
        ('mortgage', 'Mortgage Foreclosure'),
        ('slander', 'Slander'),
        ('defamation', 'Defamation'),
        ('bond', 'Bond'),
        ('promissory note', 'Promissory Note'),
    ]
    
    for keyword, sub_type in common_patterns:
        if keyword in t:
            return sub_type
    
    return "Unclassified"

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

# ============================================================
# STEP 9: SINGLE CASE PROCESSING
# ============================================================

def process_single_case(case_info: Dict) -> Optional[Dict]:
    try:
        case_text, text_length = extract_text_from_html(case_info['html_file'])
        if not case_text:
            return None
        
        json_metadata = extract_metadata_from_json(case_info['json_file'])
        
        verdict = extract_verdict(case_text)
        case_type = extract_case_type(case_text)
        sub_type = extract_sub_type(case_text, case_type)  # FIXED: Now uses improved function
        citations = extract_citations(case_text, json_metadata)
        year = extract_year(case_text, json_metadata)
        court = extract_court(json_metadata, case_text)
        case_name = extract_case_name(json_metadata, case_text, case_info['case_name'])
        
        return {
            'Case_ID': case_info['case_id'],
            'Case_Name': case_info['case_name'],
            #'Case_Name': case_name,
            'Source_Folder': case_info['source'],
            'Year': year,
            'Court': court,
            'Case_Text': case_text[:10000],
            'Case_Text_Full_Length': text_length,
            'Verdict': verdict,
            'Case_Type': case_type,
            'Sub_Type': sub_type,
            'Num_Citations': len(citations),
            'Legal_Citations': '; '.join(citations[:20]),
            'Has_Metadata': case_info['metadata'] is not None,
            'HTML_File': str(case_info['html_file']),
            'JSON_File': str(case_info['json_file']),
            'Decision_Date': json_metadata.get('decision_date', ''),
            'Docket_Number': json_metadata.get('docket_number', ''),
            'First_Page': json_metadata.get('first_page', ''),
            'Last_Page': json_metadata.get('last_page', ''),
        }
    except Exception as e:
        logger.error(f"Error processing {case_info['case_name']}: {e}")
        return None

# ============================================================
# STEP 10: SUMMARY REPORT
# ============================================================

def generate_summary_report(df: pd.DataFrame):
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("LEGAL NLP ETL PIPELINE - SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Total Cases: {len(df)}")
    
    if 'Source_Folder' in df.columns:
        report_lines.append("\n📁 SOURCE DISTRIBUTION:")
        for source, count in df['Source_Folder'].value_counts().items():
            report_lines.append(f"  - {source}: {count} cases")
    
    report_lines.append("\n📚 CASE TYPE DISTRIBUTION:")
    for ct, count in df['Case_Type'].value_counts().head(10).items():
        report_lines.append(f"  - {ct}: {count} ({count/len(df)*100:.1f}%)")
    
    report_lines.append("\n🔍 SUB-TYPE DISTRIBUTION (FIXED - Now includes Larceny, etc.):")
    for st, count in df['Sub_Type'].value_counts().head(15).items():
        report_lines.append(f"  - {st}: {count} ({count/len(df)*100:.1f}%)")
    
    report_lines.append("\n⚖️ VERDICT DISTRIBUTION:")
    for v, count in df['Verdict'].value_counts().head(10).items():
        report_lines.append(f"  - {v}: {count} ({count/len(df)*100:.1f}%)")
    
    if 'Year' in df.columns:
        years = pd.to_numeric(df['Year'], errors='coerce')
        report_lines.append(f"\n📅 YEAR RANGE: {years.min():.0f} - {years.max():.0f}")
    
    if 'Num_Citations' in df.columns:
        report_lines.append(f"\n📊 CITATIONS: Total {df['Num_Citations'].sum()}, Avg {df['Num_Citations'].mean():.2f} per case")
    
    report_lines.append(f"\n✅ QUALITY METRICS:")
    report_lines.append(f"  - Known verdicts: {(df['Verdict'] != 'Verdict Unknown').sum()}/{len(df)} ({(df['Verdict'] != 'Verdict Unknown').sum()/len(df)*100:.1f}%)")
    report_lines.append(f"  - Classified cases: {(df['Case_Type'] != 'Unclassified').sum()}/{len(df)} ({(df['Case_Type'] != 'Unclassified').sum()/len(df)*100:.1f}%)")
    report_lines.append(f"  - Cases with specific sub-types: {(df['Sub_Type'] != 'General').sum()}/{len(df)} ({(df['Sub_Type'] != 'General').sum()/len(df)*100:.1f}%)")
    
    report_path = os.path.join(OUTPUT_DIR, 'etl_summary_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print('\n' + '\n'.join(report_lines))
    logger.info(f"Report saved: {report_path}")

# ============================================================
# STEP 11: MAIN PIPELINE
# ============================================================

def run_etl_pipeline():
    logger.info("=" * 70)
    logger.info("LEGAL NLP ETL PIPELINE - STARTING (FIXED VERSION)")
    logger.info("=" * 70)
    
    # Step 1: Discover files
    logger.info("\n📁 Step 1: Discovering and matching files...")
    structure = discover_and_match_files(ROOT_PATH)
    
    if not structure:
        logger.error("❌ No valid sources found!")
        logger.info("\nExpected structure: allcase/folder_name/html/*.html + json/*.json + metadata/all_metadata.json")
        return pd.DataFrame()
    
    total_cases = sum(s['total_cases'] for s in structure.values())
    logger.info(f"\n📊 Found {total_cases} total cases across {len(structure)} sources")
    
    # Step 2: Process cases
    logger.info("\n⚙️ Step 2: Processing cases...")
    all_results = []
    
    for source_name, source_info in structure.items():
        logger.info(f"\n📂 Source: {source_name} ({source_info['total_cases']} cases)")
        for idx, case_info in enumerate(source_info['cases'], 1):
            logger.info(f"  [{idx}/{source_info['total_cases']}] {case_info['case_name']}")
            result = process_single_case(case_info)
            if result:
                all_results.append(result)
    
    # Step 3: Create DataFrame
    logger.info("\n📊 Step 3: Creating DataFrame...")
    df = pd.DataFrame(all_results)
    
    if df.empty:
        logger.error("❌ No cases processed!")
        return df
    
    # Step 4: Validate columns
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        logger.warning(f"Missing columns: {missing}")
    else:
        logger.info("✅ All required columns present")
    
    # Step 5: Save outputs
    logger.info("\n💾 Step 5: Saving outputs...")
    
    csv_path = os.path.join(OUTPUT_DIR, 'legal_cases_complete.csv')
    df.to_csv(csv_path, index=False, encoding='utf-8')
    logger.info(f"  ✅ CSV: {csv_path}")
    
    json_path = os.path.join(OUTPUT_DIR, 'legal_cases_complete.json')
    df.to_json(json_path, orient='records', indent=2, force_ascii=False)
    logger.info(f"  ✅ JSON: {json_path}")
    
    try:
        parquet_path = os.path.join(OUTPUT_DIR, 'legal_cases_complete.parquet')
        df.to_parquet(parquet_path, index=False)
        logger.info(f"  ✅ Parquet: {parquet_path}")
    except:
        pass

