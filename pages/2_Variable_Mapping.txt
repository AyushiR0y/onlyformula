import os
import re
import json
import tempfile
from typing import List, Dict, Tuple, Optional, Set
import streamlit as st
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from openai import AzureOpenAI
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from difflib import SequenceMatcher
import openpyxl
from pathlib import Path
import time
import io
from functools import lru_cache

@lru_cache(maxsize=1)
def get_all_master_variables_cached():
    """Cached version - only recalculates when session changes"""
    return get_all_master_variables()
load_dotenv()

# Configuration
ALLOWED_EXCEL_EXTENSIONS = {'xlsx', 'xls', 'csv'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB

# Azure OpenAI Configuration (for AI semantic matching only)
AZURE_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
DEPLOYMENT_NAME = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

if AZURE_API_KEY:
    try:
        client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            api_version=AZURE_API_VERSION
        )
        MOCK_MODE = False
    except Exception as e:
        st.error(f"❌ Failed to initialize Azure OpenAI client: {str(e)}")
        MOCK_MODE = True
        client = None
else:
    MOCK_MODE = True
    client = None

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.90,    # Must be very similar
    'medium': 0.80,  # Good similarity
    'low': 0.75       # Minimum similarity to accept a match (Raised from 0.40)
}
# --- INPUT VARIABLES DEFINITIONS ---
INPUT_VARIABLES = {
    'TERM_START_DATE': 'Date when the policy starts',
    'FUP_Date': 'First Unpaid Premium date',
    'ENTRY_AGE': 'Age of the policyholder at policy inception',
    'FULL_TERM_PREMIUM': 'Annual Premium amount',
    'BOOKING_FREQUENCY': 'Frequency of premium booking (monthly, quarterly, yearly)',
    'PREMIUM_TERM': 'Premium Paying Term - duration for paying premiums',
    'SUM_ASSURED': 'Sum Assured - guaranteed amount on maturity/death',
    'Income_Benefit_Amount': 'Amount of income benefit',
    'Income_Benefit_Frequency': 'Frequency of income benefit payout',
    'DATE_OF_SURRENDER': 'Date when policy is surrendered',
    'no_of_premium_paid': 'Years passed since date of commencement till FUP',
    'maturity_date': 'Date of commencement + (BENEFIT_TERM * 12 months)',
    'policy_year': 'Years passed + 1 between date of commencement and surrender date',
    'BENEFIT_TERM': 'The duration (in years) for which the policy benefits are payable',
    'GSV_FACTOR': 'Guaranteed Surrender Value Factor',
    'SSV1_FACTOR': 'Surrender Value Factor',
    'SSV3_FACTOR': 'Special Surrender Value Factor for paid-up income benefits',
    'SSV2_FACTOR': 'Special Surrender Value Factor for return of premium',
    'FUND_VALUE': 'The total value of the policy fund at surrender or maturity',
    'SYSTEM_PAID': 'Amount paid by system for surrender or maturity',
    'CAPITAL_UNITS_VALUE': 'Number of units in policy fund at surrender or maturity',
}

BASIC_DERIVED_FORMULAS = {
    'no_of_premium_paid': 'Calculate based on difference between TERM_START_DATE and FUP_Date',
    'policy_year': 'Calculate based on difference between TERM_START_DATE and DATE_OF_SURRENDER + 1',
    'maturity_date': 'TERM_START_DATE + (BENEFIT_TERM* 12) months',
    'CAPITAL_FUND_VALUE': 'Total policy fund value including bonuses',
    'FUND_FACTOR': 'Factor to compute fund value based on premiums and term'
}

DEFAULT_TARGET_OUTPUT_VARIABLES = [
    'TOTAL_PREMIUM_PAID', 'TEN_TIMES_AP', 'one_oh_five_percent_total_premium',
    'SUM_ASSURED_ON_DEATH', 'GSV', 'PAID_UP_SA',
    'PAID_UP_SA_ON_DEATH', 'PAID_UP_INCOME_INSTALLMENT',
    'SSV1', 'SSV2', 'SSV3', 'SSV', 'SURRENDER_PAID_AMOUNT',
]
def extract_variables_from_formulas(formulas: List[Dict]) -> Tuple[Set[str], Dict[str, str]]:
    """Extract all unique variables from formula expressions AND calculation steps
    Returns: (all_variables, derived_variables_mapping)
    """
    variables = set()
    derived_vars = {}  # Maps derived var to its formula
    
    # Pattern to match variable names
    var_pattern = r'\b([a-zA-Z][a-zA-Z0-9_]*)\b'
    
    for formula in formulas:
        # Extract from main formula expression
        expr = formula.get('formula_expression', '')
        matches = re.findall(var_pattern, expr)
        variables.update(matches)
        
        # Extract from calculation steps (intermediate variables created by AI)
        calc_steps = formula.get('calculation_steps', [])
        if isinstance(calc_steps, list):
            for step in calc_steps:
                if isinstance(step, dict):
                    step_text = step.get('step', '') + ' ' + step.get('formula', '')
                    step_formula = step.get('formula', '')
                    step_matches = re.findall(var_pattern, step_text)
                    variables.update(step_matches)
                    
                    # Try to identify derived variable definitions (e.g., "variable_name = expression")
                    derived_match = re.match(r'([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*(.+)', step_formula)
                    if derived_match:
                        var_name = derived_match.group(1)
                        var_formula = derived_match.group(2)
                        derived_vars[var_name] = var_formula
                        
                elif isinstance(step, str):
                    step_matches = re.findall(var_pattern, step)
                    variables.update(step_matches)
                    
                    # Try to identify derived variable definitions
                    derived_match = re.match(r'([a-zA-Z][a-zA-Z0-9_]*)\s*=\s*(.+)', step)
                    if derived_match:
                        var_name = derived_match.group(1)
                        var_formula = derived_match.group(2)
                        derived_vars[var_name] = var_formula
    
    # Expanded filter list
    operators = {
        'MAX', 'MIN', 'SUM', 'AVG', 'AVERAGE', 'IF', 'THEN', 'ELSE', 'AND', 'OR', 'NOT',
        'ROUND', 'CEILING', 'FLOOR', 'ABS', 'POWER', 'SQRT', 'MOD', 'INT',
        'COUNT', 'VLOOKUP', 'INDEX', 'MATCH', 'ISERROR', 'ISBLANK',
        'TRUE', 'FALSE', 'NA', 'PI', 'EXP', 'LN', 'LOG', 'LOG10',
        'SIN', 'COS', 'TAN', 'ASIN', 'ACOS', 'ATAN',
        'CONCATENATE', 'LEFT', 'RIGHT', 'MID', 'LEN', 'TRIM',
        'UPPER', 'LOWER', 'PROPER', 'SUBSTITUTE', 'REPLACE',
        'TODAY', 'NOW', 'YEAR', 'MONTH', 'DAY', 'HOUR', 'MINUTE', 'SECOND',
        'DATE', 'TIME', 'DATEVALUE', 'TIMEVALUE',
        'COUNTIF', 'SUMIF', 'AVERAGEIF', 'MAXIFS', 'MINIFS',
        'LOOKUP', 'HLOOKUP', 'CHOOSE', 'OFFSET',
        'for', 'while', 'do', 'break', 'continue', 'return',
        'def', 'class', 'import', 'from', 'as', 'with',
        'try', 'except', 'finally', 'raise', 'assert',
        'in', 'is', 'None', 'True', 'False',
        'div', 'mod', 'abs', 'max', 'min', 'sum', 'avg',
        'Step', 'Calculate', 'Formula', 'Result', 'Value'
    }
    
    operators_normalized = {op.upper() for op in operators} | {op.lower() for op in operators}
    
    # Filter out operators and purely numeric strings
    variables = {v for v in variables if v not in operators_normalized and not v.isdigit()}
    
    return variables, derived_vars

def load_excel_file(file_bytes, file_extension: str) -> Tuple[pd.DataFrame, List[str]]:
    """Load Excel file and extract headers"""
    try:
        if file_extension == '.csv':
            # Use io.BytesIO instead of pd.io.common.BytesIO
            df = pd.read_csv(io.BytesIO(file_bytes))
        else:
            df = pd.read_excel(io.BytesIO(file_bytes))
        
        headers = df.columns.tolist()
        return df, headers
    
    except Exception as e:
        st.error(f"Error loading Excel file: {e}")
        return None, []
def get_all_master_variables():
    """Aggregates variables from Input, Formula, Derived, and Custom sources"""
    all_vars = set()
    
    # 1. Static Input Variables
    all_vars.update(INPUT_VARIABLES.keys())
    
    # 2. Basic Derived Formulas (FIX: Add this block)
    # We add the keys (the variable names) from the dictionary
    all_vars.update(BASIC_DERIVED_FORMULAS.keys())
    
    # 3. Default Target Output Variables (FIX: Add this block)
    all_vars.update(DEFAULT_TARGET_OUTPUT_VARIABLES)
    
    # 4. Extracted from Main Formulas
    if 'formulas' in st.session_state and st.session_state.formulas:
        formula_vars, derived_defs = extract_variables_from_formulas(st.session_state.formulas)
        all_vars.update(formula_vars)
        all_vars.update(derived_defs.keys())
    
    # 5. Extracted from Custom Formulas
    if 'custom_formulas' in st.session_state and st.session_state.custom_formulas:
        cf_vars, cf_derived = extract_variables_from_formulas(st.session_state.custom_formulas)
        all_vars.update(cf_vars)
        all_vars.update(cf_derived.keys())
            
    return sorted(list(all_vars))

@dataclass
class VariableMapping:
    variable_name: str
    mapped_header: str
    confidence_score: float
    matching_method: str
    is_verified: bool = False
    
    def to_dict(self):
        return asdict(self)

class VariableHeaderMatcher:
    """Matches formula variables to Excel headers using multiple strategies"""
    
    def __init__(self):
        self.mappings: Dict[str, VariableMapping] = {}
        
    def normalize_text(self, text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def expand_abbreviations(self, text: str) -> str:
        """Expand common insurance/financial abbreviations"""
        abbreviations = {
            'sa': 'sum assured',
            'dob': 'date of birth',
            'fup': 'first unpaid premium',
            'gsv': 'guaranteed surrender value',
            'ssv': 'special surrender value',
            'ssv1': 'special surrender value 1',
            'ssv2': 'special surrender value 2',
            'ssv3': 'special surrender value 3',
            'rop': 'return of premium',
            'ap': 'annual premium',
            'pp': 'premium paid',
            'amt': 'amount',
            'calc': 'calculation',
            'freq': 'frequency',
            'mat': 'maturity',
            'ben': 'benefit',
            'dt': 'date',
            'no': 'number',
            'yr': 'year',
            'yrs': 'years',
            'pct': 'percent',
            'val': 'value',
            'pd': 'paid',
            'term': 'term',
            'pol': 'policy',
            'prem': 'premium',
            'tot': 'total'
        }
        
        text_lower = text.lower()
        for abbr, full in abbreviations.items():
            text_lower = re.sub(r'\b' + abbr + r'\b', full, text_lower)
        
        return text_lower
    
    def lexical_similarity(self, var: str, header: str) -> float:
        """Calculate lexical similarity using multiple methods"""
        var_norm = self.normalize_text(var)
        header_norm = self.normalize_text(header)
        
        # Expand abbreviations for better matching
        var_expanded = self.expand_abbreviations(var_norm)
        header_expanded = self.expand_abbreviations(header_norm)
        
        # Exact match
        if var_expanded == header_expanded:
            return 1.0
        
        # Substring match
        if var_expanded in header_expanded or header_expanded in var_expanded:
            return 0.95
        
        # Token-based matching
        var_tokens = set(var_expanded.split())
        header_tokens = set(header_expanded.split())
        
        if var_tokens and header_tokens:
            intersection = len(var_tokens & header_tokens)
            union = len(var_tokens | header_tokens)
            jaccard = intersection / union if union > 0 else 0
            
            # If all variable tokens are in header, give high score
            if var_tokens.issubset(header_tokens):
                return 0.90
            
            return jaccard * 0.85
        
        return 0.0
    
    def fuzzy_similarity(self, var: str, header: str) -> float:
        """Calculate fuzzy string similarity with improved length penalty"""
        var_norm = self.normalize_text(var)
        header_norm = self.normalize_text(header)
        
        # Expand abbreviations
        var_expanded = self.expand_abbreviations(var_norm)
        header_expanded = self.expand_abbreviations(header_norm)
        
        # rapidfuzz returns 0-100, so divide by 100
        ratio = fuzz.ratio(var_expanded, header_expanded) / 100.0
        partial_ratio = fuzz.partial_ratio(var_expanded, header_expanded) / 100.0
        token_sort_ratio = fuzz.token_sort_ratio(var_expanded, header_expanded) / 100.0
        token_set_ratio = fuzz.token_set_ratio(var_expanded, header_expanded) / 100.0
        
        # Weighted average favoring token-based matching
        score = (ratio * 0.2 + partial_ratio * 0.2 + token_sort_ratio * 0.3 + token_set_ratio * 0.3)
        
        # --- IMPROVED LENGTH PENALTY ---
        # Penalize matches where one string is much shorter than the other
        # This prevents single letters like "N" from matching everything
        
        len_var = len(var_expanded)
        len_header = len(header_expanded)
        len_diff = abs(len_var - len_header)
        
        # Special case: Single character variables should ONLY match single character headers
        if len_var == 1 or len_header == 1:
            if len_var != len_header:
                score = score * 0.01  # 99% penalty
            return score
        
        # Very short variables (2-3 chars) need exact or very close matches
        if len_var <= 3 or len_header <= 3:
            if len_diff > 2:
                score = score * 0.1  # 90% penalty
            return score
        
        # General length difference penalties
        if len_diff > 15:
            score = score * 0.3  # 70% penalty
        elif len_diff > 10:
            score = score * 0.5  # 50% penalty
        elif len_diff > 5:
            score = score * 0.8  # 20% penalty
            
        return score
    
   

    def semantic_similarity_ai_batch(self, headers: List[str], variables: List[str]) -> Dict[str, Tuple[str, float, str]]:
        """
        Single AI call to match ALL headers to variables.
        Much more efficient than batching!
        """
        if MOCK_MODE or not client:
            return {h: ("", 0.0, "AI unavailable") for h in headers}
        
        try:
            # Simple, efficient prompt
            prompt = f"""Match these Excel headers to variable names using syntactic and semantic similarity. Output ONLY a JSON object.

    HEADERS: {headers}

    VARIABLES: {variables}
    1. Headers such as Surrender Paid Amount should match SURRENDER_V
    Return a JSON object where each key is a header and value is an object with:
    - "variable": best matching variable name or null if no good match
    - "score": confidence 0.0-1.0 (1.0=exact, 0.9=substring, 0.7-0.8=semantic, 0.0=no match)
    - "reason": brief explanation
    

    Example output format:
    {{
    "TOTAL_PREMIUM": {{"variable": "TOTAL_PREMIUM", "score": 1.0, "reason": "Exact match"}},
    "FULL_TERM_PREMIUM": {{"variable": "TOTAL_PREMIUM", "score": 0.85, "reason": "Semantic match"}},
    "RandomCol": {{"variable": null, "score": 0.0, "reason": "No match"}}
    }}

    Output ONLY valid JSON, no other text."""

            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=3000,  # JSON is compact
                temperature=0.0
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = re.sub(r'^```json?\s*|\s*```', '', response_text, flags=re.MULTILINE).strip()
            
            # Parse JSON
            try:
                mappings_json = json.loads(response_text)
            except json.JSONDecodeError as e:
                st.error(f"AI returned invalid JSON: {e}")
                st.code(response_text[:500])
                return {h: ("", 0.0, "JSON parse error") for h in headers}
            
            # Convert to expected format
            results = {}
            for header in headers:
                if header in mappings_json:
                    mapping = mappings_json[header]
                    var = mapping.get('variable')
                    score = float(mapping.get('score', 0.0))
                    reason = mapping.get('reason', 'No reason')
                    
                    # Convert null to empty string
                    if var is None or var == 'null':
                        var = ""
                        score = 0.0
                    
                    results[header] = (var, score, reason)
                else:
                    results[header] = ("", 0.0, "Not in AI response")
            
            return results
            
        except Exception as e:
            st.error(f"AI matching error: {e}")
            return {h: ("", 0.0, f"Error: {str(e)[:50]}") for h in headers}

    def validate_and_correct_mappings(self, ai_results: Dict[str, Tuple[str, float, str]], available_variables: List[str]) -> Dict[str, Tuple[str, float, str]]:
        """
        Validates AI-generated mappings against business rules.
        Prevents incorrect mappings and corrects them according to policy.
        
        Returns corrected mappings with updated reasons if validation failed.
        
        Rules:
        - POLICY_REF: should NOT map to policy_year
        - POLICY_STATUS: no restrictions
        - DATE_OF_PAYMENT: should NOT map to DATE_OF_SURRENDER
        - INTERIM_BONUS: should NOT map to BENEFIT_TERM
        - ADV_PREMIUM: should NOT map to PREMIUM_TERM
        - ST_MVA_AMOUNT: should NOT map to SURRENDER_PAID_AMOUNT
        - DIS_ADV_PREMIUM, SUSP_COLLECT_AMT, ANNUITY_AMOUNT, WOO_PV_VALUE: only map if exact match exists
        - RIDER_TERMI_VALUE: only map if 'rider' is in the variable name
        - PV: prefer SURRENDER_PAID_AMOUNT (or any variant with 'paid')
        - SV_FACTOR: prefer SSV1_FACTOR
        """
        
        # Define what each variable/header should NOT map to: {var_name: [forbidden_targets]}
        forbidden_mappings = {
            'POLICY_REF': ['policy_year'],
            'DATE_OF_PAYMENT': ['DATE_OF_SURRENDER'],
            'INTERIM_BONUS': ['BENEFIT_TERM'],
            'ADV_PREMIUM': ['PREMIUM_TERM'],
            'ST_MVA_AMOUNT': ['SURRENDER_PAID_AMOUNT'],
        }
        
        # Variables that should ONLY map if exact match exists in available variables
        exact_match_required_vars = {'DIS_ADV_PREMIUM', 'SUSP_COLLECT_AMT', 'ANNUITY_AMOUNT', 'WOO_PV_VALUE'}
        
        # Preferred mappings: try to map these variables to their preferred targets
        preferred_mappings = {
            'PV': 'SURRENDER_PAID_AMOUNT',
            'SV_FACTOR': 'SSV1_FACTOR',
        }
        
        corrected_results = {}
        
        for header, (mapped_var, score, reason) in ai_results.items():
            corrected_var = mapped_var
            corrected_score = score
            corrected_reason = reason
            
            # Skip if nothing was mapped
            if not mapped_var:
                corrected_results[header] = (corrected_var, corrected_score, corrected_reason)
                continue
            
            # RULE 1: Check forbidden mappings
            # Check if the header name matches a variable with restrictions
            for restricted_var, forbidden_targets in forbidden_mappings.items():
                # Check if header matches the restricted variable (case-insensitive)
                if header.upper() == restricted_var.upper():
                    # Check if mapped_var is in the forbidden list
                    if any(mapped_var.upper() == ft.upper() for ft in forbidden_targets):
                        corrected_var = ""
                        corrected_score = -1.0  # Special marker: intentionally blocked
                        corrected_reason = f"BLOCKED: {header} cannot map to {mapped_var}"
                        break
            
            if not corrected_var:
                corrected_results[header] = (corrected_var, corrected_score, corrected_reason)
                continue
            
            # RULE 2: Exact match requirements
            if header.upper() in {v.upper() for v in exact_match_required_vars}:
                # Check if there's an exact match in available_variables
                has_exact_match = any(v.upper() == header.upper() for v in available_variables)
                if not has_exact_match:
                    # No exact match found, so don't map this variable
                    corrected_var = ""
                    corrected_score = -1.0  # Special marker: intentionally blocked
                    corrected_reason = f"BLOCKED: Exact match required for {header} - not found in available variables"
            
            if not corrected_var:
                corrected_results[header] = (corrected_var, corrected_score, corrected_reason)
                continue
            
            # RULE 3: RIDER_TERMI_VALUE special handling
            if header.upper() == 'RIDER_TERMI_VALUE':
                if 'rider' not in mapped_var.lower():
                    corrected_var = ""
                    corrected_score = -1.0  # Special marker: intentionally blocked
                    corrected_reason = f"BLOCKED: RIDER_TERMI_VALUE requires 'rider' in variable name"
            
            if not corrected_var:
                corrected_results[header] = (corrected_var, corrected_score, corrected_reason)
                continue
            
            # RULE 4: Preferred mappings (only if score is low or no match)
            for source_var, preferred_target in preferred_mappings.items():
                if header.upper() == source_var.upper():
                    # Check if preferred target exists in available variables
                    matching_targets = [v for v in available_variables 
                                       if v.upper() == preferred_target.upper() or preferred_target.lower() in v.lower()]
                    
                    if matching_targets:
                        # If score is low, prefer the target
                        if score < 0.85:
                            corrected_var = matching_targets[0]
                            corrected_score = 0.95
                            corrected_reason = f"Preferred mapping: {source_var} -> {matching_targets[0]}"
            
            # RULE 5: PV fallback to any PAID variant
            if header.upper() == 'PV':
                pref_target = preferred_mappings['PV']
                has_pref = any(v.upper() == pref_target.upper() for v in available_variables)
                
                if not has_pref and score < 0.85:
                    # Look for any variant with PAID
                    paid_variants = [v for v in available_variables if 'PAID' in v.upper()]
                    if paid_variants:
                        corrected_var = paid_variants[0]
                        corrected_score = 0.90
                        corrected_reason = f"PV mapped to {paid_variants[0]} (paid variant)"
            
            corrected_results[header] = (corrected_var, corrected_score, corrected_reason)
        
        return corrected_results


    # FIX 4: Add debug output to the main matching function
    def match_all_with_ai(self, headers: List[str], variables: List[str], use_ai: bool = True) -> Dict[str, VariableMapping]:
        """
        Optimized matching with fallback strategy and debugging.
        """
        mappings = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        if use_ai and not MOCK_MODE and client:
            status_text.text("AI semantic matching in progress...")
            progress_bar.progress(0.3)
            
            # Primary: AI batch matching
            ai_results = self.semantic_similarity_ai_batch(headers, variables)
            progress_bar.progress(0.5)
            
            # Validate and correct AI mappings against business rules
            status_text.text("Validating mappings against business rules...")
            ai_results = self.validate_and_correct_mappings(ai_results, variables)
            progress_bar.progress(0.6)
            
            # Process AI results with fallback
            for idx, header in enumerate(headers):
                ai_variable, ai_score, ai_reason = ai_results.get(header, ("", 0.0, "No AI response"))
                
                # Check if this mapping was intentionally blocked by validation (score = -1.0)
                if ai_score == -1.0:
                    # Don't apply fallback - keep it blocked
                    mappings[header] = VariableMapping(
                        variable_name=header,
                        mapped_header="",
                        confidence_score=0.0,
                        matching_method=ai_reason,
                        is_verified=False
                    )
                # Lower threshold for AI since it's more intelligent
                elif ai_variable and ai_score >= 0.6:
                    mappings[header] = VariableMapping(
                        variable_name=header,
                        mapped_header=ai_variable,
                        confidence_score=ai_score,
                        matching_method=f"AI: {ai_reason[:50]}",
                        is_verified=ai_score >= 0.85
                    )
                else:
                    # Fallback to lexical/fuzzy for this specific header
                    best_score = 0.0
                    best_candidate = None
                    best_method = "no_match"
                    
                    for candidate in variables:
                        # Try lexical
                        lex_score = self.lexical_similarity(header, candidate)
                        if lex_score > best_score:
                            best_score = lex_score
                            best_candidate = candidate
                            best_method = "lexical"
                        
                        # Try fuzzy
                        fuzzy_score = self.fuzzy_similarity(header, candidate)
                        if fuzzy_score > best_score:
                            best_score = fuzzy_score
                            best_candidate = candidate
                            best_method = "fuzzy"
                    
                    if best_candidate and best_score >= 0.5:
                        mappings[header] = VariableMapping(
                            variable_name=header,
                            mapped_header=best_candidate,
                            confidence_score=best_score,
                            matching_method=f"{best_method} (AI fallback)",
                            is_verified=best_score >= 0.8
                        )
                    else:
                        # IMPORTANT: Still create a VariableMapping object with empty mapped_header
                        mappings[header] = VariableMapping(
                            variable_name=header,
                            mapped_header="",  # Empty but present
                            confidence_score=0.0,
                            matching_method=f"no_match (AI: {ai_reason[:30]})",
                            is_verified=False
                        )
                
                progress_bar.progress((idx + 1) / len(headers))
            
            progress_bar.progress(1.0)
            status_text.text("Matching complete!")
            time.sleep(0.5)
            
        else:
            # No AI - use lexical/fuzzy only
            status_text.text("Using lexical/fuzzy matching...")
            progress_bar.progress(0.5)
            
            for i, header in enumerate(headers):
                best_score = 0.0
                best_candidate = None
                best_method = "no_match"
                
                for candidate in variables:
                    combined, method = self.calculate_combined_score(candidate, header)
                    if combined > best_score:
                        best_score = combined
                        best_candidate = candidate
                        best_method = method
                
                if best_candidate and best_score >= 0.5:
                    mappings[header] = VariableMapping(
                        variable_name=header,
                        mapped_header=best_candidate,
                        confidence_score=best_score,
                        matching_method=best_method,
                        is_verified=best_score >= 0.8
                    )
                else:
                    # IMPORTANT: Still create mapping with empty value
                    mappings[header] = VariableMapping(
                        variable_name=header,
                        mapped_header="",
                        confidence_score=0.0,
                        matching_method="no_match",
                        is_verified=False
                    )
                
                progress_bar.progress((i + 1) / len(headers))
            
            status_text.text("Matching complete!")
        
        progress_bar.empty()
        status_text.empty()
        
        # Debug output
        st.write(f"🔍 Created {len(mappings)} mapping objects")
        non_empty = len([m for m in mappings.values() if m.mapped_header])
        st.write(f"✅ {non_empty} have actual mappings")
        
        return mappings

def enable_proceed_button():
    st.session_state.show_proceed_button = True
    
def apply_mappings_to_formulas(formulas: List[Dict], header_to_var_mapping: Dict[str, str]) -> List[Dict]:
    """
    Replace variables in formulas with mapped Excel Headers.
    header_to_var_mapping format: { "Excel_Header": "VariableName" }
    Formula: "VariableName * 2"
    Result: "[Excel_Header] * 2"
    """
    mapped_formulas = []
    
    # Create reverse mapping: variable -> header
    var_to_header = {var: header for header, var in header_to_var_mapping.items() if var}
    
    
    for formula in formulas:
        expr = formula.get('formula_expression', '')
        original_expr = expr
        
        # Replace each variable with its corresponding header
        for var_name, excel_header in var_to_header.items():
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(var_name) + r'\b'
            expr = re.sub(pattern, f'[{excel_header}]', expr)
    
        
        mapped_formulas.append({
            'formula_name': formula.get('formula_name', ''),
            'original_expression': formula.get('formula_expression', ''),
            'mapped_expression': expr
        })
    
    return mapped_formulas


def get_allowed_variables(header: str, all_variables: List[str]) -> List[str]:
    """
    Simple hardcoded function: returns list of variables allowed to map to this header.
    Blocks forbidden mappings, enforces exact match requirements, and RIDER_TERMI_VALUE rule.
    """
    # Forbidden mappings: {header: [blocked_variables]}
    forbidden = {
        'POLICY_REF': ['policy_year'],
        'POLICY_STATUS': ['policy_year'],  
        'DATE_OF_PAYMENT': ['DATE_OF_SURRENDER'],
        'SV_FACTOR': ['FUND_FACTOR'],  
        'PAIDUP_VALUE':['FUND_VALUE'],
        'INTERIM_BONUS': ['BENEFIT_TERM'],
        'ADV_PREMIUM': ['PREMIUM_TERM'],
        'ST_MVA_AMOUNT': ['SURRENDER_PAID_AMOUNT'],
    }
    
    # Variables that only map if exact match exists
    exact_match_required = {'DIS_ADV_PREMIUM', 'SUSP_COLLECT_AMT', 'ANNUITY_AMOUNT', 'WOP_PV_VALUE'}
    
    allowed = all_variables.copy()
    
    # RULE 1: Remove forbidden variables for this header
    if header in forbidden:
        blocked_vars = forbidden[header]
        allowed = [v for v in allowed if v.upper() not in [b.upper() for b in blocked_vars]]
    
    # RULE 2: If header requires exact match, only allow if exact match exists
    if header.upper() in {v.upper() for v in exact_match_required}:
        if header not in all_variables:
            return []  # No exact match exists - don't allow any mapping
    
    # RULE 3: RIDER_TERMI_VALUE only allows variables with 'rider' in name
    if header.upper() == 'RIDER_TERMI_VALUE':
        allowed = [v for v in allowed if 'rider' in v.lower()]
    
    return allowed


def load_css(file_name="style.css"):
    """
    Loads CSS file. Automatically handles cases where the script
    is inside a 'pages' subdirectory by looking one level up.
    """
    # 1. Get the directory where this script is currently running from
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Check if we are inside a 'pages' folder
    # If yes, we need to look one level up ('..') to find the CSS
    if os.path.basename(current_dir) == "pages":
        css_path = os.path.join(current_dir, "..", file_name)
    else:
        css_path = os.path.join(current_dir, file_name)
    
    # 3. Normalize the path (converts ".." to actual parent path)
    css_path = os.path.normpath(css_path)
    
    # 4. Load and inject the CSS
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        # If it still fails, show exactly where it looked
        st.error(f"⚠️ CSS file not found at: `{css_path}`. <br>Please ensure `style.css` is in the main project folder.", unsafe_allow_html=True)
def main():
    load_css()

    st.set_page_config(
        page_title="Variable Mapping",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    
    
    # Custom header
    st.markdown(
        """
        <div class="header-container">
            <div class="header-bar">
                <img src="https://raw.githubusercontent.com/AyushiR0y/streamlit_formulagen/main/assets/logo.png" style="height: 100px;">
                <div class="header-title" style="font-size: 2.5rem; font-weight: 750; color: #004DA8;">
                    Variable Mapping
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    
    # Initialize session state
    if 'formulas' not in st.session_state:
        st.session_state.formulas = []

    if not st.session_state.formulas:
        st.error("❌ No formulas found in session.")
        st.warning("⚠️ **Note:** Refreshing this page will clear your session. Use the navigation menu instead of browser refresh.")
        st.info("💡 Use the sidebar navigation to return to the main page and extract formulas again.")
        
        # Option to upload previously saved mappings
        st.markdown("---")
        st.subheader("📥 Restore Previous Session")
        
        col_restore1, col_restore2 = st.columns(2)
        
        with col_restore1:
            uploaded_json = st.file_uploader("Upload previously exported Formulas JSON", type=['json'], key="restore_json")
            if uploaded_json:
                try:
                    import_data = json.loads(uploaded_json.read())
                    if 'formulas' in import_data:
                        st.session_state.formulas = import_data['formulas']
                        st.success("✅ Formulas restored!")
                        st.rerun()
                    else:
                        st.error("❌ Invalid JSON format. Missing 'formulas' key.")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
        
        with col_restore2:
            st.markdown("**Or go back to extract formulas:**")
            st.info("Use the sidebar navigation to return to the main page.")
        
        return
    
    if 'excel_headers' not in st.session_state:
        st.session_state.excel_headers = []
    
    if 'header_to_var_mapping' not in st.session_state:
        st.session_state.header_to_var_mapping = {}
        
    if 'removed_headers' not in st.session_state:
        st.session_state.removed_headers = []
    
    if 'mapping_complete' not in st.session_state:
        st.session_state.mapping_complete = False
    
    if 'excel_df' not in st.session_state:
        st.session_state.excel_df = None
    
    if 'initial_mapping_done' not in st.session_state:
        st.session_state.initial_mapping_done = False
    
    if 'custom_formulas' not in st.session_state:
        st.session_state.custom_formulas = []
    
    if 'derived_variables' not in st.session_state:
        st.session_state.derived_variables = {}
        
    # State for user's variable selection
    if 'selected_variables_for_mapping' not in st.session_state:
        st.session_state.selected_variables_for_mapping = []
    
   

    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Excel Data File")
        st.markdown("Upload your Excel file containing the policy data that will be used for calculations.")
        
        uploaded_file = st.file_uploader(
            "Select Excel/CSV file",
            type=list(ALLOWED_EXCEL_EXTENSIONS),
            help=f"Accepts: {', '.join(ALLOWED_EXCEL_EXTENSIONS)}. Max file size: {MAX_FILE_SIZE / (1024*1024):.1f} MB",
            key="excel_uploader"
        )
        
        if uploaded_file is not None:
            if uploaded_file.size > MAX_FILE_SIZE:
                st.error(f"File size exceeds the limit. Please upload a file smaller than {MAX_FILE_SIZE / (1024*1024):.1f} MB.")
            else:
                st.info(f"**File Selected:** `{uploaded_file.name}` (`{uploaded_file.size / 1024:.1f} KB`)")
                
                # Load Excel file
                file_extension = Path(uploaded_file.name).suffix.lower()
                df, headers = load_excel_file(uploaded_file.read(), file_extension)
                
                if df is not None and headers:
                    st.session_state.excel_df = df
                    st.session_state.excel_headers = headers
                    
                    st.success(f"✅ Successfully loaded {len(df)} rows with {len(headers)} columns")
                    
                    
                    # --- Filter Variables Step ---
                    st.markdown("---")
                    st.subheader("🛠️ Filter Variables for Mapping")
                    st.markdown("Deselect variables that are **not** relevant to this specific Excel file.")
                    
                    # When getting master variables for mapping
                    all_master_vars = get_all_master_variables()

                    # Filter out single-letter variables unless explicitly selected
                    # These require manual mapping due to ambiguity
                    single_letter_vars = ['N', 'M', 'X', 'Y', 'Z']
                    default_vars = [v for v in all_master_vars if v not in single_letter_vars]
                    
                    # Default to selecting all variables if it's a fresh file upload
                    if st.session_state.selected_variables_for_mapping == [] or \
                       set(st.session_state.excel_headers) != set(st.session_state.get('last_uploaded_headers', [])):
                        st.session_state.selected_variables_for_mapping = all_master_vars
                        st.session_state.last_uploaded_headers = st.session_state.excel_headers
                        
                    valid_default_selection = [
                        v for v in st.session_state.selected_variables_for_mapping 
                        if v in all_master_vars
                    ]
                    
                    st.session_state.selected_variables_for_mapping = st.multiselect(
                        "Select Variables to Map",
                        options=all_master_vars,
                        default=valid_default_selection,
                        key="variable_filter_multiselect",
                        help="Variables NOT selected here will be ignored by the automatic mapper."
                    )
                    
                    # Button directly below
                    st.markdown('<br>', unsafe_allow_html=True)
                    if st.button("🔗 Start Automatic Mapping", type="primary", key="start_mapping_btn", disabled=st.session_state.initial_mapping_done):
                        # Determine if AI should be used
                        use_ai_mapping = not MOCK_MODE
                    
                        
                        with st.spinner("Analyzing headers and matching with variables..."):
                            # 1. CLEAR OLD MAPPINGS
                            st.session_state.header_to_var_mapping = {}
                            
                            active_variables = st.session_state.selected_variables_for_mapping
                            matcher = VariableHeaderMatcher()
                            
                            # 2. Map Headers -> Variables (with integrated AI)
                            mappings = matcher.match_all_with_ai(
                                headers=headers,
                                variables=active_variables,
                                use_ai=use_ai_mapping
                            )
                            
                            # 3. Update session state - THIS IS THE FIX
                            new_mapping = {}
                            for header, mapping_obj in mappings.items():
                                # Only store non-empty mappings
                                if mapping_obj.mapped_header:
                                    new_mapping[header] = mapping_obj.mapped_header
                                else:
                                    # Store empty string for unmapped headers
                                    new_mapping[header] = ""
                            
                            st.session_state.header_to_var_mapping = new_mapping
                            st.session_state.initial_mapping_done = True
                            
                            
                            
                            # Show matching statistics
                            total = len(mappings)
                            mapped = len([m for m in mappings.values() if m.mapped_header])
                            
                            

    
    with col2:
        st.subheader("📋 Available Variables")
        st.markdown("Variables available for mapping: **Input**, **Derived**, and **Extracted** from formulas.")
        
        # Get consolidated variable list
        all_variables = get_all_master_variables()
        
        if all_variables:
            # Categorize variables for display
            input_vars = set(INPUT_VARIABLES.keys())
            formula_vars, derived_defs = extract_variables_from_formulas(st.session_state.formulas)
            
            var_df_data = []
            for var in sorted(all_variables):
                v_type = "Input"
                if var in derived_defs:
                    v_type = "Derived"
                elif var not in input_vars:
                    v_type = "Extracted"
                var_df_data.append({'Variable Name': var, 'Type': v_type})
            
            var_df = pd.DataFrame(var_df_data)
            st.dataframe(var_df, use_container_width=True, hide_index=True)
            
            # Show derived variable formulas
            if derived_defs:
                with st.expander("📐 Derived Variable Definitions", expanded=False):
                    for var, formula in sorted(derived_defs.items()):
                        st.markdown(f"**`{var}`** = `{formula}`")
        else:
            st.info("No variables detected.")
       
        st.subheader("➕ Add Custom Formula")
        
        custom_name = st.text_input("Name", placeholder="Custom_Calc", key="cf_name")
        custom_expr = st.text_input("Expression", placeholder="var1 + var2 * 0.5", key="cf_expr")
        
        col_cf1, col_cf2 = st.columns([3, 1])
        with col_cf1:
            if st.button("Add", key="add_cf"):
                if custom_name and custom_expr:
                    st.session_state.custom_formulas.append({
                        'formula_name': custom_name,
                        'formula_expression': custom_expr
                    })
                    st.success(f"✅ Added: {custom_name}")
                    st.rerun()
        
        if st.session_state.custom_formulas:
            st.caption("**Custom Formulas:**")
            for idx, cf in enumerate(st.session_state.custom_formulas):
                col_a, col_b = st.columns([5, 1])
                with col_a:
                    st.caption(f"`{cf['formula_name']}`")
                with col_b:
                    if st.button("🗑️", key=f"del_cf_{idx}"):
                        st.session_state.custom_formulas.pop(idx)
                        st.rerun()
        # ------------------------------------------------
                
    
    # Variable Mapping Section
    if st.session_state.initial_mapping_done:
        st.markdown("---")
        st.subheader("🔗 Header to Variable Mappings")
        st.markdown("Review and edit the mappings. Rows represent **Excel Headers**. Map them to the **Variables** used in formulas.")
        
        # Debug: Show current mapping state
        with st.expander("🔍 Debug: Current Mapping State", expanded=False):
            st.json(st.session_state.header_to_var_mapping)
            st.write(f"Total mappings: {len(st.session_state.header_to_var_mapping)}")
            st.write(f"Non-empty mappings: {len([v for v in st.session_state.header_to_var_mapping.values() if v])}")
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["📊 Mappings Table", "📋 JSON View"])
        
        with tab1:
            st.markdown("#### Mapping Configuration")
            
            # Header row
            col_h1, col_h2, col_h3 = st.columns([3, 3, 1])
            with col_h1:
                st.markdown("**Excel Header**")
            with col_h2:
                st.markdown("**Mapped Variable**")
            with col_h3:
                st.markdown("**Actions**")
                        
            st.markdown('<hr style="margin: 0.5rem 0; border: 0; border-top: 2px solid #004DA8;">', unsafe_allow_html=True)
            
            # Get current variables for dropdown
            current_variables = get_all_master_variables()
            
            # Filter out removed headers
            active_headers = [h for h in st.session_state.excel_headers if h not in st.session_state.removed_headers]
            
            # Create a form to batch updates
            with st.form(key="mapping_form"):
                updated_mappings = {}
                
                for header in active_headers:
                    # Get current mapping for this header
                    current_var = st.session_state.header_to_var_mapping.get(header, "")
                    
                    col1, col2, col3 = st.columns([3, 3, 1])

                    with col1:
                        st.text_input("Header", value=header, key=f"h_txt_{header}", label_visibility="collapsed", disabled=True)
                    
                    with col2:
                        # Get allowed variables for this header based on business rules
                        allowed_vars = get_allowed_variables(header, current_variables)
                        dropdown_options = ["(None of the following)"] + allowed_vars
                        
                        # Determine index based on current mapping
                        if current_var and current_var in dropdown_options:
                            idx = dropdown_options.index(current_var)
                        else:
                            idx = 0  # Default to "(None of the following)"
                        
                        # Selectbox with unique key
                        new_var = st.selectbox(
                            "Variable",
                            options=dropdown_options,
                            index=idx,
                            key=f"var_select_{header}",
                            label_visibility="collapsed"
                        )
                        
                        # Store in temporary dict (will be applied on form submit)
                        final_var = "" if new_var == "(None of the following)" else new_var
                        updated_mappings[header] = final_var
                    
                    with col3:
                        # Remove checkbox instead of button (since we're in a form)
                        remove = st.checkbox("🗑️", key=f"remove_{header}", help="Remove this column")
                        if remove:
                            if header not in st.session_state.removed_headers:
                                st.session_state.removed_headers.append(header)
                                
                    st.markdown('<hr style="margin: 0.5rem 0; border: 0; border-top: 1px solid #e0e0e0;">', unsafe_allow_html=True)
                
                # Submit button for the form
                submitted = st.form_submit_button("💾 Save Changes", type="primary")
                
                if submitted:
                    # Apply all changes at once
                    for header, var in updated_mappings.items():
                        if header not in st.session_state.removed_headers:
                            st.session_state.header_to_var_mapping[header] = var
                    
                    st.success("✅ Changes saved!")
                    st.rerun()
        
        with tab2:
            st.markdown("#### Current Mappings (JSON)")
            
            # Filter out empty mappings and removed headers
            active_mapping = {
                h: v for h, v in st.session_state.header_to_var_mapping.items() 
                if v and h not in st.session_state.removed_headers
            }
            
            st.json(active_mapping)
            
            # Show statistics
            st.write(f"**Active mappings:** {len(active_mapping)}")
            st.write(f"**Removed headers:** {len(st.session_state.removed_headers)}")
            
            # Provide download for this JSON
            st.download_button(
                label="📥 Download Mapping JSON",
                data=json.dumps(active_mapping, indent=2),
                file_name="header_to_variable_mapping.json",
                mime="application/json"
            )
            
            # Show removed headers
            if st.session_state.removed_headers:
                st.markdown("**Removed Headers:**")
                st.write(st.session_state.removed_headers)
        
        col_stat1, col_stat2, col_stat3 = st.columns(3)
        
        total_headers = len(st.session_state.excel_headers)
        active_count = len([h for h in st.session_state.excel_headers if h not in st.session_state.removed_headers])
        mapped_count = len([v for v in st.session_state.header_to_var_mapping.values() if v])
        
        with col_stat1:
            st.metric("Total Headers", total_headers)
        with col_stat2:
            st.metric("Active Columns", active_count)
        with col_stat3:
            st.metric("Mapped Variables", mapped_count)
        

        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1]) # Changed to 3 columns for JSON and CSV
        
        with col_btn1:
            if st.button("✅ Confirm Mappings", type="primary", key="confirm_mappings_btn"):
                # Check for unmapped active headers
                unmapped = [h for h in active_headers if not st.session_state.header_to_var_mapping.get(h)]
                
                if unmapped:
                    st.warning(f"⚠️ {len(unmapped)} headers are unmapped: {', '.join(unmapped[:5])}{'...' if len(unmapped) > 5 else ''}")
                
                st.session_state.mapping_complete = True
                st.success("✅ Mappings confirmed!")
                st.rerun()
        
        with col_btn2:
            # Export JSON
            active_mapping = {
                h: v for h, v in st.session_state.header_to_var_mapping.items() 
                if v and h not in st.session_state.removed_headers
            }
            st.download_button(
                label="📥 Export JSON",
                data=json.dumps(active_mapping, indent=2),
                file_name="final_mappings.json",
                mime="application/json"
            )

        with col_btn3:
            # Export CSV
            df_mapping = pd.DataFrame(list(active_mapping.items()), columns=['Excel_Header', 'Mapped_Variable'])
            csv_data = df_mapping.to_csv(index=False)
            st.download_button(
                label="📥 Export CSV",
                data=csv_data,
                file_name="final_mappings.csv",
                mime="text/csv"
            )
    
    # Show mapped formulas
    if st.session_state.mapping_complete:
        st.markdown("---")
        st.subheader("📐 Formulas with Mapped Headers")
        
        # 1. Get the mapped formulas
        mapped_formulas = apply_mappings_to_formulas(
            st.session_state.formulas,
            st.session_state.header_to_var_mapping
        )
        
        # 2. Convert to DataFrame for the table
        df_formulas = pd.DataFrame(mapped_formulas)

        # 3. Display as an editable table
        edited_df = st.data_editor(
            df_formulas,
            column_config={
                "formula_name": st.column_config.TextColumn(
                    "Formula Name", 
                    disabled=True,  # Prevent editing the name
                    width="medium"
                ),
                "original_expression": st.column_config.TextColumn(
                    "Original Expression", 
                    disabled=True,  # Prevent editing the source
                    width="large"
                ),
                "mapped_expression": st.column_config.TextColumn(
                    "Mapped Expression", 
                    width="large",
                    help="Edit the formula here. Changes are saved automatically."
                ),
            },
            use_container_width=True,
            hide_index=True,
            num_rows="fixed"  # Prevents adding/deleting rows, only editing cells
        )
        
        # 4. Save the edited data back to session state (Optional)
        # This converts the DataFrame back to a list of dicts
        st.session_state.edited_formulas = edited_df.to_dict(orient="records")
            
        # Export mapped formulas
        st.markdown("---")
        st.subheader("📥 Export Mapped Formulas")
        st.markdown("Download the Mapped formulas json to proceed to calculations.")
        col_exp1, col_exp2, col_exp3 = st.columns([1, 1, 1])
        
        if "show_proceed_button" not in st.session_state:
            st.session_state.show_proceed_button = False

        with col_exp1:
            st.download_button(
                label="📥 Download Formulas (JSON)",
                data=json.dumps(mapped_formulas, indent=2),
                file_name="mapped_formulas.json",
                mime="application/json",
                # When this is clicked, it triggers 'enable_proceed_button'
                on_click=enable_proceed_button,
                key="download_json_btn"
            )
                
        with col_exp2:
            csv_formula_data = pd.DataFrame(mapped_formulas).to_csv(index=False)
            st.download_button(
                label="📥 Download Formulas (CSV)",
                data=csv_formula_data,
                file_name="mapped_formulas.csv",
                mime="text/csv"
            )
                
        with col_exp3:
            # 3. The button ONLY exists if 'show_proceed_button' is True
            if st.session_state.show_proceed_button:
                if st.button("➡️ Proceed to Calculations", type="primary", key="goto_calc"):
                    st.switch_page("pages/3_Calculation_Engine.py")
    
    # Footer
    st.markdown(
        """
        <div style="text-align: center; margin-top: 50px; color: #7f8c8d; font-size: 0.9em;">
            <p>Variable Mapping Module | Developed with Streamlit and AI</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()