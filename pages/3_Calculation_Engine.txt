import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import re
from pathlib import Path
from dataclasses import dataclass
import os
import math
from datetime import datetime, date
import json
from dateutil.relativedelta import relativedelta

# Load Common CSS
def load_css(file_name="style.css"):
    """Loads CSS file."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(current_dir) == "pages":
        css_path = os.path.join(current_dir, "..", file_name)
    else:
        css_path = os.path.join(current_dir, file_name)
    css_path = os.path.normpath(css_path)
    if os.path.exists(css_path):
        with open(css_path, 'r') as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def format_indian_number(num):
    """Format number in Indian numbering system (lakhs, crores)"""
    if pd.isna(num) or num is None:
        return "N/A"
    
    num = float(num)
    
    if num < 0:
        sign = "-"
        num = abs(num)
    else:
        sign = ""
    
    s = f"{num:.2f}"
    if '.' in s:
        integer_part, decimal_part = s.split('.')
    else:
        integer_part = s
        decimal_part = "00"
    
    # Indian formatting: last 3 digits, then groups of 2
    if len(integer_part) <= 3:
        formatted = integer_part
    else:
        last_three = integer_part[-3:]
        remaining = integer_part[:-3]
        
        # Group remaining digits in pairs from right to left
        pairs = []
        while len(remaining) > 0:
            if len(remaining) >= 2:
                pairs.insert(0, remaining[-2:])
                remaining = remaining[:-2]
            else:
                pairs.insert(0, remaining)
                remaining = ""
        
        formatted = ",".join(pairs) + "," + last_three
    
    return f"{sign}{formatted}.{decimal_part}"

# --- Data Classes ---
@dataclass
class CalculationResult:
    formula_name: str
    rows_calculated: int
    errors: List[str]
    success_rate: float

# --- Derived Formulas ---
BASIC_DERIVED_FORMULAS = {
    'no_of_premium_paid': {
        'description': 'Number of years of premiums paid (FUP_Date - TERM_START_DATE) / 12',
        'formula': 'MONTHS_BETWEEN(TERM_START_DATE, FUP_Date) / 12', 
        'variables': ['FUP_Date', 'TERM_START_DATE']
    },
    'policy_year': {
        'description': 'Policy year based on term start and surrender date',
        'formula': 'int(MONTHS_BETWEEN(TERM_START_DATE, DATE_OF_SURRENDER) / 12 + 1)',
        'variables': ['DATE_OF_SURRENDER', 'TERM_START_DATE']
    }
}

FORMULA_ALIASES = {
    # Multiple formula names that should write to/read from the same column
    'ROP_BENEFIT': 'TOTAL_PREMIUM_PAID',
    'ROP_Benefit': 'TOTAL_PREMIUM_PAID',
    'TOTAL_PREMIUMS_PAID': 'TOTAL_PREMIUM_PAID',
    'Income_Benefit_Amount': 'PAID_UP_INCOME_BENEFIT_AMOUNT',
    'PAID_UP_SA_ON_DEATH': 'PAID_UP_SA_ON_DEATH',
    'Present_Value_of_paid_up_sum_assured_on_death': 'PAID_UP_SA_ON_DEATH',
    'SURRENDER_PAID_AMOUNT': 'SURRENDER_PAID_AMOUNT',
    'PV': 'SURRENDER_PAID_AMOUNT',
    'PAID_UP_INCOME_INSTALLMENT': 'PAID_UP_INCOME_INSTALMENT'
}


def build_var_to_headers(header_to_var_mapping: Dict[str, str]) -> Dict[str, List[str]]:
    """Build reverse mapping: variable_name -> [list of headers]"""
    var_to_headers: Dict[str, List[str]] = {}
    for header, var_name in header_to_var_mapping.items():
        if not var_name:
            continue
        var_to_headers.setdefault(var_name, []).append(header)
    return var_to_headers


def get_first_mapped_value(row: pd.Series, headers: List[str]) -> Tuple[Any, Optional[str]]:
    """Get first non-NaN value from first mapped header that has data"""
    for header in headers:
        if header in row.index:
            val = row[header]
            if pd.notna(val):
                return val, header
    return None, None


def get_output_column_name(formula_name: str, formula_info: Dict, var_to_header_mapping: Dict[str, List[str]], header_to_var_mapping: Dict[str, str]) -> str:
    """
    Determine the actual column name to use for a formula.
    
    Logic:
    1. Check if formula has explicit output_column
    2. Search mappings for: column_header: variable_name where variable_name == formula_name
    3. Check aliases
    4. Default to formula_name
    
    Mappings structure: {"Column Header": "Variable_Name"}
    So if we have {"SURRENDER PAID AMOUNT": "SURRENDER_PAID_AMOUNT"},
    and formula_name is "SURRENDER_PAID_AMOUNT",
    we should return "SURRENDER PAID AMOUNT"
    """
    # PRIORITY 1: Check if formula has explicit output_column specified
    if 'output_column' in formula_info and formula_info['output_column']:
        return formula_info['output_column']
    
    # PRIORITY 2: Search header_to_var_mapping for column where value matches formula_name
    # header_to_var_mapping format: {"Column Header": "Variable_Name"}
    # We want to find the key (column header) where value == formula_name
    for column_header, variable_name in header_to_var_mapping.items():
        if variable_name == formula_name:
            # Found it! The output column is the key (column_header)
            return column_header
    
    # PRIORITY 3: Check if this formula has an alias
    if formula_name in FORMULA_ALIASES:
        aliased = FORMULA_ALIASES[formula_name]
        # Check if the aliased name maps to a column
        for column_header, variable_name in header_to_var_mapping.items():
            if variable_name == aliased:
                return column_header
        return aliased
    
    # PRIORITY 4: Default to formula name itself
    return formula_name

def get_derived_formulas() -> List[Dict]:
    """Convert derived formulas to standard formula format"""
    formulas = []
    for name, info in BASIC_DERIVED_FORMULAS.items():
        formulas.append({
            'formula_name': name,
            'formula_expression': info['formula'],
            'description': info['description'],
            'variables_used': ', '.join(info['variables']),
            'is_pre_mapped': False
        })
    return formulas

# --- Helper Functions ---

def safe_convert_to_number(value: Any) -> float:
    """Safely convert various types to float - FIXED VERSION"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    if isinstance(value, str) and (value == '' or value.strip() == ''):
        return 0.0
    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value)
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return float(value.year)
    if isinstance(value, str):
        try:
            cleaned = value.replace(',', '').replace('$', '').replace('%', '').strip()
            if cleaned:
                return float(cleaned)
            return 0.0
        except ValueError:
            try:
                parsed_date = pd.to_datetime(value)
                return float(parsed_date.year)
            except:
                return 0.0
    return 0.0


def months_between(date1, date2):
    """Calculate months between two dates (date2 - date1)"""
    try:
        if pd.isna(date1) or pd.isna(date2):
            return 0
        d1 = pd.to_datetime(date1)
        d2 = pd.to_datetime(date2)
        months = (d2.year - d1.year) * 12 + (d2.month - d1.month)
        return float(months)
    except:
        return 0

def add_months(date, months):
    """Add months to a date"""
    try:
        if pd.isna(date):
            return None
        from dateutil.relativedelta import relativedelta
        d = pd.to_datetime(date)
        result = d + relativedelta(months=int(months))
        return result
    except:
        return None

def safe_eval(expression: str, variables: Dict[str, Any]) -> Any:
    """Safely evaluate a mathematical expression - CORRECTED VERSION with = handling"""
    try:
        eval_expr = expression.strip()
        
        print(f"\n{'='*60}")
        print(f"🔍 DEBUGGING safe_eval")
        print(f"{'='*60}")
        print(f"Original expression: {expression}")
        print(f"Variables passed: {variables}")

        # ENHANCED: Remove assignment if present (handles any = that slipped through)
        if '=' in eval_expr and not any(op in eval_expr for op in ['==', '!=', '<=', '>=']):
            parts = eval_expr.split('=')
            if len(parts) >= 2:
                # Take everything after the FIRST = sign
                old_expr = eval_expr
                eval_expr = '='.join(parts[1:]).strip()
                print(f"⚙️ Stripped assignment: '{old_expr}' → '{eval_expr}'")

        # Process MONTHS_BETWEEN
        if 'MONTHS_BETWEEN' in eval_expr.upper():
            pattern = r'MONTHS_BETWEEN\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
            matches = list(re.finditer(pattern, eval_expr, re.IGNORECASE))
            for match in reversed(matches):
                var1, var2 = match.group(1).strip(), match.group(2).strip()
                val1 = variables.get(var1, var1)
                val2 = variables.get(var2, var2)
                result = months_between(val1, val2)
                eval_expr = eval_expr[:match.start()] + str(result) + eval_expr[match.end():]
        
        # Process ADD_MONTHS
        if 'ADD_MONTHS' in eval_expr.upper():
            pattern = r'ADD_MONTHS\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
            matches = list(re.finditer(pattern, eval_expr, re.IGNORECASE))
            for match in reversed(matches):
                var1, var2 = match.group(1).strip(), match.group(2).strip()
                val1 = variables.get(var1, var1)
                
                try:
                    var2_eval = var2
                    for var_name in sorted(variables.keys(), key=len, reverse=True):
                        if var_name in var2_eval:
                            var2_eval = var2_eval.replace(var_name, str(safe_convert_to_number(variables[var_name])))
                    val2_float = eval(var2_eval)
                except:
                    val2_float = safe_convert_to_number(var2)
                
                result = add_months(val1, val2_float)
                if result:
                    return result
                else:
                    eval_expr = eval_expr[:match.start()] + '0' + eval_expr[match.end():]
        
        # Process CURRENT_DATE
        if 'CURRENT_DATE' in eval_expr.upper():
            current_date = datetime.now()
            eval_expr = re.sub(r'\bCURRENT_DATE\b', f"'{current_date.strftime('%Y-%m-%d')}'", 
                              eval_expr, flags=re.IGNORECASE)
        
        # Map function names (case insensitive)
        func_mappings = {
            r'\bMAX\s*\(': 'max(',
            r'\bMIN\s*\(': 'min(',
            r'\bABS\s*\(': 'abs(',
            r'\bROUND\s*\(': 'round(',
            r'\bPOWER\s*\(': 'pow(',
            r'\bSQRT\s*\(': 'math.sqrt(',
            r'\bSUM\s*\(': 'sum(',
        }
        
        for pattern, replacement in func_mappings.items():
            eval_expr = re.sub(pattern, replacement, eval_expr, flags=re.IGNORECASE)
        
        print(f"After function mapping: {eval_expr}")
        
        # CRITICAL FIX: Replace variables in a single pass with unique placeholders
        # Sort by length (longest first) to prevent partial matches
        sorted_vars = sorted(variables.keys(), key=len, reverse=True)
        
        # Create unique placeholders that won't conflict
        placeholder_map = {}
        temp_expr = eval_expr
        
        for idx, var_name in enumerate(sorted_vars):
            value = variables[var_name]
            
            # CRITICAL: Don't convert to number for bracketed variables - keep original value
            if var_name.startswith('[') and var_name.endswith(']'):
                # For bracketed variables, use the original value directly
                numeric_value = value
                print(f"  Found bracketed variable {var_name} = {value} (type: {type(value).__name__})")
            else:
                # For regular variables, convert to number
                numeric_value = safe_convert_to_number(value)
            
            # Use a placeholder that's guaranteed not to appear in formulas
            placeholder = f"§§§VAR{idx}§§§"
            placeholder_map[placeholder] = numeric_value
            
            # Replace the variable with placeholder - use word boundaries for non-bracketed variables
            if var_name.startswith('[') and var_name.endswith(']'):
                # For bracketed variables, do exact replacement
                temp_expr = temp_expr.replace(var_name, placeholder)
                print(f"  Replaced {var_name} → {placeholder} (value: {numeric_value})")
            else:
                # For regular variables, use word boundaries
                pattern = r'\b' + re.escape(var_name) + r'\b'
                matches = re.findall(pattern, temp_expr, flags=re.IGNORECASE)
                if matches:
                    temp_expr = re.sub(pattern, placeholder, temp_expr, flags=re.IGNORECASE)
                    print(f"  Replaced {var_name} → {placeholder} (value: {numeric_value})")
        
        print(f"After variable → placeholder: {temp_expr}")
        
        # Now replace all placeholders with actual numbers
        for placeholder, numeric_value in placeholder_map.items():
            temp_expr = temp_expr.replace(placeholder, str(numeric_value))
        
        print(f"After placeholder → number: {temp_expr}")
        
        # NOW convert percentages (after all variables are replaced)
        # Only match actual percentage signs
        temp_expr = re.sub(
            r'(\d+(?:\.\d+)?)\s*%',
            r'((\1)/100.0)',
            temp_expr
        )
        
        print(f"After percentage conversion: {temp_expr}")
        
        # Define allowed functions
        allowed_builtins = {
            'max': max, 'min': min, 'abs': abs, 'round': round,
            'int': int, 'float': float, 'pow': pow, 'sum': sum, 'len': len
        }
        
        # Evaluate
        result = eval(temp_expr, {"__builtins__": allowed_builtins, "math": math}, {})
        
        print(f"✅ Eval successful!")
        print(f"   Result: {result}")
        print(f"   Type: {type(result).__name__}")
        print(f"{'='*60}\n")
        
        # Return based on type
        if isinstance(result, (int, float)):
            if math.isnan(result) or math.isinf(result):
                print(f"⚠️ Result is NaN or Inf, returning None")
                return None
            return float(result)
        elif isinstance(result, (datetime, date, pd.Timestamp)):
            return result
        else:
            print(f"⚠️ Unexpected result type: {type(result)}, returning None")
            return None
    
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"❌ EVALUATION ERROR")
        print(f"{'='*60}")
        print(f"Error: {e}")
        print(f"Original expression: {expression}")
        print(f"Final expression: {temp_expr if 'temp_expr' in locals() else 'N/A'}")
        print(f"Variables: {variables}")
        import traceback
        traceback.print_exc()
        print(f"{'='*60}\n")
        return None

def calculate_row(row: pd.Series, formula_expr: str, header_to_var_mapping: Dict[str, str], is_pre_mapped: bool = False) -> Any:
    """
    Calculate formula result for a single row.
    HYBRID LOGIC: Handles [Bracketed Headers], Existing Columns, and Standard Variables.
    PRIORITY: Non-NaN original data > Calculated columns > Aliases > Mappings
    """
    var_values = {}
    
    # Create reverse mapping: variable_name -> [list of headers] (supports 1-to-many)
    var_to_header_mapping = build_var_to_headers(header_to_var_mapping)
    
    # 1. EXTRACT Bracketed Headers [Name] for Pre-Mapped formulas
    bracketed_headers = set()
    if is_pre_mapped:
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, formula_expr)
        bracketed_headers.update(matches)
        
        for header_name in bracketed_headers:
            val = None
            
            # PRIORITY 1: Check direct column match (original data or calculated)
            if header_name in row.index:
                val = row[header_name]
                if pd.notna(val):
                    var_values[f"[{header_name}]"] = val
                    print(f"✓ [PRE-MAPPED] Found [{header_name}] directly = {val}")
                    continue
            
            # PRIORITY 2: Check if this is an aliased variable
            if header_name in FORMULA_ALIASES:
                actual_column_to_check = FORMULA_ALIASES[header_name]
                if actual_column_to_check in row.index:
                    val = row[actual_column_to_check]
                    if pd.notna(val):
                        var_values[f"[{header_name}]"] = val
                        continue
            
            # PRIORITY 3: Check if header_name is a variable that maps to a column
            if header_name in var_to_header_mapping:
                val, actual_header = get_first_mapped_value(row, var_to_header_mapping[header_name])
                if actual_header is not None:
                    var_values[f"[{header_name}]"] = val
                    continue
            
            # PRIORITY 4: Case-insensitive column match
            for col in row.index:
                if col.lower() == header_name.lower():
                    val = row[col]
                    if pd.notna(val):
                        var_values[f"[{header_name}]"] = val
                        break
            
            if f"[{header_name}]" not in var_values:
                print(f"❌ WARNING: Bracketed variable '[{header_name}]' not found, defaulting to 0.0")
                var_values[f"[{header_name}]"] = 0.0

    # 2. IDENTIFY Potential Variables (non-bracketed) - FIXED to exclude function names
    potential_vars = set()
    
    # Extract from MONTHS_BETWEEN
    if 'MONTHS_BETWEEN' in formula_expr.upper():
        pattern = r'MONTHS_BETWEEN\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
        matches = re.findall(pattern, formula_expr, flags=re.IGNORECASE)
        for match in matches:
            v1 = re.findall(r'\b\w+\b', match[0])
            v2 = re.findall(r'\b\w+\b', match[1])
            potential_vars.update(v1)
            potential_vars.update(v2)
            
    # Extract from ADD_MONTHS
    if 'ADD_MONTHS' in formula_expr.upper():
        pattern = r'ADD_MONTHS\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
        matches = re.findall(pattern, formula_expr, flags=re.IGNORECASE)
        for match in matches:
            v1 = re.findall(r'\b\w+\b', match[0])
            v2 = re.findall(r'\b\w+\b', match[1])
            potential_vars.update(v1)
            potential_vars.update(v2)

    # Extract other variables (but exclude function calls and bracketed ones)
    clean_expr = re.sub(r'MONTHS_BETWEEN\([^)]+\)', '', formula_expr, flags=re.IGNORECASE)
    clean_expr = re.sub(r'ADD_MONTHS\([^)]+\)', '', clean_expr, flags=re.IGNORECASE)
    clean_expr = re.sub(r'\[[^\]]+\]', '', clean_expr)
    
    # Remove function calls like MAX(...), MIN(...) before extracting variables
    temp_expr = clean_expr
    temp_expr = re.sub(r'\b(MAX|MIN|ABS|ROUND|SUM|POWER|SQRT|POW)\s*\(', '(', temp_expr, flags=re.IGNORECASE)
    
    other_vars = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', temp_expr))
    potential_vars.update(other_vars)
    
    # FIXED: Remove Python keywords and function names (case insensitive)
    python_keywords = {
        'max', 'min', 'abs', 'round', 'sum', 'pow', 'math', 'sqrt', 'len', 'int', 'float', 
        'current_date', 'months_between', 'add_months', 'power', 'date', 'datetime'
    }
    # Case-insensitive filtering
    potential_vars = {v for v in potential_vars if v.lower() not in python_keywords}
    
    # 3. POPULATE var_values with found variables
    # PRIORITY ORDER: Direct column match (if not NaN) > Aliases > Mappings
    for var_name in potential_vars:
        # Skip if already added as bracketed variable
        if f"[{var_name}]" in var_values:
            continue
            
        val = None
        found_source = None
        
        print(f"\n🔍 Looking up variable: {var_name}")
        print(f"   Available columns: {list(row.index)[:20]}")
        
        # PRIORITY 1: Check direct column match (original data or calculated)
        if var_name in row.index:
            val = row[var_name]
            if pd.notna(val):
                var_values[var_name] = val
                found_source = f"Direct Column: {var_name}"
                print(f"✅ Found {var_name} directly in row.index = {val}")
                continue
            else:
                print(f"⚠️ Found {var_name} in row.index but value is NaN/None")
        else:
            print(f"❌ {var_name} NOT in row.index")
        
        # PRIORITY 2: Check if it's an aliased formula
        if var_name in FORMULA_ALIASES:
            aliased_col = FORMULA_ALIASES[var_name]
            print(f"   Checking alias: {var_name} → {aliased_col}")
            if aliased_col in row.index:
                val = row[aliased_col]
                if pd.notna(val):
                    var_values[var_name] = val
                    found_source = f"Alias: {var_name} → {aliased_col}"
                    print(f"✅ Found {var_name} via alias {aliased_col} = {val}")
                    continue
        
        # PRIORITY 3: Check if var maps to a header via header_to_var_mapping
        mapped_headers = var_to_header_mapping.get(var_name, [])
        if mapped_headers:
            print(f"   Checking var_to_header mapping: {var_name} → {mapped_headers}")
            val, actual_header = get_first_mapped_value(row, mapped_headers)
            if actual_header is not None:
                var_values[var_name] = val
                found_source = f"Mapping: {var_name} → {actual_header}"
                print(f"✅ Found {var_name} via mapping {actual_header} = {val}")
                continue
        
        # PRIORITY 4: Case-insensitive search
        if val is None:
            for col in row.index:
                if col.lower() == var_name.lower():
                    val = row[col]
                    if pd.notna(val):
                        var_values[var_name] = val
                        found_source = f"Case-insensitive: {var_name} → {col}"
                        print(f"✅ Found {var_name} case-insensitive as {col} = {val}")
                        break
        
        # If still not found, default to 0.0
        if var_name not in var_values:
            print(f"❌ WARNING: Variable '{var_name}' not found anywhere, defaulting to 0.0")
            print(f"   This is likely why you're seeing 0 values!")
            print(f"   Checked:")
            print(f"     - Direct column match: {var_name in row.index}")
            print(f"     - Aliases: {var_name in FORMULA_ALIASES}")
            print(f"     - header_to_var_mapping: {var_name in header_to_var_mapping}")
            print(f"     - var_to_header_mapping: {var_name in var_to_header_mapping}")
            var_values[var_name] = 0.0
    
    # DEBUG: Print what variables we're passing to safe_eval
    print(f"\n▶ Calculating: {formula_expr}")
    print(f"▶ Variables being passed to safe_eval: {list(var_values.keys())}")
    
    result = safe_eval(formula_expr, var_values)
    return result

def find_matching_column(formula_name: str, df_columns: List[str], header_to_var_mapping: Dict[str, str]) -> str:
    """Find the best matching column for a formula"""
    formula_lower = formula_name.lower().replace('_', '').replace(' ', '')
    
    for col in df_columns:
        col_clean = col.lower().replace('_', '').replace(' ', '')
        if col_clean == formula_lower:
            return col
    
    for col in df_columns:
        col_lower = col.lower()
        fname_lower = formula_name.lower()
        if fname_lower in col_lower or col_lower in fname_lower:
            return col
    
    formula_tokens = set(re.findall(r'\w+', formula_name.lower()))
    best_match = None
    best_score = 0
    
    for col in df_columns:
        col_tokens = set(re.findall(r'\w+', col.lower()))
        overlap = len(formula_tokens & col_tokens)
        if overlap > best_score:
            best_score = overlap
            best_match = col
    
    if best_match and best_score > 0:
        return best_match
    
    return formula_name

def run_calculations(df: pd.DataFrame, 
                     formulas: List[Dict], 
                     header_to_var_mapping: Dict[str, str],
                     include_derived: bool = True) -> tuple[pd.DataFrame, List[CalculationResult]]:
    """Run formulas on dataframe"""
    result_df = df.copy()
    calculation_results = []
    
    # Create reverse mapping: variable_name -> [list of headers] (supports 1-to-many)
    var_to_header_mapping = build_var_to_headers(header_to_var_mapping)
    
    all_formulas = []
    if include_derived:
        derived = get_derived_formulas()
        for f in derived: 
            f['is_pre_mapped'] = False
        all_formulas.extend(derived)
        print(f"\n📊 Added {len(derived)} derived formulas (will run FIRST)")
    
    all_formulas.extend(formulas.copy())
    
    print(f"\n🔧 Processing {len(all_formulas)} total formulas in order\n")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for formula_idx, formula in enumerate(all_formulas):
        formula_name = formula.get('formula_name', 'Unknown')
        formula_expr = formula.get('formula_expression', '')
        is_pre_mapped = formula.get('is_pre_mapped', False)
        
        print(f"\n{'='*80}")
        print(f"FORMULA {formula_idx+1}/{len(all_formulas)}: {formula_name}")
        print(f"{'='*80}")
        print(f"Expression: {formula_expr}")
        print(f"Pre-mapped: {is_pre_mapped}")
        
        # 1. Get Primary Output Column (for logging/display purposes)
        output_col = get_output_column_name(formula_name, formula, var_to_header_mapping, header_to_var_mapping)
        
        print(f"\n📍 OUTPUT COLUMN DETERMINATION:")
        print(f"   Formula name: {formula_name}")
        print(f"   Primary output column: {output_col}")

        # --- UPDATED LOGIC: Find linked columns from MAPPINGS ---
        linked_output_cols = set()
        
        # A. Determine the 'Target Variable Name' we are calculating
        # If the formula name is an alias (e.g. PV -> SURRENDER_PAID_AMOUNT), we want the target variable
        target_variable = formula_name
        if formula_name in FORMULA_ALIASES:
            target_variable = FORMULA_ALIASES[formula_name]
            print(f"   Alias detected: {formula_name} -> {target_variable}")
        
        # B. Scan header_to_var_mapping for ALL headers pointing to this target variable
        # header_to_var_mapping format: {"Header Name": "Variable_Name"}
        found_mappings = []
        for header, var_name in header_to_var_mapping.items():
            if var_name == target_variable:
                linked_output_cols.add(header)
                found_mappings.append(header)
        
        if found_mappings:
            print(f"   Found {len(found_mappings)} mapped headers for '{target_variable}': {found_mappings}")
        
        # C. Add any explicit output_column defined in the formula (if it wasn't in mappings)
        if 'output_column' in formula and formula['output_column']:
            linked_output_cols.add(formula['output_column'])
        
        # D. Add the primary output_col just in case (prevents misses)
        linked_output_cols.add(output_col)
        
        linked_output_cols = list(linked_output_cols)
        print(f"   Final list of columns to update: {linked_output_cols}")
        # ---------------------------------------------------------

        # Create columns if they don't exist
        for col in linked_output_cols:
            if col not in result_df.columns:
                result_df[col] = np.nan
                print(f"   Created new column: {col}")
        
        if output_col != formula_name:
            print(f"   ℹ️ Primary output column '{output_col}' differs from formula name '{formula_name}'")

        # SPECIAL CASE: GSV - Use existing GSV values without calculation
        if formula_name.upper() == 'GSV':
            print(f"\n⚠️ SPECIAL HANDLING FOR GSV:")
            
            # Look for GSV input columns (columns with 'gsv' that aren't the output column)
            gsv_input_cols = [col for col in result_df.columns if 'gsv' in col.lower() and col not in [output_col]]
            
            if gsv_input_cols:
                gsv_source_col = gsv_input_cols[0]
                print(f"   Found GSV input column: {gsv_source_col}")
                
                # Count non-null GSV values in the source column
                gsv_values = result_df[gsv_source_col].dropna()
                gsv_non_null_count = len(gsv_values)
                print(f"   {gsv_non_null_count} non-null values found in '{gsv_source_col}'")
                
                if gsv_non_null_count > 0:
                    print(f"   ✅ Copying GSV values directly to {linked_output_cols} without calculation")
                    # Copy GSV values from source to output column(s)
                    for col in linked_output_cols:
                        result_df[col] = result_df[gsv_source_col].copy()
                    
                    success_count = gsv_non_null_count
                    total_rows = len(result_df)
                    errors = []
                    
                    # Skip normal calculation for this formula
                    success_rate = (success_count / total_rows * 100) if total_rows > 0 else 0
                    status_text.text(f"Processing {formula_idx+1}/{len(all_formulas)}: {formula_name} (Direct copy)")
                    progress_bar.progress((formula_idx + 1) / len(all_formulas))
                    
                    print(f"   📊 Results: {success_count}/{total_rows} rows updated from {gsv_source_col}")
                    print(f"✅ GSV processing complete - skipping calculation\n")
                    # Jump to next formula
                    continue
                else:
                    print(f"   No non-null values found, falling back to normal calculation")
            else:
                print(f"   No GSV input columns found, proceeding with normal calculation")

        errors = []
        success_count = 0
        total_rows = len(result_df)
        
        # Debug first row in console
        if total_rows > 0:
            first_row = result_df.iloc[0]
            
            print(f"\n🔍 Testing first row:")
            print(f"  Available columns: {list(first_row.index)[:20]}")
            
            if is_pre_mapped:
                print("  Mode: Pre-Mapped (Direct Header Lookup)")
                pattern = r'\[([^\]]+)\]'
                headers_in_formula = re.findall(pattern, formula_expr)
                print(f"  Headers in brackets: {headers_in_formula}")
                
                for h in headers_in_formula:
                    val = None
                    if h in var_to_header_mapping:
                        val, actual_col = get_first_mapped_value(first_row, var_to_header_mapping[h])
                        display_col = actual_col if actual_col is not None else var_to_header_mapping[h]
                        print(f"    [{h}] → {display_col} = {val if actual_col is not None else 'Not found'}")
                    elif h in first_row.index:
                        val = first_row[h]
                        print(f"    [{h}] (direct) = {val}")
                    else:
                        print(f"    [{h}] = ❌ NOT FOUND")
            else:
                print("  Mode: Variable Mapping / Hybrid")
                print(f"  Variables used: {formula.get('variables_used', 'Unknown')}")
            
            first_result = calculate_row(first_row, formula_expr, header_to_var_mapping, is_pre_mapped=is_pre_mapped)
            print(f"  ✅ Test Result: {first_result}")
            print(f"  Will write to columns: {linked_output_cols}")
            
            if first_result is None:
                print(f"  ⚠️ WARNING: Test calculation returned None!")
        
        # Update UI progress
        status_text.text(f"Processing {formula_idx+1}/{len(all_formulas)}: {formula_name}")
        
        # ROW-BY-ROW CALCULATION
        for idx in range(len(result_df)):
            try:
                row = result_df.iloc[idx]
                result = calculate_row(row, formula_expr, header_to_var_mapping, is_pre_mapped=is_pre_mapped)
                
                if result is None:
                    if idx < 5:
                        errors.append(f"Row {idx}: Calculation returned None")
                    # Write NaN to all linked columns on error
                    for col in linked_output_cols:
                        result_df.at[result_df.index[idx], col] = np.nan
                else:
                    # Write the result to ALL linked columns
                    for col in linked_output_cols:
                        result_df.at[result_df.index[idx], col] = result
                    success_count += 1
            
            except Exception as e:
                if idx < 5:
                    errors.append(f"Row {idx}: {str(e)}")
                for col in linked_output_cols:
                    result_df.at[result_df.index[idx], col] = np.nan
        
        # Update progress bar
        progress_bar.progress((formula_idx + 1) / len(all_formulas))
        
        success_rate = (success_count / total_rows * 100) if total_rows > 0 else 0
        
        # Check non-null counts for the primary column
        non_null_count = result_df[output_col].notna().sum()
        
        print(f"\n✅ Result: {success_count}/{total_rows} rows ({success_rate:.1f}% success)")
        
        if non_null_count > 0:
            sample_vals = result_df[output_col].dropna().head(3).tolist()
            print(f"Sample values in {output_col}: {sample_vals}")
        else:
            print(f"⚠️ WARNING: No values calculated for {output_col}")
        
        if errors:
            print(f"⚠️ Errors in first rows:")
            for err in errors[:5]:
                print(f"  - {err}")
        
        calculation_results.append(CalculationResult(
            formula_name=f"{formula_name} → {output_col}", 
            rows_calculated=success_count,
            errors=errors[:10],
            success_rate=success_rate
        ))
    
    progress_bar.empty()
    status_text.empty()
    
    print(f"\n{'='*80}")
    print(f"CALCULATION COMPLETE")
    print(f"{'='*80}\n")
    
    return result_df, calculation_results


# --- Import Functions ---
def import_mappings_from_json(json_file) -> Dict[str, str]:
    """Import mappings from JSON file"""
    try:
        content = json_file.read()
        mappings = json.loads(content)
        
        if not isinstance(mappings, dict):
            raise ValueError("JSON must be a dictionary")
        
        clean_mappings = {}
        for k, v in mappings.items():
            header = str(k).strip()
            var_name = str(v).strip()
            if header and var_name and header != 'nan' and var_name != 'nan':
                clean_mappings[header] = var_name
        
        return clean_mappings
    except Exception as e:
        raise ValueError(f"Error reading JSON: {str(e)}")

def import_formulas_from_json(json_file) -> List[Dict]:
    """Import formulas from JSON file - Use mapped_expression if available"""
    try:
        content = json_file.read()
        data = json.loads(content)
        
        if isinstance(data, dict) and 'formulas' in data:
            formulas = data['formulas']
        elif isinstance(data, list):
            formulas = data
        else:
            raise ValueError("Invalid format")
        
        validated_formulas = []
        for formula in formulas:
            # CRITICAL: Use mapped_expression if available, otherwise use formula_expression
            if 'mapped_expression' in formula and formula['mapped_expression']:
                formula_expr = formula['mapped_expression']
                is_pre_mapped = True  # mapped_expression always has brackets
            elif 'formula_expression' in formula:
                formula_expr = formula.get('formula_expression', '')
                is_pre_mapped = '[' in formula_expr and ']' in formula_expr
            else:
                continue
            
            # Strip = sign if present
            if '=' in formula_expr and not any(op in formula_expr for op in ['==', '!=', '<=', '>=']):
                parts = formula_expr.split('=')
                if len(parts) >= 2:
                    formula_expr = '='.join(parts[1:]).strip()
            
            if not formula.get('formula_name'):
                continue
            
            # Simplified structure - only essential fields
            validated_formula = {
                'formula_name': formula['formula_name'],
                'formula_expression': formula_expr,
                'is_pre_mapped': is_pre_mapped
            }
            
            # Add output_column if specified
            if 'output_column' in formula and formula['output_column']:
                validated_formula['output_column'] = formula['output_column']
            
            validated_formulas.append(validated_formula)
        
        print(f"\n✅ Loaded {len(validated_formulas)} formulas ({sum(1 for f in validated_formulas if f['is_pre_mapped'])} pre-mapped)")
        
        return validated_formulas
        
    except Exception as e:
        raise ValueError(f"Error reading JSON: {str(e)}")

def show_detailed_calculations_for_row(result_df: pd.DataFrame, formulas: List[Dict], 
                                       header_to_var_mapping: Dict[str, str], 
                                       row_idx: int):
    """Show detailed step-by-step calculations for a specific row"""
    
    var_to_header_mapping = build_var_to_headers(header_to_var_mapping)
    
    # Include derived formulas
    all_formulas = get_derived_formulas() + formulas
    
    st.markdown(f"## 📊 Row {row_idx + 1}")
    
    # Show POLICY_REF if it exists
    if 'POLICY_REF' in result_df.columns:
        policy_ref = result_df.iloc[row_idx]['POLICY_REF']
        st.info(f"**POLICY_REF:** {policy_ref}")
    
    row = result_df.iloc[row_idx]
    
    # Show original data values
    with st.expander(f"📋 Original Data - Row {row_idx + 1}", expanded=False):
        original_data = {}
        for col in result_df.columns:
            if col not in [f['formula_name'] for f in all_formulas]:
                original_data[col] = row[col]
        st.json(original_data)
    
    # Process each formula
    for formula_idx, formula in enumerate(all_formulas):
        formula_name = formula.get('formula_name', 'Unknown')
        formula_expr = formula.get('formula_expression', '')
        is_pre_mapped = formula.get('is_pre_mapped', False)
        
        output_col = get_output_column_name(formula_name, formula, var_to_header_mapping, header_to_var_mapping)
        
        with st.expander(f"{'🔧' if formula_idx < len(get_derived_formulas()) else '📐'} {formula_name} → {output_col}", expanded=True):
            
            # Show formula
            st.code(formula_expr, language="python")
            
            # Extract variables used
            var_values = {}
            
            if is_pre_mapped:
                # Bracketed variables
                pattern = r'\[([^\]]+)\]'
                matches = re.findall(pattern, formula_expr)
                
                st.markdown("**Variable Lookup:**")
                lookup_table = []
                
                for header_name in matches:
                    val = None
                    source = "Not Found"
                    
                    # PRIORITY 1: Calculated column
                    if header_name in row.index:
                        val = row[header_name]
                        if pd.notna(val):
                            source = "Calculated Column"
                    
                    # PRIORITY 2: Check alias
                    if val is None or pd.isna(val):
                        if header_name in FORMULA_ALIASES:
                            aliased_col = FORMULA_ALIASES[header_name]
                            if aliased_col in row.index:
                                val = row[aliased_col]
                                if pd.notna(val):
                                    source = f"Alias → {aliased_col}"
                    
                    # PRIORITY 3: Check mapping
                    if val is None or pd.isna(val):
                        if header_name in var_to_header_mapping:
                            val, actual_header = get_first_mapped_value(row, var_to_header_mapping[header_name])
                            if actual_header is not None:
                                source = f"Mapping → {actual_header}"
                    
                    if val is None or pd.isna(val):
                        val = 0.0
                        source = "Default (0.0)"
                    
                    var_values[f"[{header_name}]"] = val
                    lookup_table.append({
                        'Variable': f'[{header_name}]',
                        'Value': val,
                        'Source': source
                    })
                
                # NEW: ALSO extract non-bracketed variables for pre-mapped formulas
                clean_expr = re.sub(r'\[[^\]]+\]', '', formula_expr)
                all_vars = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', clean_expr))
                python_keywords = {
                    'max', 'min', 'abs', 'round', 'sum', 'pow', 'math', 'sqrt', 'len', 'int', 'float', 
                    'current_date', 'months_between', 'add_months', 'power', 'date', 'datetime'
                }
                potential_vars = {v for v in all_vars if v.lower() not in python_keywords}
                
                for var_name in sorted(potential_vars):
                    if f"[{var_name}]" in var_values:
                        continue
                        
                    val = None
                    source = "Not Found"
                    
                    # PRIORITY 1: Calculated column
                    if var_name in row.index:
                        val = row[var_name]
                        if pd.notna(val):
                            source = "Calculated Column"
                    
                    # PRIORITY 2: Check alias
                    if val is None or pd.isna(val):
                        if var_name in FORMULA_ALIASES:
                            aliased_col = FORMULA_ALIASES[var_name]
                            if aliased_col in row.index:
                                val = row[aliased_col]
                                if pd.notna(val):
                                    source = f"Alias → {aliased_col}"
                    
                    # PRIORITY 3: Check mapping
                    if val is None or pd.isna(val):
                        if var_name in var_to_header_mapping:
                            val, actual_header = get_first_mapped_value(row, var_to_header_mapping[var_name])
                            if actual_header is not None:
                                source = f"Mapping → {actual_header}"
                    
                    if val is None or pd.isna(val):
                        val = 0.0
                        source = "Default (0.0)"
                    
                    var_values[var_name] = val
                    lookup_table.append({
                        'Variable': var_name,
                        'Value': val,
                        'Source': source
                    })
                
                if lookup_table:
                    st.table(pd.DataFrame(lookup_table))
            
            else:
                # Non-bracketed variables
                # Extract from MONTHS_BETWEEN
                potential_vars = set()
                if 'MONTHS_BETWEEN' in formula_expr.upper():
                    pattern = r'MONTHS_BETWEEN\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
                    matches = re.findall(pattern, formula_expr, flags=re.IGNORECASE)
                    for match in matches:
                        v1 = re.findall(r'\b\w+\b', match[0])
                        v2 = re.findall(r'\b\w+\b', match[1])
                        potential_vars.update(v1)
                        potential_vars.update(v2)
                
                # Extract from ADD_MONTHS
                if 'ADD_MONTHS' in formula_expr.upper():
                    pattern = r'ADD_MONTHS\s*\(\s*([^,]+)\s*,\s*([^)]+)\s*\)'
                    matches = re.findall(pattern, formula_expr, flags=re.IGNORECASE)
                    for match in matches:
                        v1 = re.findall(r'\b\w+\b', match[0])
                        v2 = re.findall(r'\b\w+\b', match[1])
                        potential_vars.update(v1)
                        potential_vars.update(v2)
                
                # Other variables
                clean_expr = re.sub(r'MONTHS_BETWEEN\([^)]+\)', '', formula_expr, flags=re.IGNORECASE)
                clean_expr = re.sub(r'ADD_MONTHS\([^)]+\)', '', clean_expr, flags=re.IGNORECASE)
                other_vars = set(re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', clean_expr))
                potential_vars.update(other_vars)
                
                python_keywords = {
                    'max', 'min', 'abs', 'round', 'sum', 'pow', 'math', 'sqrt', 'len', 'int', 'float', 
                    'current_date', 'months_between', 'add_months', 'power', 'date', 'datetime'
                }
                potential_vars = {v for v in potential_vars if v.lower() not in python_keywords}
                
                st.markdown("**Variable Lookup:**")
                lookup_table = []
                
                for var_name in sorted(potential_vars):
                    val = None
                    source = "Not Found"
                    
                    # PRIORITY 1: Calculated column
                    if var_name in row.index:
                        val = row[var_name]
                        if pd.notna(val):
                            source = "Calculated Column"
                    
                    # PRIORITY 2: Check alias
                    if val is None or pd.isna(val):
                        if var_name in FORMULA_ALIASES:
                            aliased_col = FORMULA_ALIASES[var_name]
                            if aliased_col in row.index:
                                val = row[aliased_col]
                                if pd.notna(val):
                                    source = f"Alias → {aliased_col}"
                    
                    # PRIORITY 3: Check mapping
                    if val is None or pd.isna(val):
                        if var_name in var_to_header_mapping:
                            val, actual_header = get_first_mapped_value(row, var_to_header_mapping[var_name])
                            if actual_header is not None:
                                source = f"Mapping → {actual_header}"
                    
                    if val is None or pd.isna(val):
                        val = 0.0
                        source = "Default (0.0)"
                    
                    var_values[var_name] = val
                    lookup_table.append({
                        'Variable': var_name,
                        'Value': val,
                        'Source': source
                    })
                
                if lookup_table:
                    st.table(pd.DataFrame(lookup_table))
            
            # Calculate result
            try:
                result = calculate_row(row, formula_expr, header_to_var_mapping, is_pre_mapped=is_pre_mapped)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("**Calculation:**")
                with col2:
                    # Show simplified expression with values substituted
                    simplified_expr = formula_expr
                    for var, val in var_values.items():
                        if isinstance(val, (int, float)):
                            simplified_expr = simplified_expr.replace(var, f"{val:.6f}")
                        else:
                            simplified_expr = simplified_expr.replace(var, str(val))
                    st.code(simplified_expr, language="python")
                
                if result is not None:
                    if isinstance(result, (datetime, pd.Timestamp)):
                        st.success(f"✅ **Result:** {result.strftime('%Y-%m-%d')}")
                    elif isinstance(result, float):
                        st.success(f"✅ **Result:** {result:,.6f}")
                    else:
                        st.success(f"✅ **Result:** {result}")
                    
                    st.info(f"💾 Stored in column: **{output_col}**")
                else:
                    st.error("❌ **Result:** None (calculation failed)")
                    
            except Exception as e:
                st.error(f"❌ **Error:** {str(e)}")
                import traceback
                st.code(traceback.format_exc())
            
            st.markdown("---")

def show_detailed_calculations(result_df: pd.DataFrame, formulas: List[Dict], 
                               header_to_var_mapping: Dict[str, str], 
                               num_rows: int = 3):
    """Show detailed step-by-step calculations for the first N rows"""
    
    st.markdown("---")
    st.markdown(f"### 🔍 Detailed Calculation Breakdown (First {num_rows} Rows)")
    
    for row_idx in range(min(num_rows, len(result_df))):
        show_detailed_calculations_for_row(result_df, formulas, header_to_var_mapping, row_idx)
        st.markdown("---")
        st.markdown("---")

# --- Main App ---
def main():
    st.set_page_config(page_title="Calculation Engine", page_icon="🧮", layout="wide")
    load_css()
    
    st.markdown("""
        <div class="header-container">
            <div class="header-bar">
                <img src="https://raw.githubusercontent.com/AyushiR0y/streamlit_formulagen/main/assets/logo.png" style="height: 100px;">
                <div class="header-title" style="font-size: 2.5rem; font-weight: 750; color: #004DA8;">
                    Calculation Engine
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    
    has_mappings = 'header_to_var_mapping' in st.session_state and st.session_state.header_to_var_mapping
    # Only check for mapped_formulas from previous page
    has_formulas = 'mapped_formulas' in st.session_state and st.session_state.mapped_formulas
    has_data = 'excel_df' in st.session_state and st.session_state.excel_df is not None
    
    # CRITICAL FIX: Normalize mappings to ensure correct format
    if has_mappings and 'mappings_normalized' not in st.session_state:
        print("\n" + "="*80)
        print("🔄 NORMALIZING MAPPINGS FROM PREVIOUS PAGE")
        print("="*80)
        
        normalized_mappings = {}
        
        for k, v in st.session_state.header_to_var_mapping.items():
            # Ensure both key and value are strings
            header = str(k).strip()
            var_name = str(v).strip()
            
            print(f"  Mapping: '{k}' ({type(k).__name__}) → '{v}' ({type(v).__name__})")
            
            # Skip empty or 'nan' values
            if header and var_name and header != 'nan' and var_name != 'nan':
                normalized_mappings[header] = var_name
                print(f"    ✓ Normalized to: '{header}' → '{var_name}'")
            else:
                print(f"    ✗ SKIPPED (empty or nan)")
        
        print(f"\n✅ Total normalized mappings: {len(normalized_mappings)}")
        print("="*80 + "\n")
        
        st.session_state.header_to_var_mapping = normalized_mappings
        st.session_state.mappings_normalized = True
    
    # Process mapped_formulas from previous page
    if has_formulas and 'formulas_reprocessed' not in st.session_state:
        print("\n" + "="*80)
        print("🔄 PROCESSING MAPPED_FORMULAS FROM PREVIOUS PAGE")
        print("="*80)
        
        simplified_formulas = []
        for idx, formula in enumerate(st.session_state.mapped_formulas):
            print(f"\nFormula {idx+1}: {formula.get('formula_name')}")
            
            # mapped_formulas has the structure with mapped_expression
            if 'mapped_expression' in formula:
                formula_expr = formula['mapped_expression']
                print(f"  Mapped expression: {formula_expr}")
            else:
                formula_expr = formula.get('formula_expression', '')
                print(f"  Formula expression: {formula_expr}")
            
            # mapped_formulas ALWAYS has brackets, so is_pre_mapped = True
            is_pre_mapped = True
            print(f"  Pre-mapped: {is_pre_mapped}")
            
            # Strip = sign if present
            if '=' in formula_expr and not any(op in formula_expr for op in ['==', '!=', '<=', '>=']):
                parts = formula_expr.split('=')
                if len(parts) >= 2:
                    old_expr = formula_expr
                    formula_expr = '='.join(parts[1:]).strip()
                    print(f"  ⚙️ Stripped assignment: '{old_expr}' → '{formula_expr}'")
            
            # Simplified structure - only essential fields
            simplified_formula = {
                'formula_name': formula.get('formula_name'),
                'formula_expression': formula_expr,
                'is_pre_mapped': is_pre_mapped
            }
            
            # Preserve output_column if present
            if 'output_column' in formula:
                simplified_formula['output_column'] = formula['output_column']
                print(f"  Output column: {formula['output_column']}")
            
            simplified_formulas.append(simplified_formula)
        
        print(f"\n✅ Processed {len(simplified_formulas)} mapped formulas")
        print(f"   All are pre-mapped: {all(f['is_pre_mapped'] for f in simplified_formulas)}")
        print("="*80 + "\n")
        
        # Store for use by calculation engine
        st.session_state.formulas = simplified_formulas
        st.session_state.formulas_reprocessed = True
    
    if not has_mappings or not has_formulas:
        st.warning("⚠️ Missing configuration files")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📋 Variable Mappings")
            if has_mappings:
                # Single line with count and icons
                col_info, col_download, col_clear = st.columns([3, 1, 1])
                with col_info:
                    st.success(f"✅ {len(st.session_state.header_to_var_mapping)} loaded")
                with col_download:
                    mappings_json = json.dumps(st.session_state.header_to_var_mapping, indent=2)
                    st.download_button(
                        label="📥",
                        data=mappings_json,
                        file_name="variable_mappings.json",
                        mime="application/json",
                        help="Download mappings",
                        use_container_width=True
                    )
                with col_clear:
                    if st.button("🗑️", key="clear_mappings", help="Clear mappings", use_container_width=True):
                        del st.session_state.header_to_var_mapping
                        if 'mappings_normalized' in st.session_state:
                            del st.session_state.mappings_normalized
                        st.rerun()
            
            uploaded_mapping = st.file_uploader("Upload Mappings (JSON)", type=['json'], key="map_up", label_visibility="collapsed")
            if uploaded_mapping and not has_mappings:
                try:
                    imported_mappings = import_mappings_from_json(uploaded_mapping)
                    st.session_state.header_to_var_mapping = imported_mappings
                    st.success(f"✅ {len(imported_mappings)} mappings loaded")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        with col2:
            st.markdown("### 🧮 Mapped Formulas")
            if has_formulas:
                # Get formula count from mapped_formulas
                formula_count = len(st.session_state.mapped_formulas)
                
                # Single line with count and icons
                col_info, col_download, col_clear = st.columns([3, 1, 1])
                with col_info:
                    st.success(f"✅ {formula_count} loaded")
                with col_download:
                    formulas_json = json.dumps(st.session_state.mapped_formulas, indent=2)
                    st.download_button(
                        label="📥",
                        data=formulas_json,
                        file_name="mapped_formulas.json",
                        mime="application/json",
                        help="Download mapped formulas",
                        use_container_width=True
                    )
                with col_clear:
                    if st.button("🗑️", key="clear_formulas", help="Clear formulas", use_container_width=True):
                        if 'mapped_formulas' in st.session_state:
                            del st.session_state.mapped_formulas
                        if 'formulas' in st.session_state:
                            del st.session_state.formulas
                        if 'formulas_reprocessed' in st.session_state:
                            del st.session_state.formulas_reprocessed
                        st.rerun()
            
            uploaded_formulas = st.file_uploader("Upload Mapped Formulas (JSON)", type=['json'], key="form_up", label_visibility="collapsed")
            if uploaded_formulas and not has_formulas:
                try:
                    imported_formulas = import_formulas_from_json(uploaded_formulas)
                    # Store as mapped_formulas
                    st.session_state.mapped_formulas = imported_formulas
                    st.success(f"✅ {len(imported_formulas)} mapped formulas loaded")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error: {e}")
        
        if not has_mappings or not has_formulas:
            return
    else:
        col1, col2 = st.columns(2)
        with col1:
            # Single line with count and icons
            col_info, col_download, col_clear = st.columns([3, 1, 1])
            with col_info:
                st.success(f"✅ **Mappings:** {len(st.session_state.header_to_var_mapping)}")
            with col_download:
                mappings_json = json.dumps(st.session_state.header_to_var_mapping, indent=2)
                st.download_button(
                    label="📥",
                    data=mappings_json,
                    file_name="variable_mappings.json",
                    mime="application/json",
                    key="download_mappings",
                    help="Download mappings",
                    use_container_width=True
                )
            with col_clear:
                if st.button("🗑️", key="clear_mappings_main", help="Clear mappings", use_container_width=True):
                    del st.session_state.header_to_var_mapping
                    if 'mappings_normalized' in st.session_state:
                        del st.session_state.mappings_normalized
                    st.rerun()
            
        with col2:
            # Use mapped_formulas
            formula_count = len(st.session_state.mapped_formulas) if 'mapped_formulas' in st.session_state else 0
            formulas_to_download = st.session_state.mapped_formulas if 'mapped_formulas' in st.session_state else []
            
            # Single line with count and icons
            col_info, col_download, col_clear = st.columns([3, 1, 1])
            with col_info:
                st.success(f"✅ **Mapped Formulas:** {formula_count}")
            with col_download:
                formulas_json = json.dumps(formulas_to_download, indent=2)
                st.download_button(
                    label="📥",
                    data=formulas_json,
                    file_name="mapped_formulas.json",
                    mime="application/json",
                    key="download_formulas",
                    help="Download mapped formulas",
                    use_container_width=True
                )
            with col_clear:
                if st.button("🗑️", key="clear_formulas_main", help="Clear formulas", use_container_width=True):
                    if 'mapped_formulas' in st.session_state:
                        del st.session_state.mapped_formulas
                    if 'formulas' in st.session_state:
                        del st.session_state.formulas
                    if 'formulas_reprocessed' in st.session_state:
                        del st.session_state.formulas_reprocessed
                    st.rerun()
        
        
    if not has_data:
        st.warning("⚠️ No data file loaded")
        uploaded_data = st.file_uploader("Upload Data File", type=['csv', 'xlsx', 'xls', 'json'])
        
        if uploaded_data:
            try:
                ext = Path(uploaded_data.name).suffix.lower()
                if ext == '.csv':
                    df = pd.read_csv(uploaded_data)
                elif ext == '.json':
                    df = pd.read_json(uploaded_data)
                else:
                    df = pd.read_excel(uploaded_data)
                
                st.session_state.excel_df = df
                st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
        return
    
    calc_df = st.session_state.excel_df
    st.info(f"📊 **Data:** {len(calc_df)} rows × {len(calc_df.columns)} columns")
        
    include_derived = st.checkbox("Include derived formulas", value=True, 
                                  help="Automatically include standard derived calculations")
    
    if include_derived:
        with st.expander("📋 Derived Formulas"):
            for name, info in BASIC_DERIVED_FORMULAS.items():
                st.write(f"**{name}**: {info['description']}")
                st.code(info['formula'])
        
    col1, col2 = st.columns([1, 3])
    with col1:
        if st.button("▶️ Run Calculations", type="primary", use_container_width=True):
            with st.spinner("Calculating..."):
                try:
                    result_df, calc_results = run_calculations(
                        calc_df,
                        st.session_state.formulas,
                        st.session_state.header_to_var_mapping,
                        include_derived
                    )
                    
                    st.session_state.results_df = result_df
                    st.session_state.calc_results = calc_results
                    st.success("✅ Calculations complete!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    with col2:
        if st.button("🔄 Reset Results"):
            if 'results_df' in st.session_state:
                del st.session_state.results_df
            if 'calc_results' in st.session_state:
                del st.session_state.calc_results
            st.rerun()
    
    if 'results_df' in st.session_state and st.session_state.results_df is not None:
        st.subheader("✅ Results")
        
        total_rows = len(st.session_state.results_df)
        total_formulas = len(st.session_state.calc_results)
        avg_success = sum(r.success_rate for r in st.session_state.calc_results) / total_formulas if total_formulas > 0 else 0
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", total_rows)
        with col2:
            st.metric("Formulas Applied", total_formulas)
        with col3:
            st.metric("Avg Success", f"{avg_success:.1f}%")
                
        with st.expander("📊 Formula Details", expanded=False):
            for calc_result in st.session_state.calc_results:
                icon = "✅" if calc_result.success_rate >= 90 else "⚠️" if calc_result.success_rate >= 50 else "❌"
                st.write(f"{icon} **{calc_result.formula_name}**: {calc_result.success_rate:.1f}%")
                if calc_result.errors:
                    for err in calc_result.errors[:3]:
                        st.caption(f"  {err}")
        
        st.dataframe(st.session_state.results_df, use_container_width=True, height=400)
        
        st.markdown("---")
        
        # NEW: Enhanced detailed calculation view - select specific row
        st.markdown("### 🔍 Detailed Row Calculations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Option 1: By POLICY_REF (if column exists)
            if 'POLICY_REF' in st.session_state.results_df.columns:
                policy_refs = st.session_state.results_df['POLICY_REF'].dropna().unique().tolist()
                selected_policy_ref = st.selectbox(
                    "Select by POLICY_REF:",
                    options=[''] + policy_refs,
                    format_func=lambda x: "Select..." if x == '' else str(x)
                )
                
                if selected_policy_ref:
                    row_idx = st.session_state.results_df[
                        st.session_state.results_df['POLICY_REF'] == selected_policy_ref
                    ].index[0]
                    st.session_state.selected_row_idx = row_idx
        
        with col2:
            # Option 2: By Row Number
            row_number = st.number_input(
                "Or select by row number:",
                min_value=1,
                max_value=len(st.session_state.results_df),
                value=1,
                step=1
            )
            
            if st.button("Show Row Details", type="primary"):
                st.session_state.selected_row_idx = row_number - 1
        
        # Display detailed calculations for selected row
        if 'selected_row_idx' in st.session_state:
            show_detailed_calculations_for_row(
                st.session_state.results_df,
                st.session_state.formulas,
                st.session_state.header_to_var_mapping,
                st.session_state.selected_row_idx
            )
        
        # Option to show first 3 rows as before
        if st.checkbox("🔍 Also Show First 3 Rows", value=False):
            show_detailed_calculations(
                st.session_state.results_df, 
                st.session_state.formulas,
                st.session_state.header_to_var_mapping,
                num_rows=3
            )
        
        st.markdown("---")
        
        from io import BytesIO
        col1, col2 = st.columns(2)
        
        with col1:
            @st.cache_data
            def convert_df(df):
                return df.to_csv(index=False).encode('utf-8')
            
            csv = convert_df(st.session_state.results_df)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="results.csv",
                mime="text/csv"
            )
        
        with col2:
            buffer = BytesIO()
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                st.session_state.results_df.to_excel(writer, index=False)
            
            st.download_button(
                label="📥 Download Excel",
                data=buffer.getvalue(),
                file_name="results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

if __name__ == "__main__":
    main()
    