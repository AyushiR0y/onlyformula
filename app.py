import os
import re
import json
import tempfile
from typing import List, Dict, Optional
import streamlit as st
import streamlit.components.v1 as components
from pdfminer.high_level import extract_text as extract_text_from_pdf_lib
from dotenv import load_dotenv
from dataclasses import dataclass, asdict
from openai import AzureOpenAI
import hashlib
import time
from pathlib import Path
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx'}
MAX_FILE_SIZE = 16 * 1024 * 1024
MAX_VARIANTS = 5

AZURE_API_KEY     = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_ENDPOINT    = os.getenv('AZURE_OPENAI_ENDPOINT')
DEPLOYMENT_NAME   = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
AZURE_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')

if not AZURE_API_KEY:
    MOCK_MODE = True
    client = None
else:
    try:
        client = AzureOpenAI(azure_endpoint=AZURE_ENDPOINT, api_key=AZURE_API_KEY, api_version=AZURE_API_VERSION)
        MOCK_MODE = False
    except Exception:
        MOCK_MODE = True
        client = None

STABLE_CHUNK_CONFIG = {'chunk_size':1500,'chunk_overlap':300,'max_chunks_per_formula':5,
                        'relevance_threshold':0.3,'min_chunk_size':500,'max_context_length':4000}
QUOTA_CONFIG = {'max_retries':3,'retry_delay':5,'fallback_enabled':True,
                'batch_size':5,'emergency_stop_after_failures':10}

INPUT_VARIABLES = {
    'TERM_START_DATE':'Date when the policy starts',
    'FUP_Date':'First Unpaid Premium date',
    'ENTRY_AGE':'Age of the policyholder at policy inception',
    'FULL_TERM_PREMIUM':'Annual Premium amount',
    'BOOKING_FREQUENCY':'Frequency of premium payments',
    'PREMIUM_TERM':'Premium Payment Term',
    'SUM_ASSURED':'Sum Assured - guaranteed amount on maturity/death',
    'Income_Benefit_Amount':'Amount of income benefit',
    'Income_Benefit_Frequency':'Frequency of income benefit payout',
    'DATE_OF_SURRENDER':'Date when policy is surrendered',
    'no_of_premium_paid':'Years passed since date of commencement till FUP',
    'maturity_date':'Date of commencement + (BENEFIT_TERM * 12 months)',
    'BENEFIT_TERM':'Duration (in years) for which policy benefits are payable',
    'GSV_FACTOR':'Guaranteed Surrender Value Factor',
    'SSV1_FACTOR':'Surrender Value Factor for sum assured on death',
    'SSV3_FACTOR':'Special factor for paid-up income benefits',
    'SSV2_FACTOR':'Special factor for return of premium (ROP)',
    'FUND_VALUE':'Total value of the policy fund at surrender/maturity',
    'N':'min(Policy_term, 20) - Elapsed_policy_duration',
    'SYSTEM_PAID':'Amount paid by the system for surrender or maturity',
    'FUND_FACTOR':'Factor for Surrender Charge computation',
}
BASIC_DERIVED_FORMULAS = {
    'no_of_premium_paid':'Difference between TERM_START_DATE and FUP_Date',
    'policy_year':'Difference between TERM_START_DATE and DATE_OF_SURRENDER + 1',
    'maturity_date':'TERM_START_DATE + (BENEFIT_TERM * 12) months',
    'Final_surrender_value':'Final surrender value paid',
    'Elapsed_policy_duration':'Years elapsed since policy start',
    'CAPITAL_FUND_VALUE':'Policy fund value including bonuses',
    'FUND_FACTOR':'Factor based on total premiums paid and policy term',
}
DEFAULT_TARGET_OUTPUT_VARIABLES = [
    'TOTAL_PREMIUM_PAID','TEN_TIMES_AP','one_oh_five_percent_total_premium',
    'SUM_ASSURED_ON_DEATH','GSV','PAID_UP_SA','PAID_UP_SA_ON_DEATH',
    'PAID_UP_INCOME_INSTALLMENT','SSV1','SSV2','SSV3','SSV','SURRENDER_PAID_AMOUNT',
]


# ═══════════════════════ DATA CLASSES ════════════════════════

@dataclass
class ExtractedFormula:
    formula_name: str
    formula_expression: str
    variants_info: str
    business_context: str
    source_method: str
    document_evidence: str
    specific_variables: Dict[str, str]
    is_conditional: bool = False
    conditions: List[Dict] = None
    def to_dict(self): return asdict(self)

@dataclass
class DocumentExtractionResult:
    input_variables: Dict[str, str]
    basic_derived_formulas: Dict[str, str]
    extracted_formulas: List[ExtractedFormula]
    def to_dict(self): return asdict(self)


# ═══════════════════════ COMPARISON LOGIC ════════════════════

def _normalize_commutative(expr: str) -> str:
    tokens = re.findall(r'[a-z0-9_%\.]+', (expr or "").lower())
    return str(tuple(sorted(tokens)))

def compare_formula_expressions(expr1: str, expr2: str) -> bool:
    """True only if expressions are genuinely different (not just reordered)."""
    def simple(e): return re.sub(r'\s+', '', (e or "").lower())
    if simple(expr1) == simple(expr2): return False
    if _normalize_commutative(expr1) == _normalize_commutative(expr2): return False
    return True


# ═══════════════════════ EXTRACTOR ═══════════════════════════

class StableChunkedDocumentFormulaExtractor:
    def __init__(self, target_outputs):
        self.input_variables = INPUT_VARIABLES
        self.basic_derived = BASIC_DERIVED_FORMULAS
        self.target_outputs = target_outputs
        self.config = STABLE_CHUNK_CONFIG
        self.quota_config = QUOTA_CONFIG
        self.failure_count = 0
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config['chunk_size'], chunk_overlap=self.config['chunk_overlap'],
            length_function=len, separators=["\n\n","\n","."," ",""])
        self.formula_keywords = {
            'high_priority':['surrender','gsv','ssv','formula','calculate','premium','benefit'],
            'medium_priority':['paid-up','maturity','death','sum assured','charge','value'],
            'low_priority':['policy','term','amount','date','factor','rate']}

    def _extract_offline(self, fname, context):
        patterns = [r'([A-Z_]+)\s*[=:]\s*([^.\n]+)', r'([A-Z_]+)\s*(?:is calculated|calculated as)\s*([^.\n]+)']
        fl = fname.lower()
        if fl not in context.lower(): return None
        for sentence in [s.strip() for s in context.split('.') if fl in s.lower()]:
            for pat in patterns:
                matches = re.findall(pat, sentence, re.IGNORECASE)
                if matches:
                    expr = matches[0][1] if len(matches[0]) > 1 else matches[0][0]
                    vars_found = [v for v in self.input_variables if v.lower() in expr.lower()]
                    return ExtractedFormula(
                        formula_name=fname.upper(), formula_expression=expr.strip(),
                        variants_info="Offline pattern match", business_context=f"Offline: {fname}",
                        source_method='offline', document_evidence=sentence[:200],
                        specific_variables={v: self.input_variables[v] for v in vars_found},
                        is_conditional=False, conditions=None)
        return None

    def extract_formulas_from_document(self, text):
        if MOCK_MODE or not AZURE_API_KEY: return self._no_extraction()
        try:
            prog = st.progress(0); prog.progress(10)
            chunks = self._create_chunks(text); prog.progress(20)
            scored = self._score_chunks(chunks)
            if not self._check_api(): return self._fallback(text)
            prog.progress(30)
            extracted = []; total = len(self.target_outputs)
            for bs in range(0, total, self.quota_config['batch_size']):
                be = min(bs + self.quota_config['batch_size'], total)
                for i, fname in enumerate(self.target_outputs[bs:be]):
                    prog.progress(30 + int(((bs+i)/total)*60))
                    if self.failure_count >= self.quota_config['emergency_stop_after_failures']:
                        for rem in self.target_outputs[bs+i:]:
                            r = self._extract_offline(rem, text)
                            if r: extracted.append(r)
                        break
                    result = self._extract_stable(fname, scored, text)
                    if result: extracted.append(result); self.failure_count = 0
                    else: self.failure_count += 1
                    time.sleep(0.5 if result else 1.0)
                if self.failure_count >= self.quota_config['emergency_stop_after_failures']: break
                if be < total: time.sleep(self.quota_config['retry_delay'])
            prog.progress(100)
            return DocumentExtractionResult(self.input_variables, self.basic_derived, extracted)
        except Exception as e:
            st.error(f"Extraction failed: {e}")
            return self._no_extraction()

    def _check_api(self):
        try:
            client.chat.completions.create(model=DEPLOYMENT_NAME, messages=[{"role":"user","content":"Test"}])
            return True
        except: return False

    def _fallback(self, text):
        extracted = []
        for fname in self.target_outputs:
            r = self._extract_offline(fname, text)
            if r: extracted.append(r)
        return DocumentExtractionResult(self.input_variables, self.basic_derived, extracted)

    def _create_chunks(self, text):
        parts = self.text_splitter.split_text(text)
        return [{'id':i,'text':c,'char_count':len(c),'word_count':len(c.split()),
                 'relevance_score':0.0,'position_ratio':i/len(parts)}
                for i,c in enumerate(parts) if len(c) >= self.config['min_chunk_size']]

    def _score_chunks(self, chunks):
        for ch in chunks:
            tl = ch['text'].lower()
            sc  = sum(tl.count(k)*3 for k in self.formula_keywords['high_priority'])
            sc += sum(tl.count(k)*2 for k in self.formula_keywords['medium_priority'])
            sc += sum(tl.count(k)*1 for k in self.formula_keywords['low_priority'])
            ch['relevance_score'] = (sc/(ch['word_count']/100))*(1.0 - ch['position_ratio']*0.3)
        return sorted(chunks, key=lambda x: x['relevance_score'], reverse=True)

    def _extract_stable(self, fname, scored, full_text):
        kw_map = {'surrender':['surrender','gsv','ssv','elapsed'],'premium':['premium','payment'],
                  'benefit':['benefit','income'],'death':['death','sum assured'],'maturity':['maturity']}
        rel_kw = [k for cat,kws in kw_map.items() if cat in fname.lower() for k in kws]
        formula_scored = []
        for ch in scored:
            tl = ch['text'].lower()
            sc = ch['relevance_score'] + sum(2.0 for k in rel_kw if k in tl) + (5.0 if fname.lower() in tl else 0)
            c = ch.copy(); c['fs'] = sc; formula_scored.append(c)
        formula_scored.sort(key=lambda x: x['fs'], reverse=True)
        selected = [c for c in formula_scored
                    if len([x for x in formula_scored[:self.config['max_chunks_per_formula']]]) < self.config['max_chunks_per_formula']
                    or c['fs'] >= self.config['relevance_threshold']][:self.config['max_chunks_per_formula']]
        if not selected: return None
        ctx = ""; length = 0
        for ch in selected:
            ct = ch['text']
            if length + len(ct) > self.config['max_context_length']:
                rem = self.config['max_context_length'] - length
                if rem > 200: ctx += f"\n\n--- Chunk {ch['id']} ---\n{ct[:rem]}..."
                break
            ctx += f"\n\n--- Chunk {ch['id']} ---\n{ct}"; length += len(ct)
        return self._call_api(fname, ctx)

    def _call_api(self, fname, context):
        prompt = f"""Extract the calculation formula for "{fname}" from the document.

DOCUMENT CONTENT:
{context}

AVAILABLE VARIABLES: {', '.join(self.input_variables.keys())}

INSTRUCTIONS:
1. Find or infer a mathematical formula for "{fname}" using the listed variables.
2. For MAX/MIN use MAX(...)/MIN(...). For conditions use Python-style if/else.
3. Note ON_DEATH qualifiers. Use PAID_UP_SA_ON_DEATH not longer aliases.
4. You MUST provide a formula even if inferred.

RESPONSE FORMAT (strict, no extra text):
FORMULA_EXPRESSION: [expression]
IS_CONDITIONAL: [YES or NO]
CONDITIONS:
- CONDITION_1: [condition] | EXPRESSION_1: [formula]
- CONDITION_2: [condition] | EXPRESSION_2: [formula]
VARIABLES_USED: [comma-separated]
DOCUMENT_EVIDENCE: [verbatim passage or INFERRED]
BUSINESS_CONTEXT: [1-sentence explanation]"""
        try:
            resp = client.chat.completions.create(
                model=DEPLOYMENT_NAME, messages=[{"role":"user","content":prompt}],
                max_tokens=800, temperature=0.1, top_p=0.95)
            return self._parse(resp.choices[0].message.content, fname)
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower(): time.sleep(2)
            r = self._extract_offline(fname, context)
            return r or self._placeholder(fname)

    def _placeholder(self, fname):
        return ExtractedFormula(formula_name=fname.upper(), formula_expression=f"{fname}  # not found",
                                variants_info="Placeholder", business_context=f"Placeholder for {fname}",
                                source_method='placeholder', document_evidence="No evidence",
                                specific_variables={}, is_conditional=False, conditions=None)

    def _parse(self, text, fname):
        try:
            fm = re.search(r'FORMULA_EXPRESSION:\s*(.+?)(?=\nIS_CONDITIONAL|\nVARIABLES_USED|$)', text, re.DOTALL|re.IGNORECASE)
            expr = fm.group(1).strip() if fm else "Not defined"
            cm = re.search(r'IS_CONDITIONAL:\s*(YES|NO)', text, re.IGNORECASE)
            is_cond = bool(cm and cm.group(1).upper()=="YES")
            conditions = None
            if is_cond:
                cb = re.search(r'CONDITIONS:\s*(.+?)(?=\nVARIABLES_USED|$)', text, re.DOTALL|re.IGNORECASE)
                if cb:
                    conditions = [{"condition":c.strip(),"expression":e.strip()}
                                  for c,e in re.findall(r'-\s*CONDITION_\d+:\s*(.+?)\s*\|\s*EXPRESSION_\d+:\s*(.+)', cb.group(1))]
            vm = re.search(r'VARIABLES_USED:\s*(.+?)(?=\nDOCUMENT_EVIDENCE|$)', text, re.DOTALL|re.IGNORECASE)
            spec = {}
            if vm:
                for vn in [v.strip().upper() for v in vm.group(1).split(',') if v.strip()]:
                    if vn in self.input_variables: spec[vn] = self.input_variables[vn]
                    elif vn in self.basic_derived: spec[vn] = self.basic_derived[vn]
                    else:
                        for iv in self.input_variables:
                            if vn in iv or iv in vn: spec[iv] = self.input_variables[iv]; break
            em = re.search(r'DOCUMENT_EVIDENCE:\s*(.+?)(?=\nBUSINESS_CONTEXT|$)', text, re.DOTALL|re.IGNORECASE)
            evidence = em.group(1).strip() if em else "No supporting evidence found"
            bm = re.search(r'BUSINESS_CONTEXT:\s*(.+?)$', text, re.DOTALL|re.IGNORECASE)
            biz = bm.group(1).strip() if bm else f"Calculation for {fname}"
            return ExtractedFormula(formula_name=fname.upper(), formula_expression=expr,
                                    variants_info="Stable chunking", business_context=biz,
                                    source_method='stable_chunked', document_evidence=evidence[:800],
                                    specific_variables=spec, is_conditional=is_cond, conditions=conditions)
        except: return None

    def _no_extraction(self):
        return DocumentExtractionResult(self.input_variables, self.basic_derived, [])


def normalize_formulas(formulas):
    out = []
    for f in formulas:
        expr = f.formula_expression or ""
        if (f.formula_name or "").upper() == "TOTAL_PREMIUM_PAID":
            expr = "FULL_TERM_PREMIUM * no_of_premium_paid * BOOKING_FREQUENCY"
        expr = re.sub(r'present_value_of_paid_up_sum_assured_on_death','PAID_UP_SA_ON_DEATH',expr,flags=re.IGNORECASE)
        if expr != f.formula_expression:
            f = ExtractedFormula(f.formula_name, expr, f.variants_info, f.business_context,
                                  f.source_method, f.document_evidence, f.specific_variables,
                                  f.is_conditional, f.conditions)
        out.append(f)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def extract_formulas_cached(file_hash, text, target_outputs, chunk_size, chunk_overlap):
    extractor = StableChunkedDocumentFormulaExtractor(target_outputs=target_outputs)
    result = extractor.extract_formulas_from_document(text)
    return {'extracted_formulas': [f.to_dict() for f in result.extracted_formulas]}


def extract_text_from_file(file_bytes, ext):
    try:
        if ext == '.pdf':
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
                tmp.write(file_bytes); path = tmp.name
            try: text = extract_text_from_pdf_lib(path)
            finally: os.unlink(path)
            return text
        elif ext == '.txt': return file_bytes.decode('utf-8')
        elif ext == '.docx':
            import docx; from io import BytesIO
            return '\n'.join(p.text for p in docx.Document(BytesIO(file_bytes)).paragraphs)
    except Exception as e: st.error(f"File read error: {e}")
    return ""


def run_extraction_for_variant(variant_name, uploaded_files, target_outputs):
    if not uploaded_files: return None
    combined = "".join("\n\n" + extract_text_from_file(uf.read(), Path(uf.name).suffix.lower()) for uf in uploaded_files)
    if not combined.strip(): st.error(f"No text for '{variant_name}'"); return None
    if not MOCK_MODE and AZURE_API_KEY:
        cache_key = hashlib.md5(combined.encode()).hexdigest() + "_".join(sorted(target_outputs))
        try:
            rd = extract_formulas_cached(cache_key, combined, target_outputs,
                                          STABLE_CHUNK_CONFIG['chunk_size'], STABLE_CHUNK_CONFIG['chunk_overlap'])
            fms = normalize_formulas([ExtractedFormula(**f) for f in rd['extracted_formulas']])
        except:
            ext = StableChunkedDocumentFormulaExtractor(target_outputs=target_outputs)
            fms = normalize_formulas(ext.extract_formulas_from_document(combined).extracted_formulas)
        return [f.to_dict() for f in fms]
    st.warning("⚠️ Azure OpenAI not configured.")
    return None


def build_comparison_table(vr, vnames):
    all_fn = []
    for vn in vnames:
        for f in vr.get(vn,[]):
            fn = f.get("formula_name","")
            if fn and fn not in all_fn: all_fn.append(fn)
    rows = []
    for fn in all_fn:
        row = {"Formula": fn}
        exprs = {}
        for vn in vnames:
            m = next((f for f in vr.get(vn,[]) if f.get("formula_name")==fn), None)
            e = m["formula_expression"] if m else "—"
            row[vn] = e; exprs[vn] = e
        unique = [e for e in exprs.values() if e != "—"]
        has_diff = len(unique) > 1 and any(
            compare_formula_expressions(unique[i], unique[j])
            for i in range(len(unique)) for j in range(i+1,len(unique)))
        row["_diff"] = has_diff; rows.append(row)
    return pd.DataFrame(rows)


# ═══════════════════════ CSS / THEME ═════════════════════════

def load_css():
    css_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "style.css")
    css_content = ""
    if os.path.exists(css_path):
        with open(css_path, 'r', encoding='utf-8') as f:
            css_content = f.read()

    # Add viewport meta tag for mobile responsiveness
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes">
    """, unsafe_allow_html=True)
    
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


# ═══════════════════════ UI HELPERS ══════════════════════════

def render_header():
    st.markdown("""
    <div class="fai-header">
        <div class="fai-logo-area">
            <img src="https://raw.githubusercontent.com/AyushiR0y/streamlit_formulagen/main/assets/logo.png"
                 onerror="this.style.display='none'" alt="">
            <div>
                <div class="fai-title">Formula AI</div>
                <div class="fai-subtitle">Document Intelligence · Formula Extraction</div>
            </div>
        </div>
        <label class="theme-toggle" title="Toggle dark / light mode">
            <input type="checkbox" id="faiToggle" onchange="
                if (window.FAITheme) {
                    const t = window.FAITheme.toggle();
                    this.checked = (t === 'light');
                }
            ">
            <div class="toggle-track">
                <span class="toggle-icon moon">🌙</span>
                <span class="toggle-icon sun">☀️</span>
                <div class="toggle-knob"></div>
            </div>
        </label>
    </div>
    <script>
    (function() {
        if (window.FAITheme) return; // Already initialized
        
        const KEY = 'fai_theme';
        
        function apply(t) {
            document.documentElement.setAttribute('data-theme', t);
            document.body.setAttribute('data-theme', t);
            const app = document.querySelector('[data-testid="stAppViewContainer"]');
            if (app) app.setAttribute('data-theme', t);
            const main = document.querySelector('[data-testid="stMain"]');
            if (main) main.setAttribute('data-theme', t);
            
            if (t === 'light') {
                document.body.style.backgroundColor = '#f8fafc';
            } else {
                document.body.style.backgroundColor = '#05070f';
            }
        }
        
        const saved = localStorage.getItem(KEY) || 'dark';
        apply(saved);
        
        window.FAITheme = {
            get: () => localStorage.getItem(KEY) || 'dark',
            toggle: function() {
                const next = (localStorage.getItem(KEY)||'dark') === 'dark' ? 'light' : 'dark';
                localStorage.setItem(KEY, next);
                apply(next);
                return next;
            }
        };
        
        // Sync checkbox state
        setTimeout(function() {
            const cb = document.getElementById('faiToggle');
            if (cb && window.FAITheme) cb.checked = window.FAITheme.get() === 'light';
        }, 100);
        
        // Re-apply on changes
        const obs = new MutationObserver(() => {
            const t = localStorage.getItem(KEY) || 'dark';
            if (document.documentElement.getAttribute('data-theme') !== t) {
                apply(t);
            }
        });
        obs.observe(document.body, {childList: true, subtree: true});
    })();
    </script>
    """, unsafe_allow_html=True)


def section_header(icon, title, subtitle=""):
    sub = f'<p class="sh-sub">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div class="section-header animate-in">
        <div class="sh-icon">{icon}</div>
        <div class="sh-text"><h3>{title}</h3>{sub}</div>
    </div>""", unsafe_allow_html=True)


def stat_bar(*pills):
    html = '<div class="stat-bar animate-in">'
    for label, value, cls in pills:
        html += f'<div class="stat-pill {cls}"><strong>{value}</strong>&nbsp;{label}</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════ FORMULA CARD ════════════════════════

def render_formula_card(formula: Dict, highlight: bool = False, idx: int = 0):
    expr      = formula.get("formula_expression", "")
    name      = formula.get("formula_name", "")
    evidence  = formula.get("document_evidence", "")
    context   = formula.get("business_context", "")
    is_cond   = formula.get("is_conditional", False)
    conditions = formula.get("conditions") or []
    is_inf    = evidence in ("INFERRED",)
    no_ev     = not evidence or evidence in ("No supporting evidence found","No evidence found")

    badges = ""
    if highlight:  badges += '<span class="badge badge-diff">⚠ Differs</span>'
    else:          badges += '<span class="badge badge-same">✓ Match</span>'
    if is_inf:     badges += ' <span class="badge badge-warn">Inferred</span>'
    if is_cond:    badges += ' <span class="badge badge-info">Conditional</span>'

    ctx_html = f'<div class="fc-context">💡 {context}</div>' if (context and context != f"Calculation for {name}") else ""
    card_cls = "glass-card fc differs" if highlight else "glass-card fc"

    # Header — pure HTML, no nested st.* widgets
    st.markdown(f"""
    <div class="{card_cls}" style="animation-delay:{idx*0.04:.2f}s">
      <div class="fc-inner">
        <div class="fc-header">
          <span class="fc-name">{name}</span>
          <span class="fc-badges">{badges}</span>
        </div>
        {ctx_html}
      </div>
    </div>""", unsafe_allow_html=True)

    # Code block — native Streamlit, outside the HTML div
    st.code(expr, language="python")

    # Conditional branches
    if is_cond and conditions:
        with st.expander("📊 Conditional Branches", expanded=True):
            for i, cond in enumerate(conditions):
                st.markdown(f"""
                <div class="branch-block">
                  <div class="branch-label">Branch {i+1}</div>
                  <div class="branch-cond">{cond.get('condition','')}</div>
                </div>""", unsafe_allow_html=True)
                st.code(cond.get('expression',''), language="python")

    # Evidence
    if not no_ev and not is_inf:
        with st.expander("📄 Source Evidence", expanded=False):
            st.markdown(f'<div class="evidence-block">{evidence}</div>', unsafe_allow_html=True)
    elif is_inf:
        st.markdown('<span class="badge badge-warn">⚠ Inferred — no explicit document text</span>', unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)


# ═══════════════════════ MAIN ════════════════════════════════

def main():
    st.set_page_config(page_title="Formula AI", page_icon="🧮", layout="wide",
                       initial_sidebar_state="collapsed")
    load_css()
    render_header()

    defaults = {
        'extraction_result':None,'selected_output_variables':DEFAULT_TARGET_OUTPUT_VARIABLES.copy(),
        'custom_output_variable':"",'user_defined_output_variables':[],'formulas':[],
        'formulas_saved':False,'editing_formula':-1,
        'previous_selected_variables':DEFAULT_TARGET_OUTPUT_VARIABLES.copy(),
        'variant_results':{},'num_variants':2,'variant_names':['Product A','Product B'],'mode':'single',
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    # ── KEYWORD SELECTION ─────────────────────────────────────
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        section_header("🔑", "Target Keywords", "Which formulas should be extracted?")
        all_opts = sorted(set(DEFAULT_TARGET_OUTPUT_VARIABLES + st.session_state.user_defined_output_variables))
        current_sel = st.multiselect("Keywords:", options=all_opts,
                                     default=st.session_state.selected_output_variables)
        if current_sel != st.session_state.previous_selected_variables:
            st.session_state.selected_output_variables = current_sel
            st.session_state.previous_selected_variables = current_sel.copy()
            st.session_state.extraction_result = None; st.session_state.formulas = []
            st.session_state.formulas_saved = False; st.session_state.editing_formula = -1
            st.session_state.variant_results = {}; st.rerun()
        else:
            st.session_state.selected_output_variables = current_sel

        ac1, ac2 = st.columns([5, 1])
        with ac1:
            custom = st.text_input("Custom keyword:", value=st.session_state.custom_output_variable,
                                   key="custom_input", placeholder="e.g. BONUS_AMOUNT", label_visibility="collapsed")
            st.session_state.custom_output_variable = custom
        with ac2:
            if st.button("＋", key="add_kw", use_container_width=True, help="Add"):
                nv = custom.strip()
                if nv and nv not in st.session_state.user_defined_output_variables and nv not in DEFAULT_TARGET_OUTPUT_VARIABLES:
                    st.session_state.user_defined_output_variables.append(nv)
                    if nv not in st.session_state.selected_output_variables:
                        st.session_state.selected_output_variables.append(nv)
                        st.session_state.previous_selected_variables = st.session_state.selected_output_variables.copy()
                    st.session_state.custom_output_variable = ""
                    st.session_state.extraction_result = None; st.session_state.formulas = []
                    st.session_state.variant_results = {}; st.success(f"✅ '{nv}' added!"); st.rerun()
                elif nv: st.info("Already in list.")

    with col2:
        section_header("📖", "Reference Variables")
        with st.expander("Input Variables", expanded=False):
            st.dataframe(pd.DataFrame([{"Variable":k,"Description":v} for k,v in INPUT_VARIABLES.items()]),
                         use_container_width=True, hide_index=True)
        with st.expander("Basic Derived Formulas", expanded=False):
            st.dataframe(pd.DataFrame([{"Variable":k,"Description":v} for k,v in BASIC_DERIVED_FORMULAS.items()]),
                         use_container_width=True, hide_index=True)
        targets = sorted(set(st.session_state.selected_output_variables))
        with st.expander(f"Selected ({len(targets)})", expanded=True):
            if targets:
                pills = "".join(f'<span class="badge badge-info" style="margin:3px 2px;display:inline-flex">{t}</span>' for t in targets)
                st.markdown(f'<div style="line-height:2.6">{pills}</div>', unsafe_allow_html=True)
            else: st.info("No targets selected.")

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── UPLOAD MODE ───────────────────────────────────────────
    section_header("📂", "Upload Documents")
    mc, _ = st.columns([2, 4])
    with mc:
        upload_mode = st.radio("Mode", ["Single Product","Multiple Variants (compare)"],
                               index=0 if st.session_state.mode=='single' else 1, horizontal=True)
        nm = 'single' if upload_mode == "Single Product" else 'multi'
        if nm != st.session_state.mode:
            st.session_state.mode = nm; st.session_state.extraction_result = None
            st.session_state.formulas = []; st.session_state.variant_results = {}; st.rerun()

    # ── SINGLE ────────────────────────────────────────────────
    if st.session_state.mode == 'single':
        uk = f"upl_s_{hash(str(sorted(st.session_state.selected_output_variables)))}"
        uploaded = st.file_uploader("Drop documents here", type=list(ALLOWED_EXTENSIONS),
                                    accept_multiple_files=True, key=uk)
        if uploaded:
            pills = "".join(f'<span class="stat-pill">📄 <strong>{uf.name}</strong>&nbsp;·&nbsp;{uf.size/1024:.1f} KB</span>'
                            for uf in uploaded if uf.size <= MAX_FILE_SIZE)
            st.markdown(f'<div class="stat-bar">{pills}</div>', unsafe_allow_html=True)
            if st.button("🔍 Analyze Documents", type="primary", key="analyze_s"):
                if not st.session_state.selected_output_variables:
                    st.warning("Select at least one keyword first.")
                else:
                    with st.spinner("Extracting formulas..."):
                        fms = run_extraction_for_variant("default", uploaded, st.session_state.selected_output_variables)
                    if fms is not None:
                        st.session_state.formulas = fms; st.session_state.extraction_result = True
                        st.session_state.formulas_saved = False; st.session_state.editing_formula = -1
                        st.success(f"✅ {len(fms)} formula(s) extracted!"); st.rerun()

    # ── MULTI ─────────────────────────────────────────────────
    else:
        nc, _ = st.columns([1,5])
        with nc:
            num_v = st.number_input("Variants", min_value=2, max_value=MAX_VARIANTS,
                                    value=st.session_state.num_variants, step=1)
        if num_v != st.session_state.num_variants:
            st.session_state.num_variants = num_v
            while len(st.session_state.variant_names) < num_v:
                st.session_state.variant_names.append(f"Variant {len(st.session_state.variant_names)+1}")
            st.session_state.variant_names = st.session_state.variant_names[:num_v]; st.rerun()

        vfiles = {}; all_filled = True
        cpr = min(num_v, 3)
        for ri in range((num_v+cpr-1)//cpr):
            cols = st.columns(cpr, gap="medium")
            for ci in range(cpr):
                vi = ri*cpr+ci
                if vi >= num_v: break
                with cols[ci]:
                    vn = st.text_input(f"Variant {vi+1}", value=st.session_state.variant_names[vi],
                                       key=f"vn_{vi}", placeholder="e.g. Product A")
                    st.session_state.variant_names[vi] = vn
                    fk = f"vf_{vi}_{hash(str(sorted(st.session_state.selected_output_variables)))}"
                    files = st.file_uploader(f"Files for '{vn}'", type=list(ALLOWED_EXTENSIONS),
                                             accept_multiple_files=True, key=fk)
                    vfiles[vn] = files or []
                    if not files: all_filled = False
                    else:
                        st.markdown(f'<span class="stat-pill">📎 {len(files)} file(s)&nbsp;·&nbsp;{sum(f.size for f in files)/1024:.1f} KB</span>',
                                    unsafe_allow_html=True)
        if not all_filled:
            st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
            st.info("⬆️ Upload at least one document per variant to proceed.")
        st.write("")
        if st.button("🔍 Analyze All Variants", type="primary", key="analyze_m", disabled=not all_filled):
            if not st.session_state.selected_output_variables:
                st.warning("Select at least one keyword.")
            else:
                st.session_state.variant_results = {}
                for vn in st.session_state.variant_names[:num_v]:
                    files = vfiles.get(vn,[])
                    if files:
                        with st.spinner(f"Extracting: {vn}..."):
                            fms = run_extraction_for_variant(vn, files, st.session_state.selected_output_variables)
                        if fms is not None:
                            st.session_state.variant_results[vn] = fms
                            st.success(f"✅ {vn}: {len(fms)} formula(s)")
                st.session_state.extraction_result = True; st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # RESULTS — SINGLE
    # ══════════════════════════════════════════════════════════
    if st.session_state.mode == 'single' and st.session_state.extraction_result and st.session_state.formulas:
        fms = st.session_state.formulas
        n_c = sum(1 for f in fms if f.get("is_conditional"))
        n_i = sum(1 for f in fms if f.get("document_evidence")=="INFERRED")
        section_header("📋", "Extracted Formulas", f"{len(fms)} formulas found")
        stat_bar((f"extracted", len(fms),""), ("conditional",n_c,""), ("inferred",n_i,"warn" if n_i else ""))

        tab_v, tab_e = st.tabs(["👁  View & Evidence","✏️  Edit Formulas"])
        with tab_v:
            for i,f in enumerate(fms): render_formula_card(f, idx=i)
        with tab_e:
            h1,h2,h3 = st.columns([3,5,2])
            with h1: st.markdown("**Formula**")
            with h2: st.markdown("**Expression**")
            with h3: st.markdown("**Actions**")
            for i,formula in enumerate(st.session_state.formulas):
                c1,c2,c3 = st.columns([3,5,2])
                with c1:
                    if st.session_state.editing_formula==i:
                        nn = st.text_input("n",value=formula.get("formula_name",""),key=f"en{i}",label_visibility="collapsed")
                    else: st.markdown(f'<span class="fc-name" style="font-size:.84rem">{formula.get("formula_name","")}</span>',unsafe_allow_html=True)
                with c2:
                    if st.session_state.editing_formula==i:
                        ne = st.text_area("e",value=formula.get("formula_expression",""),key=f"ee{i}",label_visibility="collapsed",height=68)
                    else: st.code(formula.get("formula_expression",""),language="python")
                with c3:
                    if st.session_state.editing_formula==i:
                        s1,s2=st.columns(2)
                        with s1:
                            if st.button("💾",key=f"sv{i}"):
                                st.session_state.formulas[i]["formula_name"]=nn
                                st.session_state.formulas[i]["formula_expression"]=ne
                                st.session_state.editing_formula=-1; st.rerun()
                        with s2:
                            if st.button("✕",key=f"cx{i}"): st.session_state.editing_formula=-1; st.rerun()
                    else:
                        s1,s2=st.columns(2)
                        with s1:
                            if st.button("✏️",key=f"ed{i}"): st.session_state.editing_formula=i; st.rerun()
                        with s2:
                            if st.button("🗑",key=f"dl{i}"):
                                st.session_state.formulas.pop(i); st.session_state.editing_formula=-1; st.rerun()
                if i < len(st.session_state.formulas)-1: st.markdown('<hr style="margin:.3rem 0">',unsafe_allow_html=True)
            st.markdown("<hr>",unsafe_allow_html=True)
            st.markdown("**➕ Add New Formula**")
            a1,a2,a3=st.columns([3,5,2])
            with a1: nfn=st.text_input("n",key="nfn",label_visibility="collapsed",placeholder="Formula name")
            with a2: nfe=st.text_area("e",key="nfe",label_visibility="collapsed",placeholder="Expression",height=68)
            with a3:
                st.write("")
                if st.button("＋ Add",key="nadd"):
                    if nfn.strip() and nfe.strip():
                        st.session_state.formulas.append({"formula_name":nfn.strip(),"formula_expression":nfe.strip()})
                        st.rerun()

        st.markdown("<hr>",unsafe_allow_html=True)
        section_header("⬇️","Export Results")
        e1,e2=st.columns(2)
        with e1:
            st.download_button("📥 Download JSON",data=json.dumps({"formulas":fms},indent=2),
                               file_name="formulas.json",mime="application/json",use_container_width=True)
        with e2:
            csv=pd.DataFrame([{"Formula":f.get("formula_name",""),"Expression":f.get("formula_expression","")} for f in fms]).to_csv(index=False)
            st.download_button("📥 Download CSV",data=csv,file_name="formulas.csv",mime="text/csv",use_container_width=True)

    # ══════════════════════════════════════════════════════════
    # RESULTS — MULTI VARIANT
    # ══════════════════════════════════════════════════════════
    elif st.session_state.mode=='multi' and st.session_state.extraction_result and st.session_state.variant_results:
        vr = st.session_state.variant_results
        vnames = [n for n in st.session_state.variant_names[:st.session_state.num_variants] if n in vr]
        if not vnames: st.warning("No results."); return

        all_fn = []
        for vn in vnames:
            for f in vr[vn]:
                fn=f.get("formula_name","")
                if fn and fn not in all_fn: all_fn.append(fn)
        differing = set()
        for fn in all_fn:
            exprs=[f.get("formula_expression","—") for vn in vnames
                   for f in [next((x for x in vr[vn] if x.get("formula_name")==fn),None)] if f]
            unique=[e for e in exprs if e!="—"]
            if len(unique)>1 and any(compare_formula_expressions(unique[i],unique[j])
                                      for i in range(len(unique)) for j in range(i+1,len(unique))):
                differing.add(fn)

        n_same=len(all_fn)-len(differing)
        section_header("📊","Multi-Variant Comparison",f"{len(vnames)} variants analyzed")
        stat_bar((f"formulas",len(all_fn),""),("differ",len(differing),"danger" if differing else "success"),
                 ("identical",n_same,"success"),(f"variants",len(vnames),""))

        with st.expander("🔍 Comparison Table",expanded=True):
            df=build_comparison_table(vr,vnames)
            disp=df.drop(columns=['_diff'])
            styled=disp.style.apply(lambda row:[
                'background-color:rgba(255,60,90,.12);color:#ff4d6d;font-weight:600'
                if df.loc[row.name,'_diff'] else
                'background-color:rgba(0,200,140,.06);color:#00e5a0'
                for _ in row],axis=1)
            st.dataframe(styled,use_container_width=True,hide_index=True)
            st.markdown("""<div class="diff-legend">
            <div class="diff-legend-item"><div class="legend-dot danger"></div>Differs across variants</div>
            <div class="diff-legend-item"><div class="legend-dot success"></div>Identical across variants</div>
            </div>""",unsafe_allow_html=True)

        st.markdown("<hr>",unsafe_allow_html=True)
        section_header("📄","Formulas per Variant")
        tabs=st.tabs(vnames)
        for tab,vn in zip(tabs,vnames):
            with tab:
                fm_list=vr[vn]
                nd=sum(1 for f in fm_list if f.get("formula_name") in differing)
                stat_bar((f"formulas",len(fm_list),""),("differ",nd,"danger" if nd else "success"))
                show_d=st.checkbox("Show only differing",key=f"donly_{vn}")
                for i,formula in enumerate(fm_list):
                    hi=formula.get("formula_name","") in differing
                    if show_d and not hi: continue
                    render_formula_card(formula,highlight=hi,idx=i)

        st.markdown("<hr>",unsafe_allow_html=True)
        section_header("⚠️","Differences Summary",f"{len(differing)} formula(s) with genuine differences")
        if differing:
            for fn in sorted(differing):
                with st.expander(f"⚠️  {fn}",expanded=False):
                    dcols=st.columns(len(vnames))
                    for dc,vn in zip(dcols,vnames):
                        with dc:
                            m=next((f for f in vr[vn] if f.get("formula_name")==fn),None)
                            st.markdown(f"**{vn}**")
                            if m:
                                st.code(m.get("formula_expression","—"),language="python")
                                ev=m.get("document_evidence","")
                                if ev and ev not in ("No supporting evidence found","No evidence found","INFERRED"):
                                    st.caption(f"📄 {ev[:280]}...")
                            else: st.markdown("*Not found*")
        else:
            st.markdown("""<div class="glass-card" style="padding:28px;text-align:center">
            <div class="fc-inner"><span style="font-size:2.4rem">✅</span>
            <h3 style="margin:12px 0 6px;font-family:'Syne',sans-serif">All Identical</h3>
            <p style="color:var(--text-secondary);margin:0">Every formula is consistent across all variants.</p>
            </div></div>""",unsafe_allow_html=True)

        st.markdown("<hr>",unsafe_allow_html=True)
        section_header("⬇️","Export Results")
        export={"variants":vnames,"differing":list(differing),"results":{vn:vr[vn] for vn in vnames}}
        flat=[{"Variant":vn,"Formula":f.get("formula_name",""),"Expression":f.get("formula_expression",""),
               "Differs":f.get("formula_name","") in differing}
              for vn in vnames for f in vr[vn]]
        e1,e2=st.columns(2)
        with e1:
            st.download_button("📥 JSON",data=json.dumps(export,indent=2,default=str),
                               file_name="variants.json",mime="application/json",use_container_width=True)
        with e2:
            st.download_button("📥 CSV",data=pd.DataFrame(flat).to_csv(index=False),
                               file_name="comparison.csv",mime="text/csv",use_container_width=True)

    st.markdown("""
    <div class="fai-footer">
        <span>DEVELOPED BY BLIC GENAI TEAM &nbsp;·&nbsp; FORMULA AI v2.0</span>
    </div>""",unsafe_allow_html=True)


if __name__ == "__main__":
    main()