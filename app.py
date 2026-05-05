"""
โปรแกรมจำลองการแก้สมการเชิงผลต่างอันดับสองแบบเอกพันธ์
Second-Order Homogeneous Difference Equation

รูปแบบมาตรฐาน:  y(n+2) + a·y(n+1) + b·y(n) = 0

วิธีรัน:
    pip install -r requirements.txt
    streamlit run app.py
"""
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


EPS = 1e-9


# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="สมการเชิงผลต่างอันดับสองแบบเอกพันธ์",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =====================================================================
# THEME — CSS variables for both modes
# =====================================================================
DARK_VARS = """
:root {
    --bg-page: #1B1B1F;
    --bg-box: #34343A;
    --border-box: #42424A;
    --text-primary: #E6E6E8;
    --text-secondary: #BFBFC4;
    --text-subtle: #9C9CA0;
    --h1-color: #F2F2F4;
    --section-title: #F0F0F2;

    --bg-general: #26354E;
    --border-general: #466292;
    --bg-particular: #263921;
    --border-particular: #527A36;

    --tag-method-bg: #2C2C32;
    --tag-method-text: #B7B7BD;
    --tag-general-bg: #1F3252;
    --tag-general-text: #9DBDE6;
    --tag-particular-bg: #233D1B;
    --tag-particular-text: #A6CD83;

    --bg-case: #2D2D33;
    --border-case: #393940;
    --text-case: #95959B;
    --marker-case: #5C5C62;
    --bg-case-chosen: #2E3A4F;
    --border-case-chosen: #5278A6;
    --text-case-chosen: #ECEFF5;
    --marker-case-chosen: #7AAEEC;

    --divider-section: #2D2D33;

    --sidebar-bg: #ECE9E1;
    --sidebar-border: #D9D5C9;
    --sidebar-h: #5A5854;
    --sidebar-text: #2A2826;
    --sidebar-caption: #6E6C66;

    --btn-bg: #2C2C32;
    --btn-text: #E6E6E8;
    --btn-border: #3A3A40;
    --btn-hover-bg: #3C3C42;
    --btn-hover-border: #4E4E54;
    --btn-disabled-bg: #1F1F23;
    --btn-disabled-text: #5A5A60;
    --btn-disabled-border: #28282E;
    --btn-primary-bg: #4F7BC4;
    --btn-primary-border: #5B8FE6;
    --btn-primary-hover-bg: #5B8FE6;

    --code-bg: #2C2C32;
    --code-text: #E6E6E8;
    --welcome-text: #8B8B91;
}
"""

LIGHT_VARS = """
:root {
    --bg-page: #F7F6F1;
    --bg-box: #EDE9DE;
    --border-box: #D5D1C4;
    --text-primary: #1F1F1F;
    --text-secondary: #5A5854;
    --text-subtle: #7A7872;
    --h1-color: #1A1A1A;
    --section-title: #1F1F1F;

    --bg-general: #DDE7F4;
    --border-general: #94B2D8;
    --bg-particular: #E1ECD0;
    --border-particular: #99BC68;

    --tag-method-bg: #DDD9CD;
    --tag-method-text: #4F4D48;
    --tag-general-bg: #C0D2EB;
    --tag-general-text: #234A78;
    --tag-particular-bg: #C5DCA6;
    --tag-particular-text: #3A5C18;

    --bg-case: #E4E0D5;
    --border-case: #CFCBBE;
    --text-case: #7A7872;
    --marker-case: #A8A39A;
    --bg-case-chosen: #BFD3EC;
    --border-case-chosen: #4F77A8;
    --text-case-chosen: #1A3859;
    --marker-case-chosen: #3D72B0;

    --divider-section: #D5D1C4;

    --sidebar-bg: #EFECE3;
    --sidebar-border: #DCD8CC;
    --sidebar-h: #5A5854;
    --sidebar-text: #2A2826;
    --sidebar-caption: #6E6C66;

    --btn-bg: #DDD9CD;
    --btn-text: #2A2826;
    --btn-border: #C9C5B8;
    --btn-hover-bg: #CCC8BB;
    --btn-hover-border: #B0AC9F;
    --btn-disabled-bg: #EFECE2;
    --btn-disabled-text: #A0A099;
    --btn-disabled-border: #DBD7CB;
    --btn-primary-bg: #3D6FBA;
    --btn-primary-border: #5189D8;
    --btn-primary-hover-bg: #5189D8;

    --code-bg: #DDD9CD;
    --code-text: #2A2826;
    --welcome-text: #6E6C66;
}
"""

DARK_GRAPH = {
    "fig_bg": "#34343A", "axis_bg": "#1B1B1F", "spine": "#3A3A40",
    "tick": "#B7B7BD", "label": "#D6D6D9", "zero": "#555555",
    "grid": "#FFFFFF", "iter": "#7AAEEC", "anal": "#E0934A",
    "legend_bg": "#2C2C32", "legend_border": "#3A3A40",
    "legend_text": "#E6E6E8", "grid_alpha": 0.18,
}
LIGHT_GRAPH = {
    "fig_bg": "#FFFFFF", "axis_bg": "#FAF9F4", "spine": "#B0ACA0",
    "tick": "#5A5854", "label": "#1F1F1F", "zero": "#999999",
    "grid": "#000000", "iter": "#1B4F7A", "anal": "#C45A30",
    "legend_bg": "#FFFFFF", "legend_border": "#D5D1C4",
    "legend_text": "#1F1F1F", "grid_alpha": 0.10,
}


# ---- Theme state ----
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
theme = st.session_state.theme


# =====================================================================
# STYLES
# =====================================================================
THEME_BLOCK = DARK_VARS if theme == "dark" else LIGHT_VARS

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');

{THEME_BLOCK}

/* ---- Fonts (avoid universal selector — preserves Material icons) ---- */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"] {{
    font-family: 'Inter', 'Noto Sans Thai', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}}
button, input, textarea, select {{ font-family: inherit; }}

/* ---- Main area ---- */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {{ background: var(--bg-page); }}
[data-testid="stMain"] {{ color: var(--text-primary); }}
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {{ color: var(--h1-color); }}

.block-container {{
    padding-top: 2.4rem;
    padding-bottom: 4rem;
    max-width: 780px;
}}

h1 {{
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
    margin: 0 0 .35rem 0 !important;
    line-height: 1.35 !important;
}}
.subtitle {{
    font-size: .9rem;
    color: var(--text-subtle);
    margin: 0 0 1.6rem 0;
    letter-spacing: .005em;
}}

/* ---- KaTeX inherits color → readable on any theme ---- */
[data-testid="stMain"] .katex,
[data-testid="stMain"] .katex * {{ color: inherit !important; }}
[data-testid="stMain"] .katex-display {{ margin: .35em 0 .55em 0 !important; }}

/* ---- Tags ---- */
.tag {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 5px;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .09em;
    margin: 0 0 14px 0;
    text-transform: uppercase;
}}
.tag-method     {{ background: var(--tag-method-bg); color: var(--tag-method-text); }}
.tag-general    {{ background: var(--tag-general-bg); color: var(--tag-general-text); }}
.tag-particular {{ background: var(--tag-particular-bg); color: var(--tag-particular-text); }}

/* ---- Section title ---- */
.sec-title {{
    font-size: 1rem;
    font-weight: 600;
    color: var(--section-title);
    margin: 4px 0 8px 0;
    letter-spacing: .005em;
}}
.sec-divider {{
    margin: 22px 0 14px 0;
    border: 0;
    border-top: 1px solid var(--divider-section);
}}

/* ---- Captions ---- */
[data-testid="stMain"] [data-testid="stCaptionContainer"] {{
    color: var(--text-secondary) !important;
    font-size: .87rem !important;
    line-height: 1.5 !important;
    margin-bottom: 2px !important;
}}

/* ---- Bordered containers ---- */
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--bg-box);
    border: 1px solid var(--border-box) !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
}}
.st-key-sol-general [data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--bg-general) !important;
    border: 1px solid var(--border-general) !important;
}}
.st-key-sol-particular [data-testid="stVerticalBlockBorderWrapper"] {{
    background: var(--bg-particular) !important;
    border: 1px solid var(--border-particular) !important;
}}

/* ---- Case selector ---- */
.case-selector {{
    display: flex; flex-direction: column; gap: 6px;
    margin: 10px 0 6px 0;
}}
.case-row {{
    display: flex; align-items: center; gap: 14px;
    padding: 9px 14px; border-radius: 8px;
    background: var(--bg-case);
    border: 1px solid var(--border-case);
    color: var(--text-case);
    font-size: .92rem;
}}
.case-row .case-marker {{
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--marker-case); flex-shrink: 0;
}}
.case-row .case-cond {{
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-style: italic;
    min-width: 56px; color: inherit;
}}
.case-row .case-name {{ color: inherit; }}
.case-row.chosen {{
    background: var(--bg-case-chosen);
    border-color: var(--border-case-chosen);
    color: var(--text-case-chosen);
}}
.case-row.chosen .case-marker {{ background: var(--marker-case-chosen); }}

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] {{
    background: var(--sidebar-bg);
    border-right: 1px solid var(--sidebar-border);
}}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {{
    color: var(--sidebar-text);
}}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {{
    font-size: .76rem !important;
    color: var(--sidebar-h) !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 700 !important;
    margin: .4rem 0 .35rem 0 !important;
}}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
    color: var(--sidebar-caption) !important;
    font-size: .82rem !important;
}}

/* Reference equation in sidebar */
.st-key-sidebar-eq {{
    margin-top: -.45rem;
    margin-bottom: .35rem;
}}
.st-key-sidebar-eq p {{
    font-size: .92rem;
    line-height: 1.35;
    color: var(--sidebar-text);
    margin-bottom: 0 !important;
}}

/* ---- Buttons (main area) ---- */
[data-testid="stMain"] .stButton > button {{
    font-weight: 500;
    border-radius: 8px;
    padding: .45rem .9rem;
    background: var(--btn-bg);
    color: var(--btn-text);
    border: 1px solid var(--btn-border);
}}
[data-testid="stMain"] .stButton > button:hover:not(:disabled) {{
    background: var(--btn-hover-bg);
    border-color: var(--btn-hover-border);
}}
[data-testid="stMain"] .stButton > button:disabled {{
    background: var(--btn-disabled-bg);
    color: var(--btn-disabled-text);
    border-color: var(--btn-disabled-border);
}}
[data-testid="stMain"] .stButton > button[kind="primary"] {{
    background: var(--btn-primary-bg);
    border-color: var(--btn-primary-border);
    color: #FFFFFF;
}}
[data-testid="stMain"] .stButton > button[kind="primary"]:hover:not(:disabled) {{
    background: var(--btn-primary-hover-bg);
}}

/* Hide Streamlit chrome */
footer {{ visibility: hidden; }}
#MainMenu {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# SOLVER  (cached for performance — repeated reruns with same input
#          return cached dict instead of recomputing)
# =====================================================================
@st.cache_data(show_spinner=False, max_entries=64)
def solve(a, b, y0, y1, has_init):
    """แก้สมการ y(n+2) + a·y(n+1) + b·y(n) = 0"""
    disc = a * a - 4 * b
    out = {
        "a": a, "b": b, "y0": y0, "y1": y1,
        "discriminant": disc, "has_init": has_init,
    }

    if disc > EPS:
        sqrt_d = math.sqrt(disc)
        r1 = (-a + sqrt_d) / 2
        r2 = (-a - sqrt_d) / 2
        out.update({"case": "real_distinct", "r1": r1, "r2": r2, "sqrt_d": sqrt_d})
        if has_init:
            denom = r1 - r2
            c1 = (y1 - r2 * y0) / denom
            c2 = y0 - c1
            out.update({"c1": c1, "c2": c2})

    elif disc < -EPS:
        p = -a / 2
        q = math.sqrt(-disc) / 2
        rho = math.sqrt(p * p + q * q)
        theta = math.atan2(q, p)
        out.update({"case": "complex", "p": p, "q": q, "rho": rho, "theta": theta})
        if has_init:
            A = y0
            sin_t = math.sin(theta)
            if abs(sin_t) < 1e-14:
                B = 0.0
            else:
                B = (y1 / rho - A * math.cos(theta)) / sin_t
            out.update({"A": A, "B": B})

    else:
        r = -a / 2
        out.update({"case": "repeated", "r": r})
        if has_init:
            c1 = y0
            if abs(r) < 1e-14:
                c2 = 0.0
            else:
                c2 = y1 / r - c1
            out.update({"c1": c1, "c2": c2})

    return out


def y_analytical(res, n):
    case = res["case"]
    if case == "real_distinct":
        return res["c1"] * res["r1"] ** n + res["c2"] * res["r2"] ** n
    if case == "complex":
        return res["rho"] ** n * (
            res["A"] * math.cos(n * res["theta"])
            + res["B"] * math.sin(n * res["theta"])
        )
    return (res["c1"] + res["c2"] * n) * res["r"] ** n


def y_iterative(a, b, y0, y1, n_max):
    seq = [y0, y1]
    for _ in range(n_max - 1):
        seq.append(-a * seq[-1] - b * seq[-2])
    return seq


# =====================================================================
# LATEX HELPERS
# =====================================================================
def clean(x):
    if isinstance(x, (int, np.integer)):
        return int(x)
    if abs(x) < 1e-12:
        return 0.0
    return float(x)


def fmt(x, decimals=4):
    x = clean(x)
    if isinstance(x, int):
        return str(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.{decimals}g}"


def latex_term(coef, var):
    coef = clean(coef)
    if abs(coef) < 1e-12:
        return ""
    if abs(coef - 1) < 1e-12:
        return f" + {var}"
    if abs(coef + 1) < 1e-12:
        return f" - {var}"
    if coef > 0:
        return f" + {fmt(coef)}\\,{var}"
    return f" - {fmt(-coef)}\\,{var}"


def latex_const(c):
    c = clean(c)
    if abs(c) < 1e-12:
        return ""
    return f" + {fmt(c)}" if c > 0 else f" - {fmt(-c)}"


def std_eq_latex(a, b):
    return f"y(n+2){latex_term(a, 'y(n+1)')}{latex_term(b, 'y(n)')} = 0"


def char_eq_latex(a, b):
    return f"r^2{latex_term(a, 'r')}{latex_const(b)} = 0"


def power_latex(base, var="n"):
    s = fmt(base)
    if s.startswith("-") or "." in s:
        return f"({s})^{{{var}}}"
    return f"{s}^{{{var}}}"


# =====================================================================
# CASE SELECTOR HTML
# =====================================================================
def case_selector_html(chosen):
    rows = [
        ("real_distinct", "Δ > 0", "รากจำนวนจริงต่างกันสองค่า"),
        ("complex",       "Δ < 0", "รากเชิงซ้อนคอนจูเกต"),
        ("repeated",      "Δ = 0", "รากซ้ำ (รากคู่จำนวนจริง)"),
    ]
    parts = ['<div class="case-selector">']
    for key, cond, name in rows:
        cls = "case-row chosen" if key == chosen else "case-row"
        parts.append(
            f'<div class="{cls}">'
            f'<span class="case-marker"></span>'
            f'<span class="case-cond">{cond}</span>'
            f'<span class="case-name">{name}</span>'
            f"</div>"
        )
    parts.append("</div>")
    return "".join(parts)


# =====================================================================
# BUILD ATOMS
#   Each atom is a dict with one of these key sets:
#     {"section": str}                        — group heading
#     {"label": str, "html": str}             — HTML inside method box
#     {"label": str|None, "latex": str|None}  — caption + optional LaTeX
#     {"solution": "general"|"particular", "latex": str}  — own box
# =====================================================================
def build_atoms(res):
    a, b, y0, y1 = res["a"], res["b"], res["y0"], res["y1"]
    case = res["case"]
    disc = res["discriminant"]
    has_init = res["has_init"]

    atoms = []

    # ----- ส่วนที่ 1 -----
    atoms += [
        {"section": "ส่วนที่ 1 — การแปลงสมการ"},
        {"label": "รูปแบบมาตรฐานของสมการเชิงผลต่างอันดับสองแบบเอกพันธ์",
         "latex": r"y(n+2) + a\,y(n+1) + b\,y(n) = 0"},
        {"label": f"แทนค่าสัมประสิทธิ์  $a = {fmt(a)}$,  $b = {fmt(b)}$",
         "latex": std_eq_latex(a, b)},
        {"label": r"สมมติคำตอบในรูป $y(n) = r^{n}$ แล้วแทนกลับเข้าสมการ",
         "latex": f"r^{{n+2}}{latex_term(a, 'r^{n+1}')}{latex_term(b, 'r^{n}')} = 0"},
        {"label": r"หารทั้งสองข้างด้วย $r^{n}$ (เนื่องจาก $r \neq 0$)",
         "latex": r"r^{2} + a\,r + b = 0"},
        {"label": "ได้สมการลักษณะเฉพาะ (Characteristic Equation)",
         "latex": char_eq_latex(a, b)},
    ]

    # ----- ส่วนที่ 2 -----
    atoms += [
        {"section": "ส่วนที่ 2 — แยกกรณีจากค่า Discriminant"},
        {"label": r"คำนวณค่า $\Delta = a^{2} - 4b$",
         "latex": f"\\Delta = ({fmt(a)})^{{2}} - 4({fmt(b)}) = {fmt(disc)}"},
        {"label": "เปรียบเทียบค่า Δ กับ 0 เพื่อเลือกใช้สูตรของแต่ละกรณี",
         "html": case_selector_html(case)},
    ]

    if case == "real_distinct":
        r1, r2, sqrt_d = res["r1"], res["r2"], res["sqrt_d"]
        atoms += [
            {"label": r"กรณีที่เลือก: ใช้สูตร $r_{1,2} = \dfrac{-a \pm \sqrt{\Delta}}{2}$",
             "latex": f"r_{{1,2}} = \\frac{{{fmt(-a)} \\pm \\sqrt{{{fmt(disc)}}}}}{{2}} "
                      f"= \\frac{{{fmt(-a)} \\pm {fmt(sqrt_d)}}}{{2}}"},
            {"label": "แยกหารากทั้งสอง",
             "latex": f"r_{{1}} = {fmt(r1)}, \\qquad r_{{2}} = {fmt(r2)}"},
            {"label": r"คำตอบมูลฐานคือ $y_{1} = r_{1}^{n}$ และ $y_{2} = r_{2}^{n}$  "
                      r"&nbsp;จึงได้คำตอบทั่วไปเป็นผลรวมเชิงเส้น",
             "latex": None},
            {"solution": "general",
             "latex": f"y(n) = c_{{1}}\\,{power_latex(r1)} + c_{{2}}\\,{power_latex(r2)}"},
        ]

    elif case == "complex":
        p, q, rho, theta = res["p"], res["q"], res["rho"], res["theta"]
        atoms += [
            {"label": r"กรณีที่เลือก: ราก $\lambda = p \pm qi$  &nbsp;โดย",
             "latex": f"p = -\\tfrac{{a}}{{2}} = {fmt(p)}, \\qquad "
                      f"q = \\tfrac{{\\sqrt{{-\\Delta}}}}{{2}} = "
                      f"\\tfrac{{\\sqrt{{{fmt(-disc)}}}}}{{2}} = {fmt(q)}"},
            {"label": "เขียนรากในรูปคอนจูเกต",
             "latex": f"\\lambda_{{1}} = {fmt(p)} + {fmt(q)}\\,i, \\qquad "
                      f"\\lambda_{{2}} = {fmt(p)} - {fmt(q)}\\,i"},
            {"label": r"แปลงเป็นรูปขั้ว: หาขนาด $\rho$ และมุม $\theta$",
             "latex": f"\\rho = \\sqrt{{p^{{2}} + q^{{2}}}} = \\sqrt{{{fmt(p*p + q*q)}}} "
                      f"= {fmt(rho)}, \\qquad "
                      f"\\theta = \\arctan\\!\\bigl(\\tfrac{{q}}{{p}}\\bigr) "
                      f"= {fmt(theta)}\\ \\text{{rad}}"},
            {"label": r"คำตอบมูลฐานคือ $\rho^{n}\cos(n\theta)$ และ $\rho^{n}\sin(n\theta)$  "
                      r"&nbsp;จึงได้คำตอบทั่วไป",
             "latex": None},
            {"solution": "general",
             "latex": f"y(n) = ({fmt(rho)})^{{n}}\\,\\bigl(\\,A\\cos(n\\cdot{fmt(theta)}) "
                      f"+ B\\sin(n\\cdot{fmt(theta)})\\,\\bigr)"},
        ]

    else:  # repeated
        r = res["r"]
        atoms += [
            {"label": r"กรณีที่เลือก: รากซ้ำ ใช้สูตร $r = -\dfrac{a}{2}$",
             "latex": f"r = -\\tfrac{{{fmt(a)}}}{{2}} = {fmt(r)}\\quad(\\text{{รากซ้ำ}})"},
            {"label": r"กรณีรากซ้ำ คำตอบมูลฐานตัวที่สองต้องคูณด้วย $n$ "
                      r"&nbsp;เพื่อให้เป็นอิสระเชิงเส้นต่อกัน",
             "latex": f"y_{{1}} = {power_latex(r)}, \\qquad "
                      f"y_{{2}} = n\\cdot {power_latex(r)}"},
            {"solution": "general",
             "latex": f"y(n) = (c_{{1}} + c_{{2}}\\,n)\\,{power_latex(r)}"},
        ]

    # ----- ส่วนที่ 3 -----
    if has_init:
        atoms.append({"section": "ส่วนที่ 3 — หาคำตอบเฉพาะจากเงื่อนไขเริ่มต้น"})

        if case == "real_distinct":
            r1, r2, c1, c2 = res["r1"], res["r2"], res["c1"], res["c2"]
            atoms += [
                {"label": f"แทน $n=0$ ในคำตอบทั่วไป  ($y(0) = {fmt(y0)}$)",
                 "latex": f"c_{{1}}\\,({fmt(r1)})^{{0}} + c_{{2}}\\,({fmt(r2)})^{{0}} "
                          f"= c_{{1}} + c_{{2}} = {fmt(y0)} \\quad\\text{{...(1)}}"},
                {"label": f"แทน $n=1$ ในคำตอบทั่วไป  ($y(1) = {fmt(y1)}$)",
                 "latex": f"{fmt(r1)}\\,c_{{1}} + {fmt(r2)}\\,c_{{2}} = {fmt(y1)} "
                          f"\\quad\\text{{...(2)}}"},
                {"label": r"จาก (1):  $c_{2} = y_{0} - c_{1}$  &nbsp;แทนใน (2)",
                 "latex": f"{fmt(r1)}\\,c_{{1}} + {fmt(r2)}\\bigl({fmt(y0)} - c_{{1}}\\bigr) "
                          f"= {fmt(y1)}"},
                {"label": r"จัดรูปและแก้สมการหา $c_{1}$",
                 "latex": f"({fmt(r1 - r2)})\\,c_{{1}} = {fmt(y1 - r2 * y0)} "
                          f"\\;\\Longrightarrow\\; c_{{1}} = {fmt(c1)}"},
                {"label": r"แทนค่ากลับใน (1) เพื่อหา $c_{2}$",
                 "latex": f"c_{{2}} = {fmt(y0)} - {fmt(c1)} = {fmt(c2)}"},
                {"solution": "particular",
                 "latex": f"y(n) = ({fmt(c1)})\\,{power_latex(r1)} "
                          f"+ ({fmt(c2)})\\,{power_latex(r2)}"},
            ]

        elif case == "complex":
            rho, theta = res["rho"], res["theta"]
            A, B = res["A"], res["B"]
            atoms += [
                {"label": f"แทน $n=0$ ในคำตอบทั่วไป  ($y(0) = {fmt(y0)}$)",
                 "latex": f"\\rho^{{0}}\\,(A\\cos 0 + B\\sin 0) = A = {fmt(y0)} "
                          f"\\;\\Longrightarrow\\; A = {fmt(A)}"},
                {"label": f"แทน $n=1$ ในคำตอบทั่วไป  ($y(1) = {fmt(y1)}$)",
                 "latex": f"{fmt(rho)}\\,(A\\cos\\theta + B\\sin\\theta) = {fmt(y1)}"},
                {"label": r"แก้สมการหา $B$",
                 "latex": f"B = \\dfrac{{y_{{1}}/\\rho - A\\cos\\theta}}{{\\sin\\theta}} "
                          f"= \\dfrac{{{fmt(y1/rho if rho else 0)} "
                          f"- ({fmt(A)})({fmt(math.cos(theta))})}}"
                          f"{{{fmt(math.sin(theta))}}} = {fmt(B)}"},
                {"solution": "particular",
                 "latex": f"y(n) = ({fmt(rho)})^{{n}}\\,\\bigl("
                          f"({fmt(A)})\\cos(n\\cdot{fmt(theta)}) "
                          f"+ ({fmt(B)})\\sin(n\\cdot{fmt(theta)})\\bigr)"},
            ]

        else:  # repeated
            r, c1, c2 = res["r"], res["c1"], res["c2"]
            atoms += [
                {"label": f"แทน $n=0$ ในคำตอบทั่วไป  ($y(0) = {fmt(y0)}$)",
                 "latex": f"(c_{{1}} + 0)\\,r^{{0}} = c_{{1}} = {fmt(y0)} "
                          f"\\;\\Longrightarrow\\; c_{{1}} = {fmt(c1)}"},
                {"label": f"แทน $n=1$ ในคำตอบทั่วไป  ($y(1) = {fmt(y1)}$)",
                 "latex": f"(c_{{1}} + c_{{2}})\\,r = {fmt(y1)}"},
                {"label": r"แก้สมการหา $c_{2}$",
                 "latex": f"c_{{2}} = \\dfrac{{y_{{1}}}}{{r}} - c_{{1}} "
                          f"= \\dfrac{{{fmt(y1)}}}{{{fmt(r)}}} - {fmt(c1)} = {fmt(c2)}"},
                {"solution": "particular",
                 "latex": f"y(n) = \\bigl({fmt(c1)} + ({fmt(c2)})\\,n\\bigr)\\,{power_latex(r)}"},
            ]

    return atoms


def count_steps(atoms):
    return sum(1 for a in atoms if any(k in a for k in ("latex", "html", "solution")))


# =====================================================================
# RENDERERS
# =====================================================================
def render_step_item(item):
    if "html" in item:
        if item.get("label"):
            st.caption(item["label"])
        st.markdown(item["html"], unsafe_allow_html=True)
        return
    if item.get("label"):
        st.caption(item["label"])
    if item.get("latex") is not None:
        st.latex(item["latex"])


def render_method_box(groups):
    if not any(items for _, items in groups):
        return
    with st.container(border=True):
        st.markdown('<span class="tag tag-method">วิธีทำ</span>',
                    unsafe_allow_html=True)
        first = True
        for sec_title, items in groups:
            if not items:
                continue
            if not first:
                st.markdown('<hr class="sec-divider">', unsafe_allow_html=True)
            if sec_title:
                st.markdown(f'<div class="sec-title">{sec_title}</div>',
                            unsafe_allow_html=True)
            for item in items:
                render_step_item(item)
            first = False


def render_solution_box(kind, latex):
    label = "คำตอบทั่วไป" if kind == "general" else "คำตอบเฉพาะ"
    with st.container(border=True, key=f"sol-{kind}"):
        st.markdown(f'<span class="tag tag-{kind}">{label}</span>',
                    unsafe_allow_html=True)
        st.latex(latex)


def split_into_blocks(atoms, max_step=None):
    blocks = []
    cur_method = None
    cur_section = None
    cur_items = []
    step_count = 0

    def flush_section():
        nonlocal cur_method, cur_items
        if cur_items:
            if cur_method is None:
                cur_method = {"type": "method", "groups": []}
                blocks.append(cur_method)
            cur_method["groups"].append((cur_section, cur_items))
            cur_items = []

    def end_method():
        nonlocal cur_method, cur_section
        flush_section()
        cur_method = None
        cur_section = None

    for atom in atoms:
        if "section" in atom:
            flush_section()
            cur_section = atom["section"]
        elif "solution" in atom:
            step_count += 1
            if max_step is not None and step_count > max_step:
                break
            end_method()
            blocks.append({"type": "solution",
                           "kind": atom["solution"],
                           "latex": atom["latex"]})
        elif "latex" in atom or "html" in atom:
            step_count += 1
            if max_step is not None and step_count > max_step:
                break
            cur_items.append(atom)
    flush_section()
    return blocks


def render_blocks(blocks):
    for block in blocks:
        if block["type"] == "method":
            render_method_box(block["groups"])
        else:
            render_solution_box(block["kind"], block["latex"])


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown("### ค่าที่ป้อน")
    # Reference equation: plain markdown so KaTeX can render the LaTeX.
    # (BUG FIX: previously wrapped in <div> which prevented KaTeX rendering.)
    with st.container(key="sidebar-eq"):
        st.markdown(r"$y(n{+}2) + a\,y(n{+}1) + b\,y(n) = 0$")

    col_a, col_b = st.columns(2)
    a = col_a.number_input("a", value=0.0, step=1.0, format="%g")
    b = col_b.number_input("b", value=0.0, step=1.0, format="%g")

    st.divider()

    has_init = st.checkbox("ใช้เงื่อนไขเริ่มต้น (เพื่อหาคำตอบเฉพาะ)", value=True)
    if has_init:
        col_y0, col_y1 = st.columns(2)
        y0 = col_y0.number_input("y(0)", value=0.0, step=1.0, format="%g")
        y1 = col_y1.number_input("y(1)", value=0.0, step=1.0, format="%g")
    else:
        y0, y1 = 0.0, 0.0

    st.divider()

    st.markdown("### โหมดการแสดงผล")
    mode = st.radio(
        "mode",
        ["แสดงทีละขั้น", "แสดงผลทันที"],
        index=0,
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### กราฟ")
    show_graph = st.checkbox("แสดงกราฟลำดับ y(n)", value=False)
    n_max = st.slider("ช่วง n", 5, 30, 15) if show_graph else 15

    st.divider()

    st.markdown("### ธีม")
    is_light_choice = st.toggle(
        "โหมดสว่าง",
        value=(theme == "light"),
        key="theme_widget",
    )
    new_theme = "light" if is_light_choice else "dark"
    if new_theme != theme:
        st.session_state.theme = new_theme
        st.rerun()


# =====================================================================
# MAIN
# =====================================================================
st.markdown("# สมการเชิงผลต่างอันดับสองแบบเอกพันธ์")
st.markdown(
    '<p class="subtitle">Second-Order Homogeneous Difference Equation</p>',
    unsafe_allow_html=True,
)

# Welcome state: when both a and b are zero, the equation is degenerate.
should_compute = not (abs(a) < EPS and abs(b) < EPS)

if not should_compute:
    st.markdown(
        '<p style="color:var(--welcome-text); margin-top:1rem; font-size:.95rem;">'
        'ป้อนค่า <code style="background:var(--code-bg); padding:1px 6px; '
        'border-radius:4px; color:var(--code-text);">a</code> '
        'และ <code style="background:var(--code-bg); padding:1px 6px; '
        'border-radius:4px; color:var(--code-text);">b</code> '
        'ที่แถบด้านข้างเพื่อเริ่มคำนวณ'
        '</p>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    res = solve(a, b, y0, y1, has_init)
    atoms = build_atoms(res)
    total = count_steps(atoms)
except Exception as e:  # noqa: BLE001
    st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
    st.stop()

# ---- session state for step-by-step ----
if "step" not in st.session_state:
    st.session_state.step = 1
if "last_input" not in st.session_state:
    st.session_state.last_input = None

current_input = (a, b, y0, y1, has_init)
if st.session_state.last_input != current_input:
    st.session_state.step = 1
    st.session_state.last_input = current_input

is_step_mode = (mode == "แสดงทีละขั้น")

if is_step_mode:
    cur = min(st.session_state.step, total)
    blocks = split_into_blocks(atoms, max_step=cur)
    render_blocks(blocks)

    st.caption(f"ขั้นที่ {cur} จาก {total}")
    c1, c2, c3 = st.columns(3)
    if c1.button("ก่อนหน้า", disabled=(cur <= 1), use_container_width=True):
        st.session_state.step = max(1, cur - 1)
        st.rerun()
    if c2.button("ถัดไป", type="primary",
                 disabled=(cur >= total), use_container_width=True):
        st.session_state.step = min(total, cur + 1)
        st.rerun()
    if c3.button("เริ่มใหม่", disabled=(cur == 1),
                 use_container_width=True):
        st.session_state.step = 1
        st.rerun()
else:
    blocks = split_into_blocks(atoms)
    render_blocks(blocks)


# =====================================================================
# GRAPH + TABLE
# =====================================================================
if show_graph and has_init:
    st.markdown("")
    st.markdown("### กราฟลำดับ y(n)")
    st.caption("เปรียบเทียบ 2 วิธี: คำนวณซ้ำตามสมการ (iterative) กับสูตรปิดที่ได้ (analytical)")

    g = DARK_GRAPH if theme == "dark" else LIGHT_GRAPH
    ns = list(range(n_max + 1))
    ys_iter = y_iterative(a, b, y0, y1, n_max)
    ys_anal = [y_analytical(res, n) for n in ns]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(g["fig_bg"])
    ax.set_facecolor(g["axis_bg"])
    ax.plot(ns, ys_iter, "o-", label="iterative",
            color=g["iter"], markersize=6, linewidth=1.4)
    ax.plot(ns, ys_anal, "x", label="analytical",
            color=g["anal"], markersize=10, markeredgewidth=2)
    ax.axhline(0, color=g["zero"], linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color(g["spine"])
    ax.tick_params(colors=g["tick"])
    ax.xaxis.label.set_color(g["label"])
    ax.yaxis.label.set_color(g["label"])
    ax.set_xlabel("n")
    ax.set_ylabel("y(n)")
    ax.legend(loc="best", framealpha=0.9, facecolor=g["legend_bg"],
              edgecolor=g["legend_border"], labelcolor=g["legend_text"])
    ax.grid(True, alpha=g["grid_alpha"], color=g["grid"])
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)  # free matplotlib resources

    with st.expander("ตารางค่า y(n)"):
        df = pd.DataFrame({
            "n": ns,
            "iterative": [round(v, 6) for v in ys_iter],
            "analytical": [round(v, 6) for v in ys_anal],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

elif show_graph and not has_init:
    st.info("ต้องเปิด 'ใช้เงื่อนไขเริ่มต้น' จึงจะแสดงกราฟได้")
