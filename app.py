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


EPS = 1e-9   # เกณฑ์การปัดเศษ/แยกกรณี


# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="สมการเชิงผลต่างอันดับสองแบบเอกพันธ์",
    layout="centered",
    initial_sidebar_state="expanded",
)


# =====================================================================
# STYLES — main area dark, sidebar light, icon fonts preserved
# =====================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans+Thai:wght@400;500;600;700&display=swap');

/* ---- Fonts (do NOT use universal selector, would break icon fonts) ---- */
html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stSidebar"] {
    font-family: 'Inter', 'Noto Sans Thai', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
}
button, input, textarea, select { font-family: inherit; }

/* ---- Main area: dark ---- */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stHeader"] {
    background: #1B1B1F;
}
[data-testid="stMain"] { color: #E6E6E8; }
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 { color: #F2F2F4; }

.block-container {
    padding-top: 2.4rem;
    padding-bottom: 4rem;
    max-width: 780px;
}

h1 {
    font-size: 1.55rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
    margin: 0 0 .35rem 0 !important;
    line-height: 1.35 !important;
}
.subtitle {
    font-size: .9rem;
    color: #9C9CA0;
    margin: 0 0 1.6rem 0;
    letter-spacing: .005em;
}

/* ---- KaTeX inherits text color so it shows on dark bg ---- */
[data-testid="stMain"] .katex,
[data-testid="stMain"] .katex * { color: inherit !important; }
[data-testid="stMain"] .katex-display { margin: .35em 0 .55em 0 !important; }

/* ---- Tags (small uppercase header inside each box) ---- */
.tag {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 5px;
    font-size: .68rem;
    font-weight: 700;
    letter-spacing: .09em;
    margin: 0 0 14px 0;
    text-transform: uppercase;
}
.tag-method     { background: #2C2C32; color: #B7B7BD; }
.tag-general    { background: #1F3252; color: #9DBDE6; }
.tag-particular { background: #233D1B; color: #A6CD83; }

/* ---- Section title inside method box ---- */
.sec-title {
    font-size: 1rem;
    font-weight: 600;
    color: #F0F0F2;
    margin: 4px 0 8px 0;
    letter-spacing: .005em;
}
.sec-divider {
    margin: 22px 0 14px 0;
    border: 0;
    border-top: 1px solid #2D2D33;
}

/* ---- Captions (the small label above each equation) ---- */
[data-testid="stMain"] [data-testid="stCaptionContainer"] {
    color: #ABABB1 !important;
    font-size: .87rem !important;
    line-height: 1.5 !important;
    margin-bottom: 2px !important;
}

/* ---- Bordered containers (st.container border=True) ---- */
[data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] {
    background: #232328;
    border: 1px solid #2E2E34 !important;
    border-radius: 12px !important;
    padding: 18px 20px !important;
}

/* Tinted backgrounds for the two solution boxes via container key */
.st-key-sol-general [data-testid="stVerticalBlockBorderWrapper"] {
    background: #1A2538 !important;
    border: 1px solid #2F4D7A !important;
}
.st-key-sol-particular [data-testid="stVerticalBlockBorderWrapper"] {
    background: #1B2A18 !important;
    border: 1px solid #3F6628 !important;
}

/* ---- Case selector (the 3-row visual in section 2) ---- */
.case-selector {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin: 10px 0 6px 0;
}
.case-row {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 9px 14px;
    border-radius: 8px;
    background: #20202560;
    border: 1px solid transparent;
    color: #76767B;
    font-size: .92rem;
}
.case-row .case-marker {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4A4A50;
    flex-shrink: 0;
}
.case-row .case-cond {
    font-family: 'Cambria Math', 'Times New Roman', serif;
    font-style: italic;
    min-width: 56px;
    color: inherit;
}
.case-row .case-name { color: inherit; }

.case-row.chosen {
    background: #2A3344;
    border-color: #4A6A95;
    color: #E6EAF2;
}
.case-row.chosen .case-marker { background: #6FA3E5; }

/* ---- Sidebar: light, kept as control panel ---- */
section[data-testid="stSidebar"] {
    background: #F7F7F5;
    border-right: 1px solid #E6E6E2;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4 {
    font-size: .76rem !important;
    color: #5A5A58 !important;
    text-transform: uppercase;
    letter-spacing: .08em;
    font-weight: 700 !important;
    margin: .4rem 0 .35rem 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #6E6E6B !important;
    font-size: .82rem !important;
}

/* ---- Buttons (main area) ---- */
[data-testid="stMain"] .stButton > button {
    font-weight: 500;
    border-radius: 8px;
    padding: .45rem .9rem;
    background: #2C2C32;
    color: #E6E6E8;
    border: 1px solid #3A3A40;
}
[data-testid="stMain"] .stButton > button:hover:not(:disabled) {
    background: #34343A;
    border-color: #4A4A50;
    color: #FFFFFF;
}
[data-testid="stMain"] .stButton > button:disabled {
    background: #1F1F23;
    color: #5A5A60;
    border-color: #28282E;
}
[data-testid="stMain"] .stButton > button[kind="primary"] {
    background: #4F7BC4;
    border-color: #5B8FE6;
    color: #FFFFFF;
}
[data-testid="stMain"] .stButton > button[kind="primary"]:hover:not(:disabled) {
    background: #5B8FE6;
}

/* Hide Streamlit chrome */
footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""",
    unsafe_allow_html=True,
)


# =====================================================================
# SOLVER
# =====================================================================
def solve(a, b, y0, y1, has_init):
    """แก้สมการ y(n+2) + a·y(n+1) + b·y(n) = 0
       คืน dict ที่บรรยายราก/สัมประสิทธิ์ของคำตอบ"""
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
                # รากซ้ำ = 0 เกิดเมื่อ a=b=0 ซึ่งดักไว้ที่หน้าโปรแกรมแล้ว
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
# LATEX FORMATTING HELPERS
# =====================================================================
def clean(x):
    """ปัดเศษค่าที่ใกล้ศูนย์มากๆ ให้เป็น 0 (กันเลขทศนิยมเล็กจิ๋วโผล่ในผลลัพธ์)"""
    if isinstance(x, (int, np.integer)):
        return int(x)
    if abs(x) < 1e-12:
        return 0.0
    return float(x)


def fmt(x, decimals=4):
    """แสดงตัวเลข: integer ถ้าใกล้จำนวนเต็ม, มิฉะนั้น %g"""
    x = clean(x)
    if isinstance(x, int):
        return str(x)
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return f"{x:.{decimals}g}"


def latex_term(coef, var):
    """' + 5\\,y(n+1)' / ' - y(n+1)' / '' (สำหรับเขียนสมการต่อเนื่อง)"""
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
    """'3^{n}' หรือ '(-0.618)^{n}'"""
    s = fmt(base)
    if s.startswith("-") or "." in s:
        return f"({s})^{{{var}}}"
    return f"{s}^{{{var}}}"


# =====================================================================
# CASE SELECTOR HTML  (visual highlight in section 2)
# =====================================================================
def case_selector_html(chosen):
    """สร้าง HTML แสดงทั้ง 3 กรณี โดยกรณีที่เลือก = ไฮไลต์"""
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
# BUILD ATOMS — ลำดับเนื้อหาในรูปแบบมาตรฐาน
#   atoms ประกอบด้วย dict ที่มีคีย์ใดคีย์หนึ่ง:
#     - "section": ชื่อหัวข้อใหญ่ (ใช้แบ่งกลุ่ม ไม่นับเป็น step)
#     - "html":    HTML แทรกในกล่องวิธีทำ (นับเป็น step)
#     - "label" + "latex":  ขั้นปกติ (นับเป็น step)
#     - "solution": "general" / "particular"  → กรอบของตัวเอง (นับเป็น step)
# =====================================================================
def build_atoms(res):
    a, b, y0, y1 = res["a"], res["b"], res["y0"], res["y1"]
    case = res["case"]
    disc = res["discriminant"]
    has_init = res["has_init"]

    atoms = []

    # ----- ส่วนที่ 1 : การแปลงสมการ -----
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

    # ----- ส่วนที่ 2 : แยกกรณี → คำตอบทั่วไป -----
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
            {"label": r"แยกหารากทั้งสอง",
             "latex": f"r_{{1}} = {fmt(r1)}, \\qquad r_{{2}} = {fmt(r2)}"},
            {"label": r"คำตอบมูลฐานคือ $y_{1} = r_{1}^{n}$ และ $y_{2} = r_{2}^{n}$  "
                      r"&nbsp;จึงได้คำตอบทั่วไปเป็นผลรวมเชิงเส้น",
             "latex": "—skip—"},
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
             "latex": "—skip—"},
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

    # ----- ส่วนที่ 3 : คำตอบเฉพาะ (เฉพาะเมื่อมีเงื่อนไขเริ่มต้น) -----
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
    """นับจำนวน step (ไม่รวม section header)"""
    return sum(1 for a in atoms if "latex" in a or "html" in a or "solution" in a)


# =====================================================================
# RENDERERS
# =====================================================================
def render_step_item(item):
    """ขั้นปกติ: caption + LaTeX (หรือ html ถ้าเป็น html atom)"""
    if "html" in item:
        if item.get("label"):
            st.caption(item["label"])
        st.markdown(item["html"], unsafe_allow_html=True)
        return

    if item.get("label"):
        st.caption(item["label"])
    if item.get("latex") and item["latex"] != "—skip—":
        st.latex(item["latex"])


def render_method_box(groups):
    """กล่อง 'วิธีทำ' รวม section ทั้งหมด"""
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
    """แปลง atoms เป็นลำดับ block สำหรับ render
       block: {"type":"method", "groups":[(section_title, items),...]}
              {"type":"solution", "kind":..., "latex":...}
       max_step จำกัดจำนวน step ที่เปิดให้เห็น (None = ทั้งหมด)"""
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
    st.caption(r"$y(n{+}2) + a\,y(n{+}1) + b\,y(n) = 0$")

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


# =====================================================================
# MAIN
# =====================================================================
st.markdown("# สมการเชิงผลต่างอันดับสองแบบเอกพันธ์")
st.markdown(
    '<p class="subtitle">Second-Order Homogeneous Difference Equation</p>',
    unsafe_allow_html=True,
)

# ---- เงื่อนไขเริ่มทำงาน: a หรือ b ต้องไม่เป็นศูนย์ทั้งคู่ ----
should_compute = not (abs(a) < EPS and abs(b) < EPS)

if not should_compute:
    # ----- หน้าเริ่มต้น (ไม่แสดงเนื้อหาเชิงคำนวณ) -----
    st.markdown(
        '<p style="color:#8B8B91; margin-top:1rem; font-size:.95rem;">'
        "ป้อนค่า <code style=\"background:#2C2C32;padding:1px 6px;border-radius:4px;color:#E6E6E8;\">a</code> "
        "และ <code style=\"background:#2C2C32;padding:1px 6px;border-radius:4px;color:#E6E6E8;\">b</code> "
        "ที่แถบด้านข้างเพื่อเริ่มคำนวณ"
        "</p>",
        unsafe_allow_html=True,
    )
    st.stop()

# ---- คำนวณ + เตรียม atoms ----
try:
    res = solve(a, b, y0, y1, has_init)
    atoms = build_atoms(res)
    total = count_steps(atoms)
except Exception as e:  # noqa: BLE001
    st.error(f"เกิดข้อผิดพลาดในการคำนวณ: {e}")
    st.stop()

# ---- session state สำหรับโหมด step-by-step ----
if "step" not in st.session_state:
    st.session_state.step = 1
if "last_input" not in st.session_state:
    st.session_state.last_input = None

current_input = (a, b, y0, y1, has_init)
if st.session_state.last_input != current_input:
    # ค่าเปลี่ยน → รีเซ็ตขั้น
    st.session_state.step = 1
    st.session_state.last_input = current_input

is_step_mode = (mode == "แสดงทีละขั้น")

if is_step_mode:
    cur = min(st.session_state.step, total)
    blocks = split_into_blocks(atoms, max_step=cur)
    render_blocks(blocks)

    # ---- ปุ่มควบคุมอยู่ใต้กรอบล่าสุด ----
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
# GRAPH + TABLE — เปรียบเทียบ iterative กับ analytical (ตรวจความถูกต้อง)
# =====================================================================
if show_graph and has_init:
    st.markdown("")
    st.markdown("### กราฟลำดับ y(n)")
    st.caption("เปรียบเทียบ 2 วิธี: คำนวณซ้ำตามสมการ (iterative) กับสูตรปิดที่ได้ (analytical)")

    ns = list(range(n_max + 1))
    ys_iter = y_iterative(a, b, y0, y1, n_max)
    ys_anal = [y_analytical(res, n) for n in ns]

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor("#232328")
    ax.set_facecolor("#1B1B1F")
    ax.plot(ns, ys_iter, "o-", label="iterative",
            color="#7AAEEC", markersize=6, linewidth=1.4)
    ax.plot(ns, ys_anal, "x", label="analytical",
            color="#E0934A", markersize=10, markeredgewidth=2)
    ax.axhline(0, color="#555", linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_color("#3A3A40")
    ax.tick_params(colors="#B7B7BD")
    ax.xaxis.label.set_color("#D6D6D9")
    ax.yaxis.label.set_color("#D6D6D9")
    ax.set_xlabel("n")
    ax.set_ylabel("y(n)")
    leg = ax.legend(loc="best", framealpha=0.9, facecolor="#2C2C32",
                    edgecolor="#3A3A40", labelcolor="#E6E6E8")
    ax.grid(True, alpha=0.18, color="#FFFFFF")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)

    with st.expander("ตารางค่า y(n)"):
        df = pd.DataFrame({
            "n": ns,
            "iterative": [round(v, 6) for v in ys_iter],
            "analytical": [round(v, 6) for v in ys_anal],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)

elif show_graph and not has_init:
    st.info("ต้องเปิด 'ใช้เงื่อนไขเริ่มต้น' จึงจะแสดงกราฟได้")
