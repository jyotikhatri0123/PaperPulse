
"""PaperPulse - exam paper pattern miner."""
import io
import re
import shutil

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "PaperPulse"
TAGLINE = "Find the chapters and questions that exam papers keep coming back to."

PRESETS = {
    "Nepal Grade 12 Mathematics": """Matrices and Determinants: matrix, determinant, cofactor, adjoint, adj, cramer, symmetric, associative, unique solution, linear equations
Input-Output Analysis: leontief, technology matrix, hawkins, viable, final demand, input-output, industries
Linear Programming: simplex, slack, surplus, objective function, feasible, pivot
Differentiation Applications: maximizes, maximum, minimum, revenue, demand equation, marginal, profit
Integration: integral, integrate, consumer surplus, producer surplus
Differential Equations: differential equation, growth rate, population
Mathematics of Finance: price reduces, depreciation, scrap value, compound, interest, annuity, yearly
Investment Appraisal: net present value, npv, present value, future returns
Probability: probability, dependent events, independent events, bayes
Statistics: regression, correlation, median, quartile, skewness, coefficient, deviation""",
    "CBSE Class 12 Mathematics": """Relations and Functions: relation, function, equivalence, bijective, injective, onto
Inverse Trigonometry: inverse trigonometric, principal value
Matrices: matrix, matrices, transpose, symmetric
Determinants: determinant, adjoint, cofactor, minor, consistent
Continuity and Differentiability: continuous, differentiable, dy/dx, derivative, rolle, mean value
Applications of Derivatives: maxima, minima, increasing, decreasing, tangent, normal, rate of change
Integrals: integrate, integral, evaluate
Applications of Integrals: area bounded, area of the region, area under
Differential Equations: differential equation, order and degree, homogeneous, integrating factor
Vectors: vector, dot product, cross product, unit vector, direction cosines
Three Dimensional Geometry: skew, shortest distance, direction ratios, plane
Linear Programming: linear programming, objective function, feasible region, corner point
Probability: probability, bayes, binomial, random variable""",
    "Custom (start empty)": "",
}

NOISE = re.compile(r"^(contd\.*|\d{3,5}\s*'?\w?'?\s*\(\d+\)|group\s*'?\w'?.*|page\s*\d+.*)$", re.I)
QSTART = re.compile(r"^(?:q(?:uestion)?\.?\s*)?(\d{1,2})\s*[\.\)]\s+\S", re.I)
QPREFIX = re.compile(r"^(?:q(?:uestion)?\.?\s*)?\d{1,2}\s*[\.\)]\s*", re.I)
MARKS = re.compile(r"[\[\(]\s*(\d{1,2})\s*(?:marks?)?\s*[\]\)]", re.I)


def ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        return shutil.which("tesseract") is not None
    except ImportError:
        return False


def ocr_image(img) -> str:
    import pytesseract
    return pytesseract.image_to_string(img)


@st.cache_data(show_spinner="Reading file...")
def read_file(name: str, data: bytes):
    ext = name.lower().rsplit(".", 1)[-1]
    note, text = "", ""
    if ext == "pdf":
        import pdfplumber
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if len(t.strip()) < 20:
                    if ocr_available():
                        t = ocr_image(page.to_image(resolution=200).original)
                        note = "Scanned pages were read with OCR."
                    else:
                        note = "Some pages are scanned and OCR (Tesseract) is not installed."
                text += t + "\n"
    elif ext in ("png", "jpg", "jpeg", "webp"):
        if ocr_available():
            from PIL import Image
            text = ocr_image(Image.open(io.BytesIO(data)))
        else:
            note = "OCR (Tesseract) is not installed, so photos cannot be read."
    else:
        text = data.decode("utf-8", errors="ignore")
    return [l.strip() for l in text.splitlines() if l.strip()], note


def detect_year(name: str, lines) -> str:
    m = re.search(r"(?:19|20)\d\d", name) or re.search(r"\b(?:19|20)\d\d\b", " ".join(lines[:15]))
    return m.group(0) if m else "Unknown"


def split_questions(lines):
    qs, cur = [], ""
    for l in lines:
        if NOISE.match(l):
            continue
        if QSTART.match(l):
            if cur:
                qs.append(cur)
            cur = QPREFIX.sub("", l)
        elif cur:
            cur += " " + l
    if cur:
        qs.append(cur)
    out = []
    for q in qs:
        marks = sum(int(x) for x in MARKS.findall(q)) or 1
        text = re.sub(r"\s+", " ", MARKS.sub("", q)).strip()
        if len(text) > 15:
            out.append((text, marks))
    return out


def parse_syllabus(text: str):
    syl = {}
    for line in text.splitlines():
        if ":" in line:
            name, kws = line.split(":", 1)
            if name.strip():
                syl[name.strip()] = [k.strip().lower() for k in kws.split(",") if k.strip()]
    return syl


def classify(q: str, syl) -> str:
    low, best, top = q.lower(), "Unclassified", 0
    for ch, kws in syl.items():
        s = sum(k in low for k in kws)
        if s > top:
            best, top = ch, s
    return best


def build_matrix(df, chapters):
    years = sorted(df["year"].unique())
    mat = df.pivot_table(index="chapter", columns="year", values="marks", aggfunc="sum", fill_value=0)
    idx = list(chapters) + (["Unclassified"] if "Unclassified" in mat.index else [])
    mat = mat.reindex(index=idx, columns=years, fill_value=0)
    mat.index.name = "chapter"
    return mat


def weighted_share(mat, years):
    sub = mat[years]
    share = sub / sub.sum().replace(0, 1)
    w = np.arange(1, len(years) + 1)
    return (share * w).sum(axis=1) / w.sum()


def priority_table(mat):
    years = list(mat.columns)
    pri = weighted_share(mat, years)
    asked = (mat > 0).sum(axis=1)
    last = mat.apply(lambda r: max([y for y in years if r[y] > 0], default="-"), axis=1)
    gap = mat.apply(lambda r: next((i for i, y in enumerate(reversed(years)) if r[y] > 0), len(years)), axis=1)
    out = pd.DataFrame({"Priority score (%)": (pri * 100).round(1), "Years asked": asked,
                        "Total marks": mat.sum(axis=1), "Last asked": last,
                        "Papers since last asked": gap})
    return out.sort_values("Priority score (%)", ascending=False)


def backtest(mat, k):
    real = [c for c in mat.index if c != "Unclassified"]
    sub, years, rows = mat.loc[real], list(mat.columns), []
    k = min(k, len(real))
    for i in range(2, len(years)):
        pred = weighted_share(sub, years[:i]).nlargest(k).index.tolist()
        act = sub[years[i]].nlargest(k).index.tolist()
        rows.append({"Paper predicted": years[i], "Papers used": i, "Predicted": ", ".join(pred),
                     "Actual top": ", ".join(act), "Hits": len(set(pred) & set(act)),
                     "Random baseline": round(k * k / len(real), 2)})
    return pd.DataFrame(rows), k


def find_repeats(df, thr):
    if len(df) < 2:
        return pd.DataFrame()
    try:
        sim = cosine_similarity(TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(df["question"]))
    except ValueError:
        return pd.DataFrame()
    rows = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if df.year.iloc[i] != df.year.iloc[j] and sim[i, j] >= thr:
                rows.append({"Similarity": round(float(sim[i, j]), 2), "Chapter": df.chapter.iloc[i],
                             "Year A": df.year.iloc[i], "Question A": df.question.iloc[i],
                             "Year B": df.year.iloc[j], "Question B": df.question.iloc[j]})
    return pd.DataFrame(rows).sort_values("Similarity", ascending=False) if rows else pd.DataFrame()


st.set_page_config(page_title=APP_NAME, layout="wide", page_icon=None)
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1200px}
.hdr{background:linear-gradient(90deg,#0B2545,#0B5FA5);color:#fff;padding:22px 28px;border-radius:8px;margin-bottom:18px}
.hdr h1{color:#fff;margin:0;font-size:1.9rem;letter-spacing:.3px}.hdr p{margin:4px 0 0;color:#cfe0f3}
div[data-testid="stMetric"]{background:#F3F6FA;border:1px solid #dde5ef;border-radius:8px;padding:12px 16px}
button[role="tab"]{font-weight:600}
</style>""", unsafe_allow_html=True)
st.markdown(f'<div class="hdr"><h1>{APP_NAME}</h1><p>{TAGLINE}</p></div>', unsafe_allow_html=True)

with st.sidebar:
    st.subheader("Settings")
    preset = st.selectbox("Syllabus preset", list(PRESETS))
    syl_text = st.text_area("Chapters and keywords (Chapter: keyword, keyword)", PRESETS[preset],
                            height=300, key=f"syl_{preset}")
    top_k = st.slider("Chapters to predict in backtest", 1, 5, 3)
    thr = st.slider("Repeat similarity threshold", 0.3, 0.9, 0.5, 0.05)
    st.caption("OCR for photos and scans: " + ("available" if ocr_available() else "not installed"))

syl = parse_syllabus(syl_text)
tabs = st.tabs(["1. Upload", "2. Review", "3. Insights", "4. Backtest", "5. Search and Export"])

rows = []
with tabs[0]:
    st.markdown("Upload one file per paper. The year is detected from the file name or text; correct it if needed.")
    files = st.file_uploader("Past papers (PDF, JPG, PNG, TXT)", type=["pdf", "png", "jpg", "jpeg", "txt"],
                             accept_multiple_files=True)
    for f in files or []:
        lines, note = read_file(f.name, f.getvalue())
        c1, c2 = st.columns([1, 3])
        yr = c1.text_input("Year", detect_year(f.name, lines), key=f"yr_{f.name}")
        qs = split_questions(lines)
        c2.write(f"**{f.name}**: {len(qs)} questions found. {note}")
        if not qs:
            c2.warning("No numbered questions found. Try the paste box below.")
        rows += [(yr, q, m) for q, m in qs]
    with st.expander("Or paste questions manually"):
        py = st.text_input("Year of pasted paper", key="paste_year")
        pt = st.text_area("One numbered question per line, e.g. '1. Define ... [2]'", height=160)
        if py and pt.strip():
            rows += [(py, q, m) for q, m in split_questions(pt.splitlines())]

df = pd.DataFrame(rows, columns=["year", "question", "marks"])
if not df.empty:
    df["chapter"] = df["question"].apply(lambda q: classify(q, syl))

with tabs[1]:
    if df.empty:
        st.info("Upload or paste at least one paper first.")
    else:
        st.markdown("Fix any wrong chapter or marks. You can also delete rows. Edits feed all other tabs.")
        opts = list(syl) + ["Unclassified"]
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True, height=420,
                            column_config={"chapter": st.column_config.SelectboxColumn("chapter", options=opts),
                                           "marks": st.column_config.NumberColumn("marks", min_value=0, step=1)})
        df = df.dropna(subset=["year", "question", "chapter"])

ready = not df.empty and len(syl) > 0
mat = build_matrix(df, list(syl)) if ready else None

with tabs[2]:
    if not ready:
        st.info("Add papers and a syllabus to see insights.")
    else:
        pt_ = priority_table(mat)
        a, b, c = st.columns(3)
        a.metric("Papers", len(mat.columns))
        b.metric("Questions", len(df))
        c.metric("Unclassified", int((df.chapter == "Unclassified").sum()))
        st.subheader("Revision priority")
        st.caption("Share of marks per chapter, weighted so recent papers count more.")
        chart = alt.Chart(pt_.reset_index()).mark_bar(color="#0B5FA5").encode(
            x=alt.X("Priority score (%):Q"), y=alt.Y("chapter:N", sort="-x", title=None),
            tooltip=list(pt_.reset_index().columns))
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(pt_, use_container_width=True)
        st.subheader("Chapter by year heatmap (marks)")
        long = mat.reset_index().melt(id_vars="chapter", var_name="year", value_name="marks")
        heat = alt.Chart(long).mark_rect().encode(
            x=alt.X("year:O", title="Paper year"), y=alt.Y("chapter:N", sort=list(mat.index), title=None),
            color=alt.Color("marks:Q", scale=alt.Scale(scheme="blues")), tooltip=["chapter", "year", "marks"])
        st.altair_chart(heat, use_container_width=True)
        st.subheader("Repeated or paraphrased questions")
        rep = find_repeats(df, thr)
        if rep.empty:
            st.write("No close repeats at this threshold.")
        else:
            st.dataframe(rep, use_container_width=True)

with tabs[3]:
    if not ready or len(mat.columns) < 3:
        st.info("Add at least 3 papers (different years) to run the backtest.")
    else:
        bt, k = backtest(mat, top_k)
        st.markdown(f"For each paper from the third onward, the model uses only earlier papers to predict its top {k} chapters by marks.")
        a, b = st.columns(2)
        a.metric("Average hits", f"{bt['Hits'].mean():.2f} of {k}")
        b.metric("Random baseline", f"{bt['Random baseline'].mean():.2f} of {k}")
        st.dataframe(bt, use_container_width=True)
        st.caption("Patterns are not guarantees. Small numbers of papers give noisy results.")

with tabs[4]:
    if not ready:
        st.info("Add papers first.")
    else:
        q = st.text_input("Search a topic (e.g. simplex, leontief)")
        if q:
            hit = df[df.question.str.contains(q, case=False, regex=False)]
            st.write(f"{len(hit)} matching questions")
            st.dataframe(hit, use_container_width=True)
        st.download_button("Download questions (CSV)", df.to_csv(index=False), "questions.csv", "text/csv")
        st.download_button("Download priority table (CSV)", priority_table(mat).to_csv(), "priority.csv", "text/csv")