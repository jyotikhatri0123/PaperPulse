"""PaperPulse v2 - exam paper pattern miner for Class 8 to 12.

Pick class, stream and subject, paste each year's paper in its own box (or upload
a PDF/photo per box), and get chapter priorities, trends, repeated questions,
a backtest and a revision plan.   Run:  streamlit run app.py
"""
import io
import json
import re
import shutil

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

APP_NAME = "PaperPulse"
TAGLINE = "Find the chapters and questions that exam papers keep coming back to."

# ------------------------------------------------------------ subject catalogue
# Edit these lists if your board or school offers different subjects.
C, O = "Compulsory", "Optional"
SEE = [("Compulsory English", C), ("Compulsory Nepali", C), ("Compulsory Mathematics", C),
       ("Science and Technology", C), ("Social Studies", C), ("Optional Mathematics", O),
       ("Computer Science", O), ("Accountancy", O), ("Economics", O),
       ("Health, Population and Environment", O), ("Moral Education", O), ("Sanskrit", O)]
SCHOOL = {
    "Class 8": [("English", C), ("Nepali", C), ("Mathematics", C), ("Science and Technology", C),
                ("Social Studies", C), ("Health, Population and Environment", C),
                ("Computer Science", O), ("Moral Education", O),
                ("Occupation, Business and Technology", O)],
    "Class 9": SEE, "Class 10": SEE,
}
NEB_COMMON = [("English", C), ("Nepali", C), ("Social Studies and Life Skills", C)]
NEB = {
    "Science": [("Physics", C), ("Chemistry", C), ("Biology", O), ("Mathematics", O), ("Computer Science", O)],
    "Management": [("Accountancy", C), ("Business Studies", C), ("Economics", O), ("Business Mathematics", O),
                   ("Marketing", O), ("Hotel Management", O), ("Computer Science", O)],
    "Humanities": [("Sociology", O), ("Population Studies", O), ("Geography", O), ("Political Science", O),
                   ("Economics", O), ("Mathematics", O), ("Rural Development", O), ("Computer Science", O)],
    "Education": [("Child Development and Learning", O), ("Mathematics", O),
                  ("Health and Physical Education", O), ("Computer Science", O)],
}
CLASSES = ["Class 8", "Class 9", "Class 10", "Class 11", "Class 12"]


def subjects_for(cls, stream):
    if cls in SCHOOL:
        return SCHOOL[cls]
    return NEB_COMMON + NEB[stream]


# Starter syllabi (Chapter: keywords). Other subjects use auto-detected topics or your own list.
SYL = {
    "Business Mathematics": """Matrices and Determinants: matrix, determinant, cofactor, adjoint, adj, cramer, symmetric, inverse, gauss, linear equations
Input-Output Analysis: leontief, technology matrix, hawkins, viable, final demand, input-output
Linear Programming: simplex, slack, surplus, objective function, feasible, pivot
Differentiation Applications: maximizes, maximum, minimum, revenue, demand function, marginal, profit, elasticity, average cost
Integration: integral, integrate, consumer surplus, producer surplus
Differential Equations: differential equation, growth rate, population, degree
Mathematics of Finance: depreciat, scrap value, compound, interest, annuity, price reduces
Investment Appraisal: net present value, npv, present value, future returns
Probability: probability, dependent events, independent events, bayes
Statistics: regression, correlation, median, quartile, skewness, standard deviation, mode""",
    "Mathematics": """Algebra: equation, factor, expression, polynomial, quadratic, simplify, inequality, sequence
Geometry: triangle, circle, angle, congruent, similar, parallel, quadrilateral, theorem
Trigonometry: sin, cos, tan, trigonometric, height and distance, angle of elevation
Mensuration: area, volume, perimeter, surface, cylinder, cone, sphere
Statistics: mean, median, mode, frequency, data, quartile
Probability: probability, chance, event, sample space
Sets and Functions: set, union, intersection, function, relation, venn
Coordinate Geometry: coordinate, slope, distance between, gradient, line""",
    "Physics": """Mechanics: velocity, acceleration, newton, force, momentum, work, energy, power, gravitation
Heat and Thermodynamics: heat, temperature, thermal, specific heat, expansion, gas
Waves and Sound: wave, sound, frequency, wavelength, resonance
Optics: lens, mirror, refraction, reflection, interference, diffraction, prism
Electricity: charge, current, resistance, ohm, capacitor, circuit, electric field
Magnetism: magnetic, solenoid, galvanometer, induction, transformer, alternating
Modern Physics: photoelectric, nucleus, radioactive, semiconductor, diode, bohr""",
    "Chemistry": """Atomic Structure: atom, electron, orbital, quantum, isotope
Periodic Table: periodic, group, period, electronegativity, ionization
Chemical Bonding: bond, ionic, covalent, hybridization, molecule
Acids, Bases and Salts: acid, base, salt, ph, buffer, neutralization
Organic Chemistry: alkane, alkene, alcohol, benzene, organic, ester, carboxylic
Equilibrium and Kinetics: equilibrium, rate of reaction, le chatelier, catalyst
Electrochemistry: electrolysis, cell, oxidation, reduction, redox, electrode
Metals and Non-metals: metal, extraction, alloy, corrosion""",
    "Biology": """Cell Biology: cell, nucleus, mitochondria, membrane, mitosis, meiosis
Plant Physiology: photosynthesis, transpiration, respiration, plant, root, xylem
Human Physiology: blood, heart, digestion, kidney, nervous, hormone, respiratory
Genetics and Evolution: gene, dna, inheritance, mutation, evolution, chromosome
Ecology: ecosystem, food chain, population, environment, biodiversity
Diversity of Life: classification, kingdom, bacteria, fungi, virus, algae""",
    "Accountancy": """Journal and Ledger: journal, ledger, debit, credit, entry, posting
Trial Balance and Final Accounts: trial balance, trading account, profit and loss, balance sheet, adjustment
Depreciation: depreciation, straight line, diminishing, asset
Bank Reconciliation: bank reconciliation, cheque, pass book, cash book
Partnership and Company: partnership, partner, share, debenture, company
Cash Flow and Ratios: cash flow, ratio, liquidity, working capital""",
    "Economics": """Demand and Supply: demand, supply, equilibrium, elasticity, price
Production and Cost: production, cost, revenue, marginal, returns
Market Structures: monopoly, perfect competition, oligopoly, market
National Income: gdp, national income, inflation, unemployment
Money and Banking: money, bank, central bank, monetary, interest
Public Finance and Trade: tax, budget, fiscal, trade, deficit, export""",
    "Computer Science": """Computer Fundamentals: hardware, software, input, output, memory, generation
Number Systems: binary, decimal, octal, hexadecimal, conversion
Programming: algorithm, flowchart, program, loop, variable, qbasic, python, array
Database: database, table, query, sql, record, field, primary key
Networking and Internet: network, internet, protocol, lan, wan, ip, email
Web and Security: html, website, virus, security, cyber, firewall""",
    "English": """Reading Comprehension: passage, comprehension, read the text, according to the text
Grammar: tense, preposition, article, voice, narration, conjunction, sentence
Writing: essay, letter, application, paragraph, notice, report, story
Literature: poem, story, character, theme, author, stanza""",
    "Social Studies": """History: history, ancient, king, empire, revolution, movement
Geography: map, climate, river, mountain, population, resources, province
Civics and Governance: constitution, government, rights, duties, election, democracy
Economy and Society: economy, culture, tradition, development, tourism, agriculture""",
}
ALIAS = {"Compulsory Mathematics": "Mathematics", "Optional Mathematics": "Mathematics",
         "Science and Technology": None, "Social Studies and Life Skills": "Social Studies"}


def default_syllabus(subject):
    key = ALIAS.get(subject, subject)
    return SYL.get(key, "") if key else ""


# ----------------------------------------------------------------- extraction
NOISE = re.compile(r"^(contd\.*|\d{3,5}\s*'?\w?'?\s*\(\d+\)|group\s*'?\w'?.*|page\s*\d+.*)$", re.I)
QSTART = re.compile(r"^(?:q(?:uestion)?\.?\s*)?\(?(\d{1,2})\s*[\.\)]\s+\S", re.I)
QPREFIX = re.compile(r"^(?:q(?:uestion)?\.?\s*)?\(?\d{1,2}\s*[\.\)]\s*", re.I)
MARKS = re.compile(r"[\[\(]\s*(\d{1,2})\s*(?:marks?)?\s*[\]\)]", re.I)


def ocr_available():
    try:
        import pytesseract  # noqa: F401
        return shutil.which("tesseract") is not None
    except ImportError:
        return False


def ocr_image(img):
    import pytesseract
    from PIL import ImageOps
    img = ImageOps.autocontrast(ImageOps.grayscale(img))
    if img.width < 2000:
        img = img.resize((2000, int(img.height * 2000 / img.width)))
    try:
        return pytesseract.image_to_string(img, lang="eng+nep", config="--psm 6")
    except Exception:
        return pytesseract.image_to_string(img, config="--psm 6")


@st.cache_data(show_spinner="Reading file...")
def read_file(name, data):
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
                        note = "Scanned pages found but OCR is not installed."
                text += t + "\n"
    elif ext in ("png", "jpg", "jpeg", "webp"):
        if ocr_available():
            from PIL import Image
            text = ocr_image(Image.open(io.BytesIO(data)))
            note = "Photo read with OCR. Check the Questions tab."
        else:
            note = "OCR is not installed, so photos cannot be read."
    else:
        text = data.decode("utf-8", errors="ignore")
    return [l.strip() for l in text.splitlines() if l.strip()], note


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
    return qs


def to_items(qs):
    out = []
    for q in qs:
        marks = sum(int(x) for x in MARKS.findall(q)) or 1
        text = re.sub(r"\s+", " ", MARKS.sub("", q)).strip()
        if len(text) > 12:
            out.append((text, marks))
    return out


def extract_questions(lines):
    """Numbered questions first; if none are found, treat each long line as a question."""
    qs = split_questions(lines)
    fallback = False
    if not qs:
        qs = [l for l in lines if len(l) > 15 and not NOISE.match(l)]
        fallback = bool(qs)
    return to_items(qs), fallback


def parse_syllabus(text):
    syl = {}
    for line in text.splitlines():
        if ":" in line:
            name, kws = line.split(":", 1)
            if name.strip():
                syl[name.strip()] = [k.strip().lower() for k in kws.split(",") if k.strip()]
    return syl


def classify(q, syl):
    low, best, top = q.lower(), "Unclassified", 0
    for ch, kws in syl.items():
        s = sum(k in low for k in kws)
        if s > top:
            best, top = ch, s
    return best


def auto_topics(df, k):
    """Cluster questions into k topics and name each by its top terms."""
    if len(df) < 3:
        return ["Unclassified"] * len(df)
    k = max(2, min(k, len(df)))
    vec = TfidfVectorizer(stop_words="english", max_features=600)
    try:
        X = vec.fit_transform(df["question"])
    except ValueError:
        return ["Unclassified"] * len(df)
    km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
    terms = np.array(vec.get_feature_names_out())
    names = {c: "Topic: " + ", ".join(terms[km.cluster_centers_[c].argsort()[::-1][:3]]) for c in range(k)}
    return [names[c] for c in km.labels_]


# ------------------------------------------------------------------- analysis
def qtype(m):
    return "Objective (1 mark)" if m <= 1 else "Short (2-3 marks)" if m <= 3 else "Long (4+ marks)"


def build_matrix(df, chapters):
    years = sorted(df["year"].unique())
    mat = df.pivot_table(index="chapter", columns="year", values="marks", aggfunc="sum", fill_value=0)
    idx = list(chapters) + [c for c in mat.index if c not in chapters]
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
    last = mat.apply(lambda r: max([y for y in years if r[y] > 0], default="-"), axis=1)
    gap = mat.apply(lambda r: next((i for i, y in enumerate(reversed(years)) if r[y] > 0), len(years)), axis=1)
    out = pd.DataFrame({"Priority score (%)": (pri * 100).round(1), "Papers it appeared in": (mat > 0).sum(axis=1),
                        "Total marks": mat.sum(axis=1), "Last asked": last, "Papers since last asked": gap})
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


def make_plan(pri, days, hpd, weak):
    pri = pri[pri.index != "Unclassified"]
    w = pri["Priority score (%)"].clip(lower=0.01) * np.where(pri.index.isin(weak), 1.5, 1.0)
    hours = (w / w.sum() * days * hpd).round(2)
    rows, day, room = [], 1, float(hpd)
    for ch, h in hours.items():
        while h > 1e-9 and day <= days:
            take = min(h, room)
            rows.append({"Day": day, "Chapter": ch, "Hours": round(take, 2)})
            h -= take
            room -= take
            if room <= 1e-9:
                day, room = day + 1, float(hpd)
    return hours.rename("Hours").reset_index(), pd.DataFrame(rows)


# ------------------------------------------------------------------------- UI
st.set_page_config(page_title=APP_NAME, layout="wide", page_icon="logo.png")
st.markdown("""<style>
.block-container{padding-top:1.2rem;max-width:1200px}
.hdr{background:linear-gradient(90deg,#0B2545,#0B5FA5);color:#fff;padding:22px 28px;border-radius:8px;margin-bottom:12px}
.hdr h1{color:#fff;margin:0;font-size:1.9rem;letter-spacing:.3px}.hdr p{margin:4px 0 0;color:#cfe0f3}
.ctx{background:#F3F6FA;border-left:4px solid #0B5FA5;padding:8px 14px;border-radius:4px;margin-bottom:12px;font-weight:600}
div[data-testid="stMetric"]{background:#F3F6FA;border:1px solid #dde5ef;border-radius:8px;padding:12px 16px}
button[role="tab"]{font-weight:600}
</style>""", unsafe_allow_html=True)
st.markdown(f'<div class="hdr"><h1>{APP_NAME}</h1><p>{TAGLINE}</p></div>', unsafe_allow_html=True)

# Apply a loaded session before any widget exists
if "_pending" in st.session_state:
    p = st.session_state.pop("_pending")
    st.session_state["cls"] = p["class"]
    if p.get("stream"):
        st.session_state["stream"] = p["stream"]
    st.session_state["custom_subject"] = p["subject"]
    st.session_state["n_papers"] = len(p["papers"])
    pns = f"{p['class']}|{p.get('stream')}|{p['subject']}"
    for i, pp_ in enumerate(p["papers"]):
        st.session_state[f"{pns}|year{i}"] = pp_["year"]
        st.session_state[f"{pns}|text{i}"] = pp_["text"]

with st.sidebar:
    st.subheader("1. Choose exam")
    cls = st.selectbox("Class", CLASSES, key="cls")
    stream = st.selectbox("Stream", list(NEB), key="stream") if cls in ("Class 11", "Class 12") else None
    subs = subjects_for(cls, stream)
    kind = st.radio("Show subjects", ["All", "Compulsory", "Optional"], horizontal=True)
    shown = [s for s in subs if kind == "All" or s[1] == kind] or subs
    label = st.selectbox("Subject", [f"{n} ({t})" for n, t in shown])
    subject = label.rsplit(" (", 1)[0]
    custom = st.text_input("Subject not listed? Type it here", key="custom_subject")
    if custom.strip():
        subject = custom.strip()
    ns = f"{cls}|{stream}|{subject}"

    st.subheader("2. Syllabus")
    dflt = default_syllabus(subject)
    auto = st.checkbox("Auto-detect topics from the questions", value=not dflt, key=f"auto|{ns}",
                       help="Groups similar questions into topics when you have no keyword list.")
    n_auto = st.slider("Topics to detect", 3, 12, 6)
    syl_text = st.text_area("Chapters and keywords (Chapter: keyword, keyword)", dflt, height=240, key=f"syl|{ns}")

    st.subheader("3. Papers")
    n_papers = st.slider("Number of papers (years)", 1, 15, 8, key="n_papers")
    latest = st.number_input("Latest paper year (fills the year boxes)", 1990, 2100, 2083)
    top_k = st.slider("Chapters to predict in backtest", 1, 5, 3)
    thr = st.slider("Repeat similarity threshold", 0.3, 0.9, 0.5, 0.05)

    st.subheader("4. Save and load")
    saved = [{"year": st.session_state.get(f"{ns}|year{i}", ""), "text": st.session_state.get(f"{ns}|text{i}", "")}
             for i in range(n_papers)]
    st.download_button("Save these papers (JSON)", json.dumps(
        {"class": cls, "stream": stream, "subject": subject, "papers": saved}), f"{subject}_papers.json")
    up = st.file_uploader("Load saved papers", type=["json"], key="loadjson")
    if up is not None and st.session_state.get("_sig") != (up.name, up.size):
        try:
            st.session_state["_pending"] = json.loads(up.getvalue())
            st.session_state["_sig"] = (up.name, up.size)
            st.rerun()
        except Exception:
            st.error("That file could not be loaded.")
    st.caption("OCR for photos and scans: " + ("available" if ocr_available() else "not installed"))

syl = parse_syllabus(syl_text)
st.markdown(f'<div class="ctx">{cls}{" | " + stream if stream else ""} | {subject}</div>', unsafe_allow_html=True)
tabs = st.tabs(["Papers", "Questions", "Insights", "Repeats", "Backtest", "Study plan"])

# ---- Papers: one box per year
rows = []
with tabs[0]:
    st.markdown("Paste **one year's paper in each box**. Number the questions (1., 2., ...) and write marks "
                "like [2] if you can. Or upload a PDF or photo inside a box.")
    for i in range(n_papers):
        ky, kt, kf = f"{ns}|year{i}", f"{ns}|text{i}", f"{ns}|file{i}"
        title = f"Paper {i + 1}: {st.session_state.get(ky, str(int(latest) - i))}"
        with st.expander(title, expanded=(i == 0)):
            yr = st.text_input("Year", str(int(latest) - i), key=ky)
            txt = st.text_area("Paste the questions", height=200, key=kt)
            f = st.file_uploader("Or upload PDF, photo or text", type=["pdf", "png", "jpg", "jpeg", "txt"], key=kf)
            lines = [l.strip() for l in txt.splitlines() if l.strip()]
            note = ""
            if f is not None:
                fl, note = read_file(f.name, f.getvalue())
                lines += fl
            items, fb = extract_questions(lines)
            if items:
                st.caption(f"{len(items)} questions found. {note}" + (" No numbering found, so each line was used." if fb else ""))
            elif lines:
                st.warning("No questions found. Start each question with a number like 1.")
            rows += [(yr.strip() or "Unknown", q, m) for q, m in items]

df = pd.DataFrame(rows, columns=["year", "question", "marks"])
if not df.empty:
    df["chapter"] = auto_topics(df, n_auto) if (auto or not syl) else df["question"].apply(lambda q: classify(q, syl))

# ---- Questions: review, search, export
with tabs[1]:
    if df.empty:
        st.info("Add at least one paper in the Papers tab.")
    else:
        use_auto = auto or not syl
        opts = sorted(df.chapter.unique()) if use_auto else list(syl) + ["Unclassified"]
        st.markdown("Fix wrong chapters or marks, or delete rows. Edits feed every other tab.")
        df = st.data_editor(df, num_rows="dynamic", use_container_width=True, height=380,
                            column_config={"chapter": st.column_config.SelectboxColumn("chapter", options=opts),
                                           "marks": st.column_config.NumberColumn("marks", min_value=0, step=1)})
        df = df.dropna(subset=["year", "question", "chapter"])
        q = st.text_input("Search a topic or word")
        if q:
            hit = df[df.question.str.contains(q, case=False, regex=False)]
            st.write(f"{len(hit)} matching questions")
            st.dataframe(hit, use_container_width=True)
        st.download_button("Download questions (CSV)", df.to_csv(index=False), f"{subject}_questions.csv", "text/csv")

ready = not df.empty
if ready:
    chapters = [c for c in syl if c in set(df.chapter)] if not (auto or not syl) else sorted(df.chapter.unique())
    mat = build_matrix(df, chapters)
    pri = priority_table(mat)

# ---- Insights
with tabs[2]:
    if not ready:
        st.info("Add papers to see insights.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Papers", len(mat.columns))
        b.metric("Questions", len(df))
        c.metric("Total marks", int(df.marks.sum()))
        d.metric("Unclassified", int((df.chapter == "Unclassified").sum()))
        top = [i for i in pri.index if i != "Unclassified"][:3]
        st.success("Top chapters to revise first: " + ", ".join(top))
        st.subheader("Revision priority")
        st.caption("Share of marks per chapter, weighted so recent papers count more.")
        pr = pri.reset_index()
        st.altair_chart(alt.Chart(pr).mark_bar(color="#0B5FA5").encode(
            x="Priority score (%):Q", y=alt.Y("chapter:N", sort="-x", title=None),
            tooltip=list(pr.columns)), use_container_width=True)
        st.dataframe(pri, use_container_width=True)
        st.subheader("Chapter by year heatmap (marks)")
        long = mat.reset_index().melt(id_vars="chapter", var_name="year", value_name="marks")
        st.altair_chart(alt.Chart(long).mark_rect().encode(
            x=alt.X("year:O", title="Paper year"), y=alt.Y("chapter:N", sort=list(mat.index), title=None),
            color=alt.Color("marks:Q", scale=alt.Scale(scheme="blues")),
            tooltip=["chapter", "year", "marks"]), use_container_width=True)
        st.subheader("Question types by year")
        tdf = df.assign(type=df.marks.map(qtype)).groupby(["year", "type"], as_index=False).marks.sum()
        st.altair_chart(alt.Chart(tdf).mark_bar().encode(
            x=alt.X("year:O", title="Paper year"), y=alt.Y("marks:Q", title="Marks"),
            color=alt.Color("type:N", title="Type"), tooltip=["year", "type", "marks"]), use_container_width=True)

# ---- Repeats
with tabs[3]:
    if not ready:
        st.info("Add papers first.")
    else:
        rep = find_repeats(df, thr)
        if rep.empty:
            st.write("No close repeats at this threshold. Try lowering it in the sidebar.")
        else:
            st.write(f"{len(rep)} similar question pairs across different years.")
            st.dataframe(rep, use_container_width=True)

# ---- Backtest
with tabs[4]:
    if not ready or len(mat.columns) < 3:
        st.info("Add papers from at least 3 different years to run the backtest.")
    else:
        bt, k = backtest(mat, top_k)
        st.markdown(f"For each paper from the third onward, only earlier papers are used to predict its top {k} chapters by marks.")
        a, b = st.columns(2)
        a.metric("Average hits", f"{bt['Hits'].mean():.2f} of {k}")
        b.metric("Random baseline", f"{bt['Random baseline'].mean():.2f} of {k}")
        st.dataframe(bt, use_container_width=True)
        st.caption("Patterns are not guarantees. Few papers give noisy results.")

# ---- Study plan
with tabs[5]:
    if not ready:
        st.info("Add papers first.")
    else:
        c1, c2 = st.columns(2)
        days = c1.number_input("Days until the exam", 1, 365, 14)
        hpd = c2.number_input("Study hours per day", 0.5, 16.0, 3.0, 0.5)
        weak = st.multiselect("Chapters you find difficult (get 50% more time)",
                              [i for i in pri.index if i != "Unclassified"])
        hours, plan = make_plan(pri, int(days), float(hpd), weak)
        st.subheader("Hours per chapter")
        st.dataframe(hours, use_container_width=True)
        st.subheader("Day-by-day schedule")
        st.dataframe(plan, use_container_width=True, height=360)
        st.download_button("Download plan (CSV)", plan.to_csv(index=False), f"{subject}_plan.csv", "text/csv")
