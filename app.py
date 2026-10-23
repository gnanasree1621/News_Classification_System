import re
from pathlib import Path

import db
import joblib
import nltk
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer



st.set_page_config(
    page_title="Automated News Classification System | News Classifier",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)



BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
TFIDF_PATH = BASE_DIR / "tfidf.pkl"

CATEGORIES = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech",
}

ICONS = {
    "World": "🌍",
    "Sports": "🏏",
    "Business": "📈",
    "Sci/Tech": "💻",
}

DESCRIPTIONS = {
    "World": "Global events and international affairs",
    "Sports": "Matches, players and sporting events",
    "Business": "Markets, companies and finance",
    "Sci/Tech": "Science, technology and innovation",
}

CATEGORY_GRADIENTS = {
    "World": "linear-gradient(135deg,#dbeafe,#eef2ff)",
    "Sports": "linear-gradient(135deg,#dcfce7,#ecfeff)",
    "Business": "linear-gradient(135deg,#fef3c7,#fff7ed)",
    "Sci/Tech": "linear-gradient(135deg,#ede9fe,#f5f3ff)",
}



if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

if "nav_page" not in st.session_state:
    st.session_state.nav_page = "🏠 Dashboard"


def navigate_to_page(page_name):
    """Callback function to safely set navigation state before widget instantiation."""
    st.session_state.nav_page = page_name



st.markdown(
    """
<style>

/* ---------- APP ---------- */
.stApp {
    background:
        radial-gradient(circle at 8% 5%, rgba(99,102,241,.12), transparent 24%),
        radial-gradient(circle at 95% 12%, rgba(59,130,246,.10), transparent 25%),
        #f7f9ff;
}

.block-container {
    max-width: 1400px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1228 0%, #111c44 52%, #25135f 100%);
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

.sidebar-brand {
    padding: 12px 4px 24px 4px;
    border-bottom: 1px solid rgba(255,255,255,.12);
    margin-bottom: 22px;
}

.sidebar-brand-title {
    font-size: 26px;
    font-weight: 900;
    letter-spacing: -.5px;
}

.sidebar-brand-subtitle {
    color: #aab7d8 !important;
    font-size: 13px;
    margin-top: 5px;
}

.sidebar-heading {
    color: #94a3b8 !important;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    margin: 22px 0 8px;
}

[data-testid="stSidebar"] div[role="radiogroup"] label {
    font-size: 18px !important;
    font-weight: 600 !important;
    padding: 6px 0 !important;
}

[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-size: 18px !important;
    font-weight: 600 !important;
}

.hero-box {
    position: relative;
    overflow: hidden;
    padding: 42px 46px;
    border-radius: 28px;
    background:
        radial-gradient(circle at 88% 15%, rgba(168,85,247,.55), transparent 27%),
        radial-gradient(circle at 8% 90%, rgba(37,99,235,.48), transparent 32%),
        linear-gradient(135deg, #0b1737 0%, #1d4ed8 57%, #7c3aed 100%);
    box-shadow: 0 24px 60px rgba(37,99,235,.22);
    margin-bottom: 30px;
    animation: heroIn .7s ease-out;
}

.hero-box::after {
    content: "";
    position: absolute;
    width: 180px;
    height: 180px;
    right: -50px;
    bottom: -80px;
    border-radius: 50%;
    background: rgba(255,255,255,.08);
}

.hero-badge {
    display: inline-block;
    padding: 7px 14px;
    border-radius: 999px;
    background: rgba(255,255,255,.13);
    border: 1px solid rgba(255,255,255,.22);
    color: #e0e7ff;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: .7px;
}

.hero-title {
    color: #ffffff;
    font-size: clamp(34px, 4vw, 52px);
    line-height: 1.05;
    font-weight: 900;
    margin: 18px 0 10px;
}

.hero-text {
    color: #dbeafe;
    font-size: 16px;
    max-width: 700px;
    line-height: 1.7;
    margin: 0;
}

.hero-mini {
    color: #bfdbfe;
    font-size: 12px;
    margin-top: 18px;
}

@keyframes heroIn {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
}


.section-title {
    font-size: 26px;
    font-weight: 900;
    color: #0f172a;
    margin: 22px 0 16px;
}

.section-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-top: -8px;
    margin-bottom: 18px;
}


.kpi-card {
    background: rgba(255,255,255,.96);
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 20px;
    min-height: 135px;
    box-shadow: 0 8px 25px rgba(15,23,42,.055);
    transition: transform .25s ease, box-shadow .25s ease;
    animation: cardIn .55s ease both;
}

.kpi-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 18px 38px rgba(37,99,235,.13);
}

.kpi-icon { font-size: 28px; }
.kpi-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-top: 10px;
}
.kpi-value {
    color: #0f172a;
    font-size: 26px;
    font-weight: 900;
    margin-top: 3px;
}


.category-card {
    border-radius: 20px;
    padding: 23px;
    min-height: 150px;
    border: 1px solid rgba(148,163,184,.22);
    box-shadow: 0 8px 25px rgba(15,23,42,.05);
    transition: all .25s ease;
    animation: cardIn .6s ease both;
}

.category-card:hover {
    transform: translateY(-7px) scale(1.01);
    box-shadow: 0 18px 42px rgba(37,99,235,.14);
}

.category-icon { font-size: 34px; }
.category-name {
    color: #0f172a;
    font-size: 20px;
    font-weight: 900;
    margin-top: 8px;
}
.category-description {
    color: #64748b;
    font-size: 13px;
    line-height: 1.5;
    margin-top: 5px;
}


.analyzer-box {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 24px;
    padding: 24px;
    box-shadow: 0 10px 30px rgba(15,23,42,.06);
}

.result-box {
    border-radius: 24px;
    padding: 28px;
    background: linear-gradient(135deg,#eff6ff,#f5f3ff);
    border: 1px solid #dbeafe;
    animation: resultIn .5s ease-out;
}

.result-category {
    font-size: 34px;
    font-weight: 900;
    color: #1d4ed8;
    margin-top: 5px;
}

.confidence-number {
    font-size: 34px;
    font-weight: 900;
    color: #111827;
}

@keyframes resultIn {
    from { opacity: 0; transform: scale(.98) translateY(8px); }
    to { opacity: 1; transform: scale(1) translateY(0); }
}

@keyframes cardIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}


.stButton > button,
.stDownloadButton > button {
    border-radius: 12px !important;
    min-height: 46px !important;
    font-weight: 800 !important;
}


div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
    border-radius: 12px !important;
}

.feature-card {
    background: #e2f4ef;
    border-radius: 10px;
    padding: 18px 20px;
    margin-bottom: 16px;
    color: #111827 !important;
    font-size: 17px;
    font-weight: 500;
    border: 1px solid #d1ebe4;
    transition: all 0.2s ease;
}

.feature-check {
    color: #111827 !important;
    font-weight: 700;
    margin-right: 6px;
}

.feature-text {
    color: #111827 !important;
}

.feature-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
}


.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding: 25px 0 5px;
}

</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading News Classifier model...")
def load_model():
    if not MODEL_PATH.exists() or not TFIDF_PATH.exists():
        return None, None

    loaded_model = joblib.load(MODEL_PATH)
    loaded_tfidf = joblib.load(TFIDF_PATH)
    return loaded_model, loaded_tfidf


@st.cache_resource(show_spinner="Preparing NLP resources...")
def load_nlp():
    resources = ["stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    return set(stopwords.words("english")), WordNetLemmatizer()


model, tfidf = load_model()

if model is None or tfidf is None:
    st.error("Model files not found. Please keep model.pkl and tfidf.pkl in the same folder as app.py.")
    st.stop()

stop_words, lemmatizer = load_nlp()




def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


def predict_article(title, description):
    combined = f"{title.strip()} {description.strip()}"
    cleaned = preprocess_text(combined)

    if not cleaned.strip():
        raise ValueError("Please enter meaningful news text.")

    vector = tfidf.transform([cleaned])

    # predict() determines the category and is always run first.
    prediction = int(model.predict(vector)[0])
    category = CATEGORIES.get(prediction, "Unknown")

    # predict_proba() is attempted separately.  If the loaded model's pickle
    # was created with a different scikit-learn version than the one currently
    # installed, predict_proba can raise an AttributeError on internal
    # attributes (e.g. 'multi_class').  In that case the category returned by
    # predict() is still valid and is preserved; only the confidence
    # percentage and probability bar-chart are unavailable.
    probability_map = {}
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(vector)[0]
            probability_map = {
                CATEGORIES.get(int(class_id), str(class_id)): float(prob)
                for class_id, prob in zip(model.classes_, probabilities)
            }
            confidence = probability_map.get(category, 0.0)
        except AttributeError as exc:
            # Internal sklearn attribute missing — version mismatch between
            # training and deployment environment.  Category is still valid.
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "predict_proba AttributeError (%s): confidence unavailable. "
                "Check that scikit-learn in requirements.txt matches the "
                "version used to train model.pkl.",
                type(exc).__name__,
            )
        except Exception as exc:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                "predict_proba failed (%s): confidence unavailable.",
                type(exc).__name__,
            )

    return category, confidence, probability_map



with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-title">📰 Automated News Classifier</div>
            <div class="sidebar-brand-subtitle">News Classification</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">Menu</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "🔍 Analyze News", "📊 Analytics", "📜 History", "ℹ️ About"],
        key="nav_page",
        label_visibility="collapsed",
    )

    st.markdown('<div class="sidebar-heading">News Categories</div>', unsafe_allow_html=True)
    for name in ["World", "Sports", "Business", "Sci/Tech"]:
        st.write(f"{ICONS[name]}  {name}")

    st.markdown("---")
    db_connected, db_status_msg = db.check_connection(timeout_ms=3000)
    if db_connected:
        st.markdown("🟢 **MongoDB Connected**")
    else:
        st.markdown("🔴 **MongoDB Disconnected**")
        with st.expander("Why?", expanded=False):
            st.caption(db_status_msg)
    st.caption("Automated News Classifier • classification platform")



if page == "🏠 Dashboard":
    kpi_metrics = db.get_kpi_metrics()
    total_predictions = kpi_metrics["total_predictions"]
    latest_category = kpi_metrics["latest_category"]
    avg_confidence = kpi_metrics["avg_confidence"]
    db_connected_for_kpi = kpi_metrics.get("db_connected", False)

    

    st.markdown(
        """
        <div class="hero-box">
            <div class="hero-badge">✨ NEWS PLATFORM</div>
            <div class="hero-title">📰 Automated News Classifier</div>
            <p class="hero-text">
                Discover the category of any news article instantly with
                clear prediction results and confidence analysis.
            </p>
            <div class="hero-mini">Fast • Simple • Interactive</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">📊 News Intelligence Overview</div>', unsafe_allow_html=True)

    kpi_data = [
        ("⚡", "Articles Analyzed", str(total_predictions)),
        ("🎯", "Latest Category", latest_category),
        ("📈", "Average Confidence", f"{avg_confidence:.1f}%"),
        ("🏷️", "Available Categories", "4"),
    ]

    cols = st.columns(4)
    for col, (icon, label, value) in zip(cols, kpi_data):
        with col:
            st.markdown(
                f"""
                <div class="kpi-card">
                    <div class="kpi-icon">{icon}</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">🌈 Explore Categories</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Automated News Classifier currently supports four major news categories.</div>', unsafe_allow_html=True)

    category_cols = st.columns(4)
    for col, category in zip(category_cols, ["World", "Sports", "Business", "Sci/Tech"]):
        with col:
            st.markdown(
                f"""
                <div class="category-card" style="background:{CATEGORY_GRADIENTS[category]};">
                    <div class="category-icon">{ICONS[category]}</div>
                    <div class="category-name">{category}</div>
                    <div class="category-description">{DESCRIPTIONS[category]}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="section-title">🚀 Start Classification</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown(
            """
            <div class="analyzer-box">
                <h3 style="margin-top:0;color:#0f172a;">Ready to classify a news article?</h3>
                <p style="color:#64748b;line-height:1.6;">
                    Enter a headline and description to receive an instant category prediction,
                    confidence value, and probability breakdown.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.button(
            "🔍 Open News Analyzer",
            type="primary",
            use_container_width=True,
            on_click=navigate_to_page,
            args=("🔍 Analyze News",),
        )

    if not db_connected_for_kpi:
        st.info(
            "📡 **MongoDB not connected** — KPI counts show 0. "
            "Connect to MongoDB Atlas to persist and view analytics data."
        )

    st.markdown(
        '<div class="footer">📰 Automated News Classifier • Analyze • Predict • Understand</div>',
        unsafe_allow_html=True,
    )


elif page == "🔍 Analyze News":
    st.markdown('<div class="section-title">🔍 News Analyzer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Enter the headline and description of a news article.</div>',
        unsafe_allow_html=True,
    )

    with st.form("news_analyzer_form", clear_on_submit=False):
        title = st.text_input(
            "News Headline",
            placeholder="Example: India wins an exciting cricket match",
        )

        description = st.text_area(
            "News Description",
            placeholder="Paste or type the news description here...",
            height=190,
        )

        submitted = st.form_submit_button(
            "🚀 Predict News Category",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not title.strip():
            st.warning("Please enter a news headline.")
        elif not description.strip():
            st.warning("Please enter a news description.")
        else:
            try:
                with st.spinner("Analyzing article..."):
                    category, confidence, probability_map = predict_article(title, description)

                conf_percentage = round(confidence * 100, 2)

                # Save prediction to MongoDB
                saved_to_db, db_result = db.save_prediction(
                    headline=title,
                    description=description,
                    category=category,
                    confidence=conf_percentage,
                    probabilities=probability_map,
                )

                record = {
                    "Headline": title.strip(),
                    "Category": category,
                    "Confidence": conf_percentage,
                }
                st.session_state.history.append(record)
                st.session_state.last_result = record

                if saved_to_db:
                    st.success(f"✅ Prediction completed and saved to MongoDB: **{category}**")
                else:
                    st.warning(
                        f"⚠️ Prediction completed: **{category}** — "
                        f"but the result could **not** be saved to MongoDB. "
                        f"Reason: {db_result}"
                    )

                left, right = st.columns([1.15, 0.85])

                with left:
                    st.markdown(
                        f"""
                        <div class="result-box">
                            <div style="color:#64748b;font-size:12px;font-weight:800;letter-spacing:1px;">
                                PREDICTED CATEGORY
                            </div>
                            <div class="result-category">
                                {ICONS.get(category, '📰')} {category}
                            </div>
                            <div style="color:#64748b;margin-top:8px;">
                                {DESCRIPTIONS.get(category, '')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                with right:
                    st.markdown("### 🎯 Confidence")
                    st.markdown(
                        f'<div class="confidence-number">{confidence * 100:.2f}%</div>',
                        unsafe_allow_html=True,
                    )
                    st.progress(float(max(0.0, min(1.0, confidence))))
                    st.caption("Model confidence for the predicted category.")

                if probability_map:
                    st.markdown("### 📊 Category Probability Breakdown")

                    names = list(CATEGORIES.values())
                    values = [probability_map.get(name, 0.0) * 100 for name in names]

                    fig = go.Figure(
                        go.Bar(
                            x=names,
                            y=values,
                            text=[f"{value:.1f}%" for value in values],
                            textposition="auto",
                            hovertemplate="%{x}<br>Probability: %{y:.2f}%<extra></extra>",
                        )
                    )
                    fig.update_layout(
                        height=390,
                        yaxis_title="Probability (%)",
                        yaxis_range=[0, 100],
                        margin=dict(l=20, r=20, t=30, b=20),
                        template="plotly_white",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

    st.markdown("### 💡 Quick Test")
    sample_title = "India wins an exciting cricket match"
    sample_description = (
        "India defeated its opponent in an international cricket match. "
        "The players delivered an excellent batting performance and secured victory."
    )
    st.caption("Sample input for testing your trained model:")
    st.code(f"Headline: {sample_title}\nDescription: {sample_description}", language="text")



elif page == "📊 Analytics":
    st.markdown('<div class="section-title">📊 Prediction Analytics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Review predictions retrieved from MongoDB.</div>',
        unsafe_allow_html=True,
    )

    success, db_docs = db.get_predictions()

    if not success:
        st.warning(
            "📡 Analytics unavailable — could not connect to MongoDB Atlas. "
            f"Details: {db_docs}"
        )
    elif not isinstance(db_docs, list):
        st.error("Unexpected response from database. Please try refreshing the page.")
    elif not db_docs:
        st.info("📊 No predictions yet. Use **Analyze News** to classify your first article.")
    else:
        analytics_data = [
            {
                "Category": d.get("category", ""),
                "Confidence": float(d.get("confidence", 0.0)),
            }
            for d in db_docs
            if isinstance(d, dict)
        ]
        df = pd.DataFrame(analytics_data)
        counts = df["Category"].value_counts().reindex(list(CATEGORIES.values()), fill_value=0)

        a, b, c = st.columns(3)
        a.metric("Total Predictions", len(df))
        b.metric("Average Confidence", f"{df['Confidence'].mean():.1f}%")
        c.metric("Categories Used", int((counts > 0).sum()))

        left, right = st.columns(2)

        with left:
            fig = go.Figure(
                go.Pie(
                    labels=counts.index,
                    values=counts.values,
                    hole=0.52,
                    textinfo="label+percent",
                )
            )
            fig.update_layout(
                title="Category Distribution",
                height=430,
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig2 = go.Figure(
                go.Bar(
                    x=counts.index,
                    y=counts.values,
                    text=counts.values,
                    textposition="auto",
                )
            )
            fig2.update_layout(
                title="Prediction Count by Category",
                height=430,
                yaxis_title="Predictions",
                margin=dict(l=20, r=20, t=60, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🧾 Recent Predictions")
        recent_rows = [
            {
                "Headline": d.get("headline", ""),
                "Category": d.get("category", ""),
                "Confidence": f"{d.get('confidence', 0.0):.2f}%",
                "Date/Time": d.get("timestamp_str", ""),
            }
            for d in db_docs
        ]
        st.dataframe(pd.DataFrame(recent_rows), use_container_width=True, hide_index=True)


elif page == "📜 History":
    st.markdown('<div class="section-title">📜 Prediction History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-subtitle">Your predictions retrieved from MongoDB (newest first).</div>',
        unsafe_allow_html=True,
    )

    success, db_docs = db.get_predictions()
    if not success:
        st.warning(
            "📡 History unavailable — could not connect to MongoDB Atlas. "
            f"Details: {db_docs}"
        )
    elif not isinstance(db_docs, list):
        st.error("Unexpected response from database. Please try refreshing the page.")
    elif not db_docs:
        st.info("📜 No predictions yet. Use **Analyze News** to create your first prediction.")
    else:
        history_rows = [
            {
                "Headline": d.get("headline", ""),
                "Category": d.get("category", ""),
                "Confidence": f"{d.get('confidence', 0.0):.2f}%",
                "Date/Time": d.get("timestamp_str", ""),
            }
            for d in db_docs
            if isinstance(d, dict)
        ]
        df = pd.DataFrame(history_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_rows = [
            {
                "Headline": d.get("headline", ""),
                "Description": d.get("description", ""),
                "Category": d.get("category", ""),
                "Confidence": d.get("confidence", 0.0),
                "Created_At": d.get("timestamp_str", ""),
            }
            for d in db_docs
        ]
        csv_data = pd.DataFrame(csv_rows).to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Download Prediction History",
            data=csv_data,
            file_name="newsai_history.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")

        if "confirm_clear" not in st.session_state:
            st.session_state.confirm_clear = False

        if not st.session_state.confirm_clear:
            if st.button("🗑️ Clear Prediction History", use_container_width=True):
                st.session_state.confirm_clear = True
                st.rerun()
        else:
            st.warning("⚠️ Are you sure you want to permanently delete all prediction records from MongoDB?")
            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button("✔️ Yes, Clear History", type="primary", use_container_width=True):
                    clear_ok, del_count = db.clear_predictions()
                    st.session_state.history = []
                    st.session_state.last_result = None
                    st.session_state.confirm_clear = False
                    if clear_ok:
                        st.success(f"Cleared {del_count} prediction record(s) from MongoDB.")
                    else:
                        st.error(f"Failed to clear MongoDB: {del_count}")
                    st.rerun()
            with col_no:
                if st.button("❌ Cancel", use_container_width=True):
                    st.session_state.confirm_clear = False
                    st.rerun()



elif page == "ℹ️ About":
    st.markdown('<div class="section-title">ℹ️ About NewsAI</div>', unsafe_allow_html=True)

    st.markdown(
        """
        <div class="analyzer-box">
            <h2 style="color:#0f172a;margin-top:0;">📰 Automated News Classifier</h2>
            <p style="color:#64748b;line-height:1.8;">
                Automated News Classifier is an interactive news classification platform that predicts
                the category of a news article from its headline and description.
                The interface is designed for fast testing, clear results and persistent storage in MongoDB.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ✨ Platform Features")
    features = [
        "Instant news category prediction",
        "Confidence and probability analysis",
        "MongoDB persistent prediction storage",
        "MongoDB status monitoring",
        "Interactive prediction visualizations",
        "Prediction history from MongoDB",
        "Analytics dashboard powered by MongoDB",
        "CSV export capability",
        "Responsive professional interface",
    ]

    cols = st.columns(2)
    for index, feature in enumerate(features):
        with cols[index % 2]:
            st.markdown(
                f"""
                    <div class="feature-card">
                    <span class="feature-check">✓</span>
                    <span class="feature-text">{feature}</span>
                    </div>
                """,
            unsafe_allow_html=True,
        )


st.markdown(
    '<div class="footer">📰 News Classifier • News Classification Platform</div>',
    unsafe_allow_html=True,
)

