"""Streamlit UI:  streamlit run app.py   (or: python3 -m streamlit run app.py)"""
import io, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay
import streamlit as st
import pipeline as P

st.set_page_config(page_title="Beyond the Fireworks", layout="wide")
sns.set_theme(style="whitegrid")
HERE = os.path.dirname(os.path.abspath(__file__))
TITLE = "Beyond the Fireworks: Machine Learning-Based Next-Day AQI Risk Prediction for Indian Cities"


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


# ------------------------------------------------------------------ sidebar: choose the data
st.sidebar.title("Data")
uploaded = st.sidebar.file_uploader("Upload the AQI CSV (optional)", type=["csv", "txt"])
season_only = st.sidebar.checkbox("Focus on festival season (Oct-Nov)", value=True,
                                  help="Untick only if your file covers the whole year.")
default = next((os.path.join(HERE, f) for f in ("aqi_data.csv", "AIML Case Study.csv")
                if os.path.exists(os.path.join(HERE, f))), None)
if uploaded is None and default is None:
    st.title(TITLE)
    st.info("Upload your AQI CSV in the sidebar, or put it in this folder and name it `aqi_data.csv`.")
    st.stop()
st.sidebar.caption("Using: " + (uploaded.name if uploaded is not None else os.path.basename(default)))


@st.cache_resource(show_spinner="Cleaning the data and training the three models...")
def load_bundle(data_bytes, path, season):
    return P.run_all(io.BytesIO(data_bytes) if data_bytes else path, season)


try:
    b = load_bundle(uploaded.getvalue() if uploaded is not None else None, default, season_only)
except ValueError as e:
    st.error(str(e))
    st.stop()

df, best, table = b["df"], b["best"], b["table"]
tn, fp, fn, tp = b["cms"][best].ravel()

st.title(TITLE)
t_home, t_data, t_eda, t_models, t_pred = st.tabs(["Project", "Data", "EDA", "Models", "Predict"])

# ------------------------------------------------------------------ tab 1: project overview
with t_home:
    st.subheader("Problem statement")
    st.markdown(
        "Around Diwali, air quality in many Indian cities jumps sharply. This project predicts, **today**, whether "
        "**tomorrow's AQI will be Hazardous** (AQI above 200: Poor, Very Poor or Severe), using today's AQI, the recent "
        "trend, the city, the main pollutant and how close the date is to Diwali.\n\n"
        "**Who benefits:** residents (children, elderly, asthma patients), schools, hospitals and city authorities, "
        "who can issue advisories a day early.  \n"
        "**Why machine learning:** the link between today's air, the season, the city and festival timing is non-linear "
        "and differs between cities, so a model learned from past data suits it better than one fixed rule.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("City-days used", "{:,}".format(len(b["feat"])))
    c2.metric("Cities", len(b["cities"]))
    c3.metric("Hazardous tomorrow", "%.1f%%" % (b["y"].mean() * 100))
    c4.metric("Best model", best)
    st.caption("Workflow: Problem, Data, Cleaning, EDA, Feature engineering, Train-test split, 3 models, "
               "Evaluation, Comparison, Best model, Prediction.")
    with st.expander("Limitations and future scope"):
        st.markdown(
            "- City-average daily AQI only: no hourly, station-level or weather data (wind, temperature, humidity).\n"
            "- Few Diwali seasons in the data; Diwali dates differ by a day between regions.\n"
            "- Stubble burning and other events are not modelled.\n"
            "- Future: add weather and fire data, predict the AQI value, forecast several days ahead, "
            "try Gradient Boosting, deploy as an SMS/WhatsApp alert.")

# ------------------------------------------------------------------ tab 2: data
with t_data:
    st.subheader("Dataset")
    st.markdown("**Source:** CPCB daily AQI bulletins compiled by Urban Emissions (urbanemissions.info), "
                "one row per city per day.")
    raw = b["raw"]
    c1, c2 = st.columns(2)
    c1.metric("Rows", "{:,}".format(len(raw)))
    c2.metric("Columns", raw.shape[1])
    st.dataframe(raw.head(20))
    c1, c2 = st.columns(2)
    c1.markdown("**Data types**")
    c1.dataframe(raw.dtypes.astype(str).to_frame("type"))
    c2.markdown("**Missing values**")
    c2.dataframe(raw.isna().sum().to_frame("missing"))
    st.markdown("**Target:** tomorrow's AQI above 200 (1 = Hazardous, 0 = Safe).  \n"
                "**Features:** today's AQI, yesterday's AQI, change, 3-day and 7-day averages, month, winter flag, "
                "days from Diwali, Diwali-window flag, city, main pollutant.")
    st.subheader("Cleaning steps")
    st.code("\n".join(b["log"]))
    st.caption("Why: rows without a date, city or AQI cannot be used; impossible AQI values are errors; duplicates "
               "would count a city-day twice; cities with little history give unreliable trend features.")

# ------------------------------------------------------------------ tab 3: EDA
with t_eda:
    st.subheader("Exploratory data analysis")
    c1, c2 = st.columns(2)
    with c1:                                                         # Graph 1
        counts = df.Bucket.value_counts().reindex(P.BUCKETS)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(P.BUCKETS, counts.values, color=["#4caf50", "#9ccc65", "#fdd835", "#fb8c00", "#e53935", "#7b1fa2"])
        ax.set_title("1. How often does each AQI category occur?")
        ax.set_ylabel("City-days")
        plt.xticks(rotation=30)
        show(fig)
        st.info("%.1f%% of city-days are Hazardous, so the classes are imbalanced: accuracy alone can mislead, "
                "recall and F1 matter." % ((df.AQI > P.HAZARD).mean() * 100))
    with c2:                                                         # Graph 2
        top = df.groupby("City").AQI.median().nlargest(15).index
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df[df.City.isin(top)], y="City", x="AQI", order=top, color="skyblue", ax=ax)
        ax.axvline(P.HAZARD, color="red", ls="--")
        ax.set_title("2. AQI spread in the 15 most polluted cities")
        show(fig)
        st.info("%s has the highest median AQI: pollution depends strongly on the city." % top[0])

    w = df[df.off.between(-20, 30)].assign(haz=lambda d: (d.AQI > P.HAZARD) * 100)
    if w.empty:
        st.warning("No Diwali season is inside this data, so the Diwali graphs are skipped.")
    else:
        daily = w.groupby(["year", "off"])[["AQI", "haz"]].mean().reset_index()
        c1, c2 = st.columns(2)
        for col, c, title, label in [("AQI", c1, "3. Average AQI around Diwali", "Average AQI (all cities)"),
                                     ("haz", c2, "4. % of cities Hazardous around Diwali", "% of cities with AQI > 200")]:
            with c:
                fig, ax = plt.subplots(figsize=(6, 4))
                for y_, g_ in daily.groupby("year"):
                    ax.plot(g_.off, g_[col], marker="o", ms=3, label=str(y_))
                ax.axvline(0, color="red", ls="--", label="Diwali")
                ax.set_title(title)
                ax.set_xlabel("Days from Diwali")
                ax.set_ylabel(label)
                ax.legend()
                show(fig)
        inwin = df.off.between(-3, 7)
        st.info("Average AQI is %.0f inside the Diwali window (-3 to +7 days) vs %.0f on other days. Hazardous "
                "city-days: %.1f%% in the 3 days before Diwali vs %.1f%% in the 3 days after." % (
                    df[inwin].AQI.mean(), df[~inwin].AQI.mean(), w[w.off.between(-3, -1)].haz.mean(),
                    w[w.off.between(1, 3)].haz.mean()))

    c1, c2 = st.columns(2)
    with c1:                                                         # Graph 5
        inw = df.off.between(-3, 7)
        share = pd.DataFrame({"Diwali window": df[inw].Prom.value_counts(normalize=True),
                              "Other days": df[~inw].Prom.value_counts(normalize=True)}).fillna(0) * 100
        share = share[(share > 2).any(axis=1)]
        fig, ax = plt.subplots(figsize=(6, 4))
        share.plot(kind="bar", ax=ax, rot=0, color=["#e8704c", "#4c9be8"])
        ax.set_title("5. Main pollutant: Diwali window vs other days")
        ax.set_ylabel("% of city-days")
        show(fig)
        st.info("Share of days each pollutant is the main driver of AQI, in and outside the Diwali window.")
    with c2:                                                         # Graph 6
        jump = (df[df.off.between(1, 3)].groupby("City").AQI.mean()
                - df[df.off.between(-3, -1)].groupby("City").AQI.mean()).dropna().nlargest(10)
        if len(jump):
            fig, ax = plt.subplots(figsize=(6, 4))
            jump[::-1].plot(kind="barh", ax=ax, color="firebrick")
            ax.set_title("6. Biggest AQI jumps after Diwali")
            ax.set_xlabel("AQI: 3 days after minus 3 days before")
            show(fig)
            st.info("%s had the biggest jump (+%.0f AQI points): the festival effect is strongest in some cities."
                    % (jump.index[0], jump.iloc[0]))

# ------------------------------------------------------------------ tab 4: models
with t_models:
    st.subheader("Model comparison")
    st.caption(b["note"])
    st.dataframe(table)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        table[["Accuracy", "Precision", "Recall", "F1-Score"]].plot(kind="bar", ax=ax, rot=0, ylim=(0, 1.05))
        ax.set_title("Model comparison")
        ax.legend(loc="lower right")
        show(fig)
    with c2:
        fig, axs = plt.subplots(1, 3, figsize=(9, 3.2))
        for i, (ax, (n, cm)) in enumerate(zip(axs, b["cms"].items())):
            ConfusionMatrixDisplay(cm, display_labels=["Safe", "Hazard"]).plot(ax=ax, colorbar=False, cmap="Blues")
            ax.set_title(n, fontsize=9)
            ax.grid(False)
            if i:                                   # keep the y-axis label only on the first matrix
                ax.set_ylabel("")
                ax.set_yticks([])
        fig.tight_layout()
        show(fig)
    st.success("Best model: **%s** (highest F2-score). It caught %d of %d hazardous days, missed %d and raised %d "
               "false alarms." % (best, tp, tp + fn, fn, fp))
    if b["win_recall"] is not None:
        st.caption("Inside the Diwali window (%d city-days in the test set): recall %.2f, precision %.2f." % (
            b["win_n"], b["win_recall"], b["win_precision"]))
    st.markdown("**Why recall?** Missing a truly hazardous day is worse than a false alarm, so we want to catch as "
                "many hazardous days as possible. A model that always says 'Hazardous' would have perfect recall, so "
                "the F2-score (recall weighted higher than precision) picks the best model.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Top 10 features (Random Forest)**")
        fig, ax = plt.subplots(figsize=(6, 4))
        b["imp"][::-1].plot(kind="barh", ax=ax, color="teal")
        show(fig)
    with c2:
        st.markdown("**Do the Diwali features help?** (Random Forest)")
        st.dataframe(b["ablation"])
        if b["one_season"]:
            st.caption("With one Diwali season, days-from-Diwali partly stands in for the calendar date, so this "
                       "comparison can look better than it really is. Mention this in the report.")

# ------------------------------------------------------------------ tab 5: predict
with t_pred:
    st.subheader("Early-warning tool")
    with st.form("predict"):
        c1, c2 = st.columns(2)
        city = c1.selectbox("City", b["cities"], index=b["cities"].index("Delhi") if "Delhi" in b["cities"] else 0)
        date = c2.date_input("Today's date", value=pd.Timestamp("2025-10-21").date())
        aqi_today = c1.number_input("Today's AQI", min_value=0, max_value=500, value=350, step=1)
        aqi_yday = c2.number_input("Yesterday's AQI", min_value=0, max_value=500, value=300, step=1)
        pollutant = c1.selectbox("Main pollutant today", b["pollutants"],
                                 index=b["pollutants"].index("PM2.5") if "PM2.5" in b["pollutants"] else 0)
        go = st.form_submit_button("Predict tomorrow's risk")
    if go:
        prob, off = P.predict_one(b, city, date, aqi_today, aqi_yday, pollutant)
        c1, c2 = st.columns(2)
        c1.metric("Risk of Hazardous AQI tomorrow", "%d%%" % round(prob * 100))
        c2.metric("Days from nearest Diwali", "%+d" % off)
        st.progress(min(max(prob, 0.0), 1.0))
        if prob >= 0.5:
            st.error("HAZARDOUS expected tomorrow in %s. Suggested advisory: limit outdoor activity, avoid morning "
                     "walks, use an N95 mask outdoors, and keep windows closed." % city)
        else:
            st.success("Tomorrow is likely SAFE in %s (AQI not above 200)." % city)
        if not any(c.startswith("City_") for c in b["X"].columns):
            st.caption("Note: with one Diwali season the model does not use the city, so the city choice has no effect.")

    st.subheader("City risk board")
    st.caption("Pick a date from the data: predicted risk for every city, next to what really happened the next day.")
    dates = sorted(b["feat"].Date.dt.date.unique())
    after = [d for d in dates if d >= (pd.Timestamp(P.DIWALI.get(max(b["feat"].year), "2025-10-20")) + pd.Timedelta(days=1)).date()]
    pick = st.selectbox("Date", dates, index=dates.index(after[0]) if after else 0)
    board = P.risk_board(b, pick)
    st.dataframe(board.head(20))
