"""
Data + ML Pipeline for:
Beyond the Fireworks: Machine Learning-Based Next-Day AQI Risk Prediction for Indian Cities
Tailored for Oct-Nov 2025 CPCB AQI Bulletin Data.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, fbeta_score, confusion_matrix
)

HAZARD = 200          # AQI > 200 (Poor, Very Poor, Severe) = Hazardous
SEED = 42
SEASON_MONTHS = [10, 11]
WINTER_MONTHS = [10, 11, 12, 1, 2]

# Diwali dates (2025 Diwali was Oct 20, 2025)
DIWALI = {
    2015: "2015-11-11", 2016: "2016-10-30", 2017: "2017-10-19",
    2018: "2018-11-07", 2019: "2019-10-27", 2020: "2020-11-14",
    2021: "2021-11-04", 2022: "2022-10-24", 2023: "2023-11-12",
    2024: "2024-10-31", 2025: "2025-10-20", 2026: "2026-11-08"
}
DW = np.array(pd.to_datetime(list(DIWALI.values())).values, dtype="datetime64[D]")

STANDARD_COLS = ["date", "city", "no_stations", "aqi_category", "aqi", "prominent_pollutant", "year", "month"]
NUM_COLS = ["AQI", "prev", "change", "roll3", "roll7", "month", "winter"]
DIWALI_COLS = ["dw_days", "dw_window"]
BUCKETS = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]


def days_to_diwali(dates):
    """Signed days from each date to the nearest Diwali (negative = before Diwali)."""
    d = pd.to_datetime(pd.Series(dates)).values.astype("datetime64[D]")
    diff = (d[:, None] - DW).astype(int)
    return diff[np.arange(len(d)), np.abs(diff).argmin(1)]


# ----------------------------------------------------------------------------- 1. LOAD
def load_raw(source):
    """Read CSV safely even if headers are shifted or repeated."""
    raw = pd.read_csv(source, header=None, dtype=str, encoding="utf-8-sig", skip_blank_lines=True)
    raw = raw.dropna(how="all").dropna(axis=1, how="all").reset_index(drop=True)
    names = [[str(v).strip().lower() for v in r] for r in raw.head(10).values.tolist()]
    hdr = next((i for i, r in enumerate(names) if "date" in r and "city" in r), None)

    if hdr is not None:
        raw.columns = names[hdr]
        raw = raw.iloc[hdr + 1:]
    elif raw.shape[1] == len(STANDARD_COLS):
        raw.columns = STANDARD_COLS
    else:
        raise ValueError(f"Could not find header row. First row: {names[0]}")

    raw = raw.rename(columns={"index value": "aqi", "prominent pollutant": "prominent_pollutant"})
    if not {"date", "city", "aqi"} <= set(raw.columns):
        raise ValueError(f"Required columns (date, city, aqi) missing. Found: {list(raw.columns)}")

    raw = raw[raw["date"].astype(str).str.strip().str.lower() != "date"].reset_index(drop=True)
    for c in ("no_stations", "aqi", "year"):
        if c in raw:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    d = pd.to_datetime(raw["date"], errors="coerce")
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce", dayfirst=True) if d.isna().mean() > 0.2 else d
    return raw


# ----------------------------------------------------------------------------- 2. CLEAN
def clean(raw, season_only=True):
    """Clean raw AQI records and return (clean_df, log_messages)."""
    log = [f"Rows loaded: {len(raw)}"]
    df = pd.DataFrame({
        "Date": raw["date"],
        "City": raw["city"].astype(str).str.strip(),
        "AQI": raw["aqi"],
        "Prom": raw["prominent_pollutant"] if "prominent_pollutant" in raw else "Unknown"
    })

    def step(msg):
        log.append(f"{msg}: {len(df)} rows")

    df = df.dropna(subset=["Date", "City", "AQI"])
    step("After removing missing date / city / AQI")

    df = df[(df.AQI >= 0) & (df.AQI <= 1000)]
    step("After removing invalid AQI values (<0 or >1000)")

    if season_only:
        df = df[df.Date.dt.month.isin(SEASON_MONTHS)]
        step("After keeping Oct-Nov festival season only")

    df = df.drop_duplicates(["City", "Date"]).sort_values(["City", "Date"]).reset_index(drop=True)
    step("After removing duplicate city-date entries")

    if df.empty:
        raise ValueError("No valid rows left after cleaning:\n" + "\n".join(log))

    # Keep cities with at least 4 consecutive/valid days so lag features work
    cnt = df.City.value_counts()
    min_days = max(4, int(0.5 * df.Date.nunique()))
    valid_cities = cnt[cnt >= min_days].index
    if len(valid_cities) >= 5:
        df = df[df.City.isin(valid_cities)].reset_index(drop=True)
        step(f"After keeping cities with >= {min_days} days of data")

    df["Prom"] = (
        df.Prom.astype(str)
        .str.split(",").str[0]
        .str.strip()
        .replace({"nan": "Unknown", "": "Unknown", "None": "Unknown"})
    )
    df["Bucket"] = pd.cut(df.AQI, [-np.inf, 50, 100, 200, 300, 400, np.inf], labels=BUCKETS)
    df["year"] = df.Date.dt.year
    df["month"] = df.Date.dt.month
    df["off"] = (df.Date - pd.to_datetime(df.year.map(DIWALI))).dt.days
    return df, log


# ----------------------------------------------------------------------------- 3. FEATURES
def make_features(df, season_only=True):
    """Engineer lag, rolling, and Diwali proximity features for next-day prediction."""
    d = df.copy().sort_values(["City", "Date"]).reset_index(drop=True)
    g = d.groupby("City")

    # Yesterday's AQI (fallback to today's if gap > 1 day)
    d["prev"] = g.AQI.shift(1).where((d.Date - g.Date.shift(1)).dt.days == 1, d.AQI)
    d["change"] = d.AQI - d.prev
    d["roll3"] = g.AQI.transform(lambda s: s.rolling(3, min_periods=1).mean())
    d["roll7"] = g.AQI.transform(lambda s: s.rolling(7, min_periods=1).mean())

    # Target: Is tomorrow's AQI > 200 (Hazardous)?
    d["target"] = (g.AQI.shift(-1) > HAZARD).astype(int)

    # Keep only rows where tomorrow's actual observation exists (consecutive day)
    d = d[(g.Date.shift(-1) - d.Date).dt.days == 1].copy().reset_index(drop=True)

    if len(d) < 20:
        raise ValueError(
            f"Only {len(d)} consecutive city-day pairs found. Make sure your CSV has consecutive dates "
            "(export ALL rows from Datasette, not just the first 1,000 rows)."
        )

    o = days_to_diwali(d.Date)
    d["dw_days"] = o.clip(-30, 30)
    d["dw_window"] = ((o >= -3) & (o <= 7)).astype(int)
    d["winter"] = d.month.isin(WINTER_MONTHS).astype(int)

    if season_only:
        d = d[d.month.isin(SEASON_MONTHS)].reset_index(drop=True)

    X = pd.concat([
        d[NUM_COLS + DIWALI_COLS],
        pd.get_dummies(d.City, prefix="City", dtype=int),
        pd.get_dummies(d.Prom, prefix="Poll", dtype=int)
    ], axis=1)

    return d, X, d.target


# ----------------------------------------------------------------------------- 4. SPLIT
def split(feat, X, y):
    """
    If multiple years exist: train on earlier years, test on newest year.
    If only Oct-Nov 2025 exists: hold out 20% of cities (GroupShuffleSplit) to prevent data leakage.
    """
    if feat.year.nunique() >= 2:
        m = feat.year == feat.year.max()
        if m.sum() >= 300 and y[m].sum() >= 20 and y[~m].sum() >= 20:
            return X, m, f"Trained on earlier years, tested on the {feat.year.max()} Diwali season."

    X = X.drop(columns=[c for c in X if c.startswith("City_")])
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    _, idx = next(gss.split(X, y, groups=feat.City))
    test = pd.Series(False, index=X.index)
    test.iloc[idx] = True
    n_test_cities = feat[test].City.nunique()
    note = (
        f"Single season (Oct–Nov 2025) detected: held out {n_test_cities} unseen cities (20%) for testing "
        "to evaluate generalization across Indian cities without city-level leakage."
    )
    return X, test, note


# ----------------------------------------------------------------------------- 5. MODELS
def make_rf():
    return RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=SEED
    )


def make_models():
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED)
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=15, class_weight="balanced", random_state=SEED
        ),
        "Random Forest": make_rf()
    }


def score(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
        "F2-Score": fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    }


def run_all(source, season_only=True):
    """Run full pipeline and return bundle dictionary for Streamlit UI."""
    raw = load_raw(source)
    df, log = clean(raw, season_only)
    feat, X, y = make_features(df, season_only)
    X, test, note = split(feat, X, y)

    Xtr, Xte, ytr, yte = X[~test], X[test], y[~test], y[test]
    models, res, cms = make_models(), {}, {}

    for name, model in models.items():
        preds = model.fit(Xtr, ytr).predict(Xte)
        res[name] = score(yte, preds)
        cms[name] = confusion_matrix(yte, preds, labels=[0, 1])

    table = pd.DataFrame(res).T.round(4)
    best = table["F2-Score"].idxmax()

    win = (Xte.dw_window == 1).values
    pw = models[best].predict(Xte[win]) if win.sum() and yte[win].sum() else None

    imp = pd.Series(models["Random Forest"].feature_importances_, index=X.columns).nlargest(10)
    no_dw_cols = [c for c in X if c not in DIWALI_COLS]
    abl = pd.DataFrame({
        "With Diwali features": res["Random Forest"],
        "Without Diwali features": score(yte, make_rf().fit(Xtr[no_dw_cols], ytr).predict(Xte[no_dw_cols]))
    }).round(4)

    return dict(
        raw=raw, df=df, log=log, feat=feat, X=X, y=y, test=test, note=note,
        models=models, table=table, cms=cms, best=best, imp=imp, ablation=abl,
        cities=sorted(df.City.unique()), pollutants=sorted(feat.Prom.unique()),
        one_season=(feat.year.nunique() == 1), win_n=int(win.sum()),
        win_recall=None if pw is None else recall_score(yte[win], pw, zero_division=0),
        win_precision=None if pw is None else precision_score(yte[win], pw, zero_division=0)
    )


# ----------------------------------------------------------------------------- 6. PREDICT
def predict_one(b, city, date, aqi_today, aqi_yesterday, pollutant):
    """Predict probability (0-1) that tomorrow's AQI will be Hazardous (>200)."""
    dt = pd.Timestamp(date)
    o = int(days_to_diwali([dt])[0])
    row = {
        "AQI": aqi_today,
        "prev": aqi_yesterday,
        "change": aqi_today - aqi_yesterday,
        "roll3": (aqi_today + aqi_yesterday) / 2.0,
        "roll7": (aqi_today + aqi_yesterday) / 2.0,
        "month": dt.month,
        "winter": int(dt.month in WINTER_MONTHS),
        "dw_days": int(np.clip(o, -30, 30)),
        "dw_window": int(-3 <= o <= 7)
    }
    x = pd.DataFrame([row]).reindex(columns=b["X"].columns, fill_value=0)
    for col in (f"City_{city}", f"Poll_{pollutant}"):
        if col in x:
            x[col] = 1
    prob = float(b["models"][b["best"]].predict_proba(x)[0, 1])
    return prob, o


def risk_board(b, date):
    """Return city-by-city predicted risk vs actual next-day outcome for a given date."""
    f = b["feat"]
    rows = f[f.Date == pd.Timestamp(date)]
    if rows.empty:
        return pd.DataFrame()
    p = b["models"][b["best"]].predict_proba(b["X"].loc[rows.index])[:, 1]
    out = pd.DataFrame({
        "City": rows.City.values,
        "AQI today": rows.AQI.round(0).astype(int).values,
        "Predicted risk %": (p * 100).round(0).astype(int),
        "Actual tomorrow": np.where(rows.target.values == 1, "Hazardous", "Safe"),
        "Data split": np.where(b["test"].loc[rows.index].values, "Test (unseen)", "Training")
    })
    return out.sort_values("Predicted risk %", ascending=False).reset_index(drop=True)