import sys
import subprocess
import importlib.util


def ensure_packages():
    packages = {
        "pandas": "pandas",
        "numpy": "numpy",
        "matplotlib": "matplotlib",
        "sklearn": "scikit-learn",
        "requests": "requests",
        "scipy": "scipy",
    }

    for import_name, pip_name in packages.items():
        if importlib.util.find_spec(import_name) is None:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )


ensure_packages()

import os
import re
import time
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from scipy.stats import kruskal


def fetch_gdelt_window(query, start_dt, end_dt, max_records=250):
    base_url = "https://api.gdeltproject.org/api/v2/doc/doc"
    headers = {"User-Agent": "Mozilla/5.0"}

    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": max_records,
        "STARTDATETIME": start_dt,
        "ENDDATETIME": end_dt,
    }

    for attempt in range(4):
        try:
            response = requests.get(base_url, params=params, headers=headers, timeout=30)

            if response.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue

            response.raise_for_status()

            if not response.text.strip():
                time.sleep(5)
                continue

            data = response.json()
            rows = []

            for article in data.get("articles", []):
                rows.append({
                    "title": article.get("title"),
                    "source_country": article.get("sourcecountry"),
                    "date": article.get("seendate"),
                    "url": article.get("url")
                })

            return pd.DataFrame(rows, columns=["title", "source_country", "date", "url"])

        except Exception:
            time.sleep(5)

    return pd.DataFrame(columns=["title", "source_country", "date", "url"])


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_clean_data(force_refetch=False):
    query = '(Palestine OR Gaza OR "West Bank" OR "Israel conflict")'
    cache_file = "gdelt_palestine_feb_apr_2026.csv"

    time_windows = [
        ("20260201000000", "20260215235959"),
        ("20260216000000", "20260228235959"),
        ("20260301000000", "20260315235959"),
        ("20260316000000", "20260331235959"),
        ("20260401000000", "20260415235959"),
        ("20260416000000", "20260430235959"),
    ]

    if (not force_refetch) and os.path.exists(cache_file):
        raw_df = pd.read_csv(cache_file)
    else:
        parts = []

        for start_dt, end_dt in time_windows:
            part = fetch_gdelt_window(query, start_dt, end_dt)
            parts.append(part)
            time.sleep(8)

        raw_df = pd.concat(parts, ignore_index=True)
        raw_df.to_csv(cache_file, index=False)

    raw_rows = len(raw_df)

    df = raw_df.copy()

    df = df.drop_duplicates(subset=["title", "source_country", "date"])
    after_dedup = len(df)

    df = df.dropna(subset=["title", "source_country", "date"]).copy()
    after_missing = len(df)

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%dT%H%M%SZ", errors="coerce")
    df = df.dropna(subset=["date"]).copy()

    df = df[df["date"] < "2026-05-01"].copy()
    after_dates = len(df)

    df["clean_text"] = df["title"].apply(clean_text)
    df["headline_length_words"] = df["clean_text"].str.split().apply(len)

    df = df[df["headline_length_words"] >= 4].copy()
    after_min_words = len(df)

    top_countries = df["source_country"].value_counts().head(4).index.tolist()
    df_selected = df[df["source_country"].isin(top_countries)].copy()
    df_selected["month"] = df_selected["date"].dt.to_period("M").astype(str)

    ml_countries = df_selected["source_country"].value_counts().head(3).index.tolist()
    df_ml = df_selected[df_selected["source_country"].isin(ml_countries)].copy()

    min_count = df_ml["source_country"].value_counts().min()

    balanced_parts = []
    for country in ml_countries:
        country_rows = df_ml[df_ml["source_country"] == country].sample(
            n=min_count,
            random_state=42
        )
        balanced_parts.append(country_rows)

    df_ml = pd.concat(balanced_parts, ignore_index=True)

    df_selected.to_csv("cleaned_palestine_headlines_selected.csv", index=False)
    df_ml.to_csv("cleaned_palestine_headlines_ml.csv", index=False)

    date_range = f"{df_selected['date'].min().date()} to {df_selected['date'].max().date()}"

    summary = pd.DataFrame({
        "Metric": [
            "Raw collected rows",
            "After duplicate removal",
            "After dropping missing values",
            "After valid date filtering",
            "After headline-length filtering",
            "Rows in selected dataset",
            "Rows in ML dataset",
            "Date range"
        ],
        "Value": [
            raw_rows,
            after_dedup,
            after_missing,
            after_dates,
            after_min_words,
            len(df_selected),
            len(df_ml),
            date_range
        ]
    })

    return df_selected, df_ml, summary


def plot_country_counts(df):
    counts = df["source_country"].value_counts()
    date_min = df["date"].min().date()
    date_max = df["date"].max().date()

    plt.figure(figsize=(8, 4))
    bars = plt.bar(counts.index, counts.values)

    plt.title(f"Headlines by Source Country\n{date_min} to {date_max}", fontsize=14)
    plt.xlabel("Source Country")
    plt.ylabel("Number of Headlines")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1,
            str(int(height)),
            ha="center"
        )

    plt.tight_layout()
    plt.show()


def plot_monthly_counts(df):
    monthly_counts = (
        df.groupby(["month", "source_country"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()

    plt.figure(figsize=(8, 4))

    for country in monthly_counts.columns:
        plt.plot(
            monthly_counts.index,
            monthly_counts[country],
            marker="o",
            linewidth=2,
            label=country
        )

    plt.title(f"Monthly Headline Coverage\n{date_min} to {date_max}", fontsize=14)
    plt.xlabel("Month")
    plt.ylabel("Number of Headlines")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_headline_length(df):
    means = (
        df.groupby("source_country")["headline_length_words"]
        .mean()
        .sort_values(ascending=False)
        .round(1)
    )

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()

    plt.figure(figsize=(8, 4))
    bars = plt.bar(means.index, means.values)

    plt.title(f"Average Headline Length by Country\n{date_min} to {date_max}", fontsize=14)
    plt.xlabel("Source Country")
    plt.ylabel("Average Number of Words")

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.1,
            f"{height:.1f}",
            ha="center"
        )

    plt.tight_layout()
    plt.show()

    groups = [
        df[df["source_country"] == country]["headline_length_words"].values
        for country in df["source_country"].value_counts().index
    ]

    stat, p_value = kruskal(*groups)

    return round(stat, 4), round(p_value, 6)


def run_ml(df_ml):
    X = df_ml["clean_text"]
    y = df_ml["source_country"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=4000, ngram_range=(1, 2), min_df=1)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    baseline_label = y_train.mode()[0]
    baseline_preds = np.full(len(y_test), baseline_label)
    baseline_acc = accuracy_score(y_test, baseline_preds)

    log_reg = LogisticRegression(max_iter=2000, random_state=42)
    log_reg.fit(X_train_tfidf, y_train)
    log_preds = log_reg.predict(X_test_tfidf)
    log_acc = accuracy_score(y_test, log_preds)

    naive_bayes = MultinomialNB()
    naive_bayes.fit(X_train_tfidf, y_train)
    nb_preds = naive_bayes.predict(X_test_tfidf)
    nb_acc = accuracy_score(y_test, nb_preds)

    results = pd.DataFrame({
        "Model": ["Baseline", "Logistic Regression", "Naive Bayes"],
        "Accuracy": [baseline_acc, log_acc, nb_acc]
    })

    return results


def plot_model_accuracy(results):
    results = results.sort_values("Accuracy", ascending=False)

    plt.figure(figsize=(7, 4))
    bars = plt.bar(results["Model"], results["Accuracy"])

    plt.title("Model Accuracy Comparison", fontsize=14)
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.02,
            f"{height:.2f}",
            ha="center"
        )

    plt.tight_layout()
    plt.show()
