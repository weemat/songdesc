
import os
import re
import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =========================
# Utility + core components
# =========================

def norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    return re.sub(r"\s+", " ", s.strip().lower())

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

# =====================
# Keyword coverage setup
# =====================

LEXICON = {
    "tempo_bpm": [r"\b\d{2,3}\s?bpm\b", r"\btempo\s*\d{2,3}\b"],
    "key_signature": [r"\b([A-G])( ?#| ?♯| ?b| ?♭)?\s?(major|minor|maj|min|m)\b"],
    "time_signature": [r"\b\d{1,2}/\d{1,2}\b"],
    "duration": [r"\b\d{1,2}:\d{2}\b"],
    "instrumentation": [
        r"\b(sub-?bass|808s?|bassline|kick|hi-?hats?|snare|claps?|toms?|percussion|shaker|ride)\b",
        r"\b(synths?|supersaws?|pads?|arps?|leads?|plucks?|stabs?)\b",
        r"\b(piano|keys?|strings?|guitars?|brass|woodwinds?|flute|violin|cello|sax(ophone)?)\b",
        r"\b(vocal chops?|falsetto|harmonies|choir|ad-?libs?)\b",
    ],
    "production_mix": [
        r"\b(side-?chain(ing)?|sidechain)\b",
        r"\b(saturation|tape sheen|distortion)\b",
        r"\b(reverb|delay|plate|slap-?back)\b",
        r"\b(compression|bus compression|limiting|master(ing)?)\b",
        r"\b(stereo (width|imaging)|mono|panning)\b",
        r"\b(transient|attack|release)\b",
    ],
    "structure_form": [
        r"\b(verse|pre-?chorus|chorus|hook|bridge|drop|break(down)?|build(up)?|outro|intro)\b",
        r"\b(8-?bar|16-?bar|turnaround|phrasing)\b",
    ],
    "genre_tags": [r"\b(house|tech house|deep house|melodic house|drum( |-)?and( |-)?bass|dnb|edm|pop|trap)\b"],
    "sonic_descriptors": [
        r"\b(warm|airy|glassy|crisp|gnarly|gritty|lush|moody|euphoric|hypnotic|punchy|tight|minimal|spacious|dreamy|dark|bright)\b",
        r"\b(rolling|groove-?locked|swing|swung|syncopated|off-?beat)\b",
    ],
}

WEIGHTS = {
    "tempo_bpm": 2.0,
    "key_signature": 2.0,
    "time_signature": 1.0,
    "duration": 1.0,
    "instrumentation": 1.5,
    "production_mix": 1.5,
    "structure_form": 1.0,
    "genre_tags": 0.5,
    "sonic_descriptors": 1.0,
}

LEX_RE = {cat: [re.compile(p, flags=re.IGNORECASE) for p in pats] for cat, pats in LEXICON.items()}

def keyword_features(text: str) -> Dict[str, int]:
    if not isinstance(text, str):
        text = ""
    feats = {}
    for cat, regs in LEX_RE.items():
        count = 0
        for rg in regs:
            count += len(rg.findall(text))
        feats[cat] = count
    feats["total_hits"] = sum(v for v in feats.values())
    return feats

def keyword_score(feats: Dict[str, int]) -> float:
    s = 0.0
    for cat, w in WEIGHTS.items():
        s += w * feats.get(cat, 0)
    return s

# ===========================
# Embedding layer (OpenAI API)
# ===========================

def embed_texts(texts: List[str], model: str = "text-embedding-3-large") -> np.ndarray:
    """
    Returns an array of shape (len(texts), 3072) using OpenAI embeddings.
    If OPENAI_API_KEY is missing, raises RuntimeError to signal skipping.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set. Skipping embeddings.")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        # Batch request
        resp = client.embeddings.create(model=model, input=texts)
        vecs = [d.embedding for d in resp.data]
        return np.array(vecs, dtype=np.float32)
    except Exception as e:
        raise RuntimeError(f"Embedding call failed: {e}")

# =====================
# Main evaluation logic
# =====================

def minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi <= lo:
        return pd.Series([0.0]*len(series), index=series.index)
    return (series - lo) / (hi - lo)

def run_default():
    # Paths relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    a_path = os.path.join(script_dir, "/Users/bowenxia/songs.csv")
    b_path = os.path.join(script_dir, "/Users/bowenxia/musicsheets_final_refined.csv")
    out_csv = os.path.join(script_dir, "/Users/bowenxia/mixed_eval_report.csv")
    out_json = os.path.join(script_dir, "/Users/bowenxia/mixed_eval_summary.json")
    model = "text-embedding-3-large"

    # Load CSVs
    if not os.path.exists(a_path):
        raise FileNotFoundError(f"Old file not found: {a_path}")
    if not os.path.exists(b_path):
        raise FileNotFoundError(f"New file not found: {b_path}")

    df_a = pd.read_csv(a_path)
    df_b = pd.read_csv(b_path)

    # Normalize key and merge
    def build_key(df):
        return df["title"].apply(norm) + " | " + df["artists"].apply(norm)

    df_a["_k"] = build_key(df_a)
    df_b["_k"] = build_key(df_b)

    a_col = "song_desc" if "song_desc" in df_a.columns else "semantic_description"
    b_col = "semantic_description"

    merged = df_b.merge(df_a[["_k", a_col]], on="_k", how="inner")
    merged.rename(columns={a_col: "semantic_a", b_col: "semantic_b"}, inplace=True)

    # Deterministic metrics per row
    rows = []
    for _, r in merged.iterrows():
        a = str(r["semantic_a"])
        b = str(r["semantic_b"])

        a_len = len(a); b_len = len(b)
        a_len_ok = 300 <= a_len <= 500
        b_len_ok = 300 <= b_len <= 500

        fa = keyword_features(a); fb = keyword_features(b)
        score_a = keyword_score(fa); score_b = keyword_score(fb)

        rows.append({
            "title": r["title"],
            "artists": r["artists"],
            "len_a": a_len, "len_b": b_len,
            "len_ok_a": a_len_ok, "len_ok_b": b_len_ok,
            "kw_score_a": score_a, "kw_score_b": score_b,
            **{f"A_{k}": v for k, v in fa.items()},
            **{f"B_{k}": v for k, v in fb.items()},
        })

    det = pd.DataFrame(rows)
    det["kw_score_delta"] = det["kw_score_b"] - det["kw_score_a"]

    # Try embeddings; skip gracefully if unavailable
    embed_summary = {}
    try:
        a_texts = merged["semantic_a"].astype(str).tolist()
        b_texts = merged["semantic_b"].astype(str).tolist()

        vec_a = embed_texts(a_texts, model=model)
        vec_b = embed_texts(b_texts, model=model)

        # Pairwise similarity per row
        pairwise_sim = np.array([
            cosine_sim(vec_a[i], vec_b[i]) for i in range(len(vec_a))
        ], dtype=np.float32)

        # Generic baseline centroid
        generic_corpus = [
            "This track features a catchy rhythm and uplifting melodies with a steady beat.",
            "The song blends electronic elements with a driving bass and polished production.",
            "An energetic arrangement with vibrant synths and atmospheric textures throughout.",
            "A modern dance track with a strong groove and memorable hooks.",
        ]
        generic_vec = embed_texts(generic_corpus, model=model).mean(axis=0)
        sim_a_generic = np.array([cosine_sim(v, generic_vec) for v in vec_a], dtype=np.float32)
        sim_b_generic = np.array([cosine_sim(v, generic_vec) for v in vec_b], dtype=np.float32)
        generic_delta = sim_a_generic - sim_b_generic  # positive => B less generic

        # Self-similarity (sampled mean off-diagonal)
        def mean_self_similarity(vecs: np.ndarray, sample: int = 5000) -> float:
            n = len(vecs)
            if n < 2:
                return 0.0
            rng = np.random.default_rng(42)
            total = 0.0; cnt = 0
            for _ in range(min(sample, n*(n-1)//2)):
                i = int(rng.integers(0, n))
                j = int(rng.integers(0, n-1))
                if j >= i: j += 1
                total += cosine_sim(vecs[i], vecs[j])
                cnt += 1
            return float(total / max(1, cnt))

        selfsim_a = mean_self_similarity(vec_a)
        selfsim_b = mean_self_similarity(vec_b)

        det["embed_pairwise_sim"] = pairwise_sim
        det["embed_generic_delta"] = generic_delta

        embed_summary = {
            "pairwise_sim_mean": float(pairwise_sim.mean()),
            "pairwise_sim_median": float(np.median(pairwise_sim)),
            "generic_delta_mean": float(generic_delta.mean()),
            "self_similarity_A": float(selfsim_a),
            "self_similarity_B": float(selfsim_b),
        }
    except RuntimeError as e:
        print(f"[Warning] {e}. Proceeding without embedding metrics.")

    # Mixed score
    det["deterministic_norm"] = minmax(det["kw_score_b"]) - minmax(det["kw_score_a"])
    if "embed_pairwise_sim" in det.columns and "embed_generic_delta" in det.columns:
        det["embed_pairwise_norm"] = minmax(det["embed_pairwise_sim"])
        det["embed_generic_norm"] = minmax(det["embed_generic_delta"])
        det["mixed_improvement"] = 0.4*det["deterministic_norm"] + 0.3*det["embed_pairwise_norm"] + 0.3*det["embed_generic_norm"]
    else:
        det["mixed_improvement"] = det["deterministic_norm"]

    # Summary
    summary = {
        "rows_compared": int(len(det)),
        "avg_kw_score_A": float(det["kw_score_a"].mean()),
        "avg_kw_score_B": float(det["kw_score_b"].mean()),
        "avg_kw_delta": float(det["kw_score_delta"].mean()),
        "pct_len_ok_A": float((det["len_ok_a"]==True).mean()),
        "pct_len_ok_B": float((det["len_ok_b"]==True).mean()),
        "avg_mixed_improvement": float(det["mixed_improvement"].mean()),
    }
    summary.update({f"embedding_{k}": v for k,v in ({} if not embed_summary else embed_summary).items()})

    # Save
    det.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print("== Mixed Evaluation Complete ==")
    print(f"Rows compared: {summary['rows_compared']}")
    print(f"Avg keyword score delta (B - A): {summary['avg_kw_delta']:.3f}")
    if 'embedding_pairwise_sim_mean' in summary:
        print(f"Mean A↔B pairwise similarity: {summary['embedding_pairwise_sim_mean']:.3f}")
    if 'embedding_generic_delta_mean' in summary:
        print(f"Mean genericness delta (A - B): {summary['embedding_generic_delta_mean']:.3f}")
    print(f"Avg mixed improvement score: {summary['avg_mixed_improvement']:.3f}")
    print(f"Report CSV: {out_csv}")
    print(f"Summary JSON: {out_json}")

if __name__ == "__main__":
    run_default()
