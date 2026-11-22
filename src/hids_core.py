"""Shared code for the HIDS project: schema, feature engineering,
prediction helper and the automated-response engine.

This single file is imported by the training script, the Streamlit app,
the Flask app and the Hugging Face (Gradio) Space, so every deployment
behaves identically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Schema of the NSL-KDD files (41 features + label + difficulty)
# --------------------------------------------------------------------------
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files",
    "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    "label", "difficulty",
]
FEATURES = COLUMNS[:41]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

# Fine-grained attack name -> 5 coarse classes (standard NSL-KDD taxonomy)
ATTACK_TO_CLASS = {
    "normal": "Normal",
    # Denial of service
    **{a: "DoS" for a in [
        "back", "land", "neptune", "pod", "smurf", "teardrop", "apache2",
        "udpstorm", "processtable", "worm", "mailbomb"]},
    # Probing / reconnaissance
    **{a: "Probe" for a in ["satan", "ipsweep", "nmap", "portsweep", "mscan", "saint"]},
    # Remote-to-local: remote attacker gains a local account (password guessing etc.)
    **{a: "R2L" for a in [
        "guess_passwd", "ftp_write", "imap", "phf", "multihop", "warezmaster",
        "warezclient", "spy", "xlock", "xsnoop", "snmpguess", "snmpgetattack",
        "httptunnel", "sendmail", "named"]},
    # User-to-root: local user escalates privileges
    **{a: "U2R" for a in [
        "buffer_overflow", "loadmodule", "rootkit", "perl", "sqlattack",
        "xterm", "ps"]},
}
CLASSES = ["Normal", "DoS", "Probe", "R2L", "U2R"]

# Human-readable meaning, used in the apps
CLASS_INFO = {
    "Normal": "Legitimate activity.",
    "DoS": "Denial-of-service: flooding the host to exhaust its resources.",
    "Probe": "Reconnaissance: port/host scanning to find weaknesses.",
    "R2L": "Remote-to-local: remote attacker gaining a local account "
           "(e.g. password guessing / brute force).",
    "U2R": "User-to-root: a local user escalating to administrator "
           "privileges (e.g. buffer overflow, rootkit).",
}

# --------------------------------------------------------------------------
# Automated response engine (rule-based policy on top of the ML prediction)
# --------------------------------------------------------------------------
RESPONSE_POLICY = {
    "Normal": {"severity": "None", "actions": ["Allow connection", "No alert"]},
    "Probe": {"severity": "Low",
              "actions": ["Log the source address", "Rate-limit source",
                          "Raise low-priority alert"]},
    "DoS": {"severity": "High",
            "actions": ["Block source address at the firewall",
                        "Drop half-open connections",
                        "Raise high-priority alert"]},
    "R2L": {"severity": "High",
            "actions": ["Temporarily lock the targeted account",
                        "Force password reset", "Block source address",
                        "Raise high-priority alert"]},
    "U2R": {"severity": "Critical",
            "actions": ["Terminate the user session",
                        "Isolate the host from the network",
                        "Snapshot logs for forensics",
                        "Page the administrator"]},
}
# Below this confidence a non-normal verdict is only flagged for review
MIN_CONFIDENCE_FOR_AUTO_ACTION = 0.60


def decide_response(label: str, confidence: float) -> dict:
    """Map a predicted class (and confidence) to a response plan.

    NOTE: the actions are *simulated* - the project returns the plan and
    writes it to a log; it does not touch a real firewall.
    """
    policy = RESPONSE_POLICY[label]
    mode = "automatic"
    actions = list(policy["actions"])
    if label != "Normal" and confidence < MIN_CONFIDENCE_FOR_AUTO_ACTION:
        mode = "manual review"
        actions = ["Flag event for analyst review (low model confidence)"]
    return {"severity": policy["severity"], "mode": mode, "actions": actions}


# --------------------------------------------------------------------------
# Feature engineering (used inside the sklearn Pipeline via FunctionTransformer)
# --------------------------------------------------------------------------
ENGINEERED = ["log_src_bytes", "log_dst_bytes", "bytes_ratio",
              "privilege_score", "error_rate_mean"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add 5 engineered columns. Pure function of each row -> no leakage."""
    df = df.copy()
    df["log_src_bytes"] = np.log1p(df["src_bytes"].astype(float))
    df["log_dst_bytes"] = np.log1p(df["dst_bytes"].astype(float))
    df["bytes_ratio"] = (df["src_bytes"].astype(float) + 1) / (df["dst_bytes"].astype(float) + 1)
    df["bytes_ratio"] = np.log1p(df["bytes_ratio"])
    df["privilege_score"] = (df["root_shell"] + df["su_attempted"] + df["num_root"]
                             + df["num_shells"] + df["num_access_files"]
                             + df["num_file_creations"])
    df["error_rate_mean"] = df[["serror_rate", "srv_serror_rate", "rerror_rate",
                                "srv_rerror_rate"]].mean(axis=1)
    return df


def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["target"] = df["label"].map(ATTACK_TO_CLASS)
    return df


def predict_events(model, df: pd.DataFrame) -> pd.DataFrame:
    """Run the saved pipeline on a dataframe of the 41 raw features."""
    X = df[FEATURES]
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    out = []
    for p in proba:
        i = int(np.argmax(p))
        label, conf = classes[i], float(p[i])
        resp = decide_response(label, conf)
        out.append({"prediction": label, "confidence": round(conf, 4),
                    "probabilities": {c: round(float(v), 4) for c, v in zip(classes, p)},
                    **resp})
    return pd.DataFrame(out)
