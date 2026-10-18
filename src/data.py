"""Immutable source identity, explicit schema and label isolation."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

TARGET = "attack_cat"
NOMINAL = ("proto", "service", "state")
BINARY = ("is_ftp_login", "is_sm_ips_ports")
DROP = ("id", "tcprtt", "stcpb", "dtcpb")
COLUMNS = tuple(
    "id dur proto service state spkts dpkts sbytes dbytes rate sttl dttl sload dload "
    "sloss dloss sinpkt dinpkt sjit djit swin stcpb dtcpb dwin tcprtt synack ackdat "
    "smean dmean trans_depth response_body_len ct_srv_src ct_state_ttl ct_dst_ltm "
    "ct_src_dport_ltm ct_dst_sport_ltm ct_dst_src_ltm is_ftp_login ct_ftp_cmd "
    "ct_flw_http_mthd ct_src_ltm ct_srv_dst is_sm_ips_ports attack_cat".split()
)
NUMERIC = tuple(c for c in COLUMNS if c not in (*NOMINAL, TARGET))
SOM_COLUMNS = tuple(c for c in COLUMNS if c not in (*DROP, TARGET))
CLASSES = (
    "Normal",
    "Reconnaissance",
    "Backdoor",
    "DoS",
    "Exploits",
    "Analysis",
    "Fuzzers",
    "Worms",
    "Shellcode",
    "Generic",
)
DATA_HASH = "acde2c646b55c9c6907ade03b1ec985bf58f53926c6045ae221a61ab17535c3b"
METADATA_HASH = "da20edf0c5daf352961ba320d176df9c04849ff4aec3a38cc65c47f4b0b7ace8"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Required input missing: {path}")
    if sha256(path) != expected:
        raise ValueError(f"Input identity mismatch: {path}")


def validate_schema(frame: pd.DataFrame, *, full: bool = True) -> None:
    if tuple(frame.columns) != COLUMNS:
        raise ValueError("Source columns/order differ from the frozen schema")
    if full and len(frame) != 257673:
        raise ValueError("Source row count differs from 257673")
    if frame.empty or not frame.id.is_unique or frame.id.isna().any():
        raise ValueError("Source id must be present and unique")
    for column in NUMERIC:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            raise ValueError(f"Non-numeric source dtype: {column}")
        values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid/non-finite numeric source: {column}")
    target = frame[TARGET]
    if target.isna().any() or not target.isin(range(10)).all():
        raise ValueError("Targets must be integral codes 0 through 9")
    if full and set(target) != set(range(10)):
        raise ValueError("Missing target class")
    if not frame.is_ftp_login.isin([0, 1, 2, 4]).all():
        raise ValueError("Unexpected FTP binary schema violation")
    if not frame.is_sm_ips_ports.isin([0, 1]).all():
        raise ValueError("Invalid binary is_sm_ips_ports")
    for column in NOMINAL:
        if not frame[column].dropna().map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"Invalid nominal source dtype: {column}")
        if column != "service" and frame[column].isna().any():
            raise ValueError(f"Unexpected missing nominal value: {column}")


def load_source(data_path: Path, metadata_path: Path) -> pd.DataFrame:
    verify_file(data_path, DATA_HASH)
    verify_file(metadata_path, METADATA_HASH)
    metadata = pd.read_csv(metadata_path)
    if "attack_cat" not in set(metadata.Name):
        raise ValueError("Target missing from supplied metadata")
    frame = pd.read_csv(data_path, na_values=["?"], keep_default_na=False)
    validate_schema(frame)
    return frame


def descriptive(frame: pd.DataFrame, *, reduced: bool = False) -> pd.DataFrame:
    """The sole labelled-frame boundary. targets are discarded explicitly."""
    columns = SOM_COLUMNS if reduced else tuple(c for c in COLUMNS if c not in ("id", TARGET))
    features = frame.loc[:, columns].copy()
    for column in NOMINAL:
        features[column] = features[column].astype("string").fillna("<Missing>")
    if reduced:
        features["is_ftp_login"] = features.is_ftp_login.gt(0).astype("int8")
    return features


def group_ids(features: pd.DataFrame) -> np.ndarray:
    """Stable hashes, with explicit collision checking against exact equality."""
    if TARGET in features or "id" in features:
        raise ValueError("Grouping must exclude target and id")
    hashes = pd.util.hash_pandas_object(features, index=False).to_numpy()
    unique = features.drop_duplicates()
    unique_hashes = pd.util.hash_pandas_object(unique, index=False)
    if unique_hashes.duplicated().any():
        raise ValueError("Feature hash collision; use an exact grouping implementation")
    return hashes


def assert_som_features(features: pd.DataFrame) -> None:
    if TARGET in features:
        raise ValueError("Target labels are prohibited in SOM inputs")
    if tuple(features.columns) != SOM_COLUMNS:
        raise ValueError("Unexpected SOM input schema; identifiers/extra columns prohibited")
