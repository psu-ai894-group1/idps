"""
AI-894
This file handles database operations for storing captured flows and model inferences. It manages the SQLite connection and schema creation.
Authors: Karla Gonzalez Caballero (kxg5613@psu.edu), Christopher Umbel (czu5008@psu.edu)
"""
import math
import os
import sqlite3
import logging
from datetime import datetime, timezone

_DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
_db_dir = _DEFAULT_DIR


def set_data_path(path):
    """Override the directory where the database is stored."""
    global _db_dir
    _db_dir = path

def get_db_path():
    return os.path.join(_db_dir, 'idps.db')

# Get a connection to the database
def get_connection():
    os.makedirs(_db_dir, exist_ok=True)
    conn = sqlite3.connect(get_db_path())
    conn.execute("PRAGMA journal_mode=WAL")
    _ensure_tables(conn)
    return conn

# Ensure the tables exist
def _ensure_tables(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS flows (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            captured_at TEXT NOT NULL,
            src_ip TEXT,
            src_port INTEGER,
            dst_ip TEXT,
            dst_port INTEGER,
            protocol INTEGER,
            flow_duration REAL,
            tot_fwd_pkts INTEGER,
            totlen_fwd_pkts REAL,
            flow_byts_s REAL,
            fwd_pkts_s REAL,
            bwd_pkts_s REAL,
            down_up_ratio REAL,
            fwd_pkt_len_mean REAL,
            fwd_pkt_len_max REAL,
            fwd_pkt_len_std REAL,
            bwd_pkt_len_mean REAL,
            bwd_pkt_len_max REAL,
            pkt_len_mean REAL,
            pkt_len_std REAL,
            fwd_iat_mean REAL,
            fwd_iat_std REAL,
            bwd_iat_mean REAL,
            bwd_iat_std REAL,
            flow_iat_mean REAL,
            flow_iat_std REAL,
            fin_flag_cnt INTEGER,
            syn_flag_cnt INTEGER,
            rst_flag_cnt INTEGER,
            psh_flag_cnt INTEGER,
            ack_flag_cnt INTEGER,
            init_fwd_win_byts REAL,
            init_bwd_win_byts REAL,
            fwd_header_len REAL,
            fwd_act_data_pkts INTEGER,
            active_mean REAL,
            active_std REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS inferences (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            flow_id INTEGER NOT NULL,
            inferred_at TEXT NOT NULL,
            predicted_label TEXT NOT NULL,
            confidence REAL NOT NULL,
            FOREIGN KEY (flow_id) REFERENCES flows(id)
        )
    """)
    conn.commit()


# Columns to pull from the flows DataFrame into the flows table.
_FLOW_COLUMNS = [
    'src_ip', 'src_port', 'dst_ip', 'dst_port', 'protocol',
    'flow_duration', 'tot_fwd_pkts', 'totlen_fwd_pkts', 'flow_byts_s',
    'fwd_pkts_s', 'bwd_pkts_s', 'down_up_ratio',
    'fwd_pkt_len_mean', 'fwd_pkt_len_max', 'fwd_pkt_len_std',
    'bwd_pkt_len_mean', 'bwd_pkt_len_max', 'pkt_len_mean', 'pkt_len_std',
    'fwd_iat_mean', 'fwd_iat_std', 'bwd_iat_mean', 'bwd_iat_std',
    'flow_iat_mean', 'flow_iat_std',
    'fin_flag_cnt', 'syn_flag_cnt', 'rst_flag_cnt', 'psh_flag_cnt', 'ack_flag_cnt',
    'init_fwd_win_byts', 'init_bwd_win_byts', 'fwd_header_len',
    'fwd_act_data_pkts', 'active_mean', 'active_std',
]


# Convert numpy/pandas types to native Python types for sqlite3
def _to_python(val):
    """Convert numpy/pandas types to native Python types for sqlite3."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return None
    if hasattr(val, 'item'):
        return val.item()
    return val

# Save flows and inferences to the database
def save_flows_and_inferences(flows_df, preds, confidences, class_names):
    """Insert captured flows and their inference results into the database."""
    conn = get_connection()
    try:
        now = datetime.now(timezone.utc).isoformat()
        cursor = conn.cursor()

        # Determine which columns are actually present in the DataFrame
        available = [c for c in _FLOW_COLUMNS if c in flows_df.columns]
        placeholders = ', '.join(['?'] * (len(available) + 1))  # +1 for captured_at
        col_names = ', '.join(['captured_at'] + available)
        insert_flow_sql = f"INSERT INTO flows ({col_names}) VALUES ({placeholders})"

        insert_inf_sql = "INSERT INTO inferences (flow_id, inferred_at, predicted_label, confidence) VALUES (?, ?, ?, ?)"

        for i in range(len(flows_df)):
            row_values = [now] + [_to_python(flows_df.iloc[i].get(c)) for c in available]
            cursor.execute(insert_flow_sql, row_values)
            flow_id = cursor.lastrowid

            label = class_names[preds[i]]
            conf = float(confidences[i])
            if math.isnan(conf) or math.isinf(conf):
                conf = 0.0
            cursor.execute(insert_inf_sql, (flow_id, now, label, conf))

        conn.commit()
        logging.info("Saved %d flows and inferences to database", len(flows_df))
    finally:
        conn.close()
