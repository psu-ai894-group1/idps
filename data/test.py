import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'idps.db')

conn = sqlite3.connect(DB_PATH)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

def port_to_int(v):
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        if len(b) == 0:
            return None
        return int.from_bytes(b, byteorder="little", signed=False)
    else:
        return v

cursor.execute("""
    SELECT f.id, f.captured_at, f.src_ip, f.src_port, f.dst_ip, f.dst_port,
           i.predicted_label, i.confidence
    FROM flows f
        INNER JOIN inferences i ON i.flow_id = f.id
    ORDER BY f.captured_at DESC
    LIMIT 20
""")

rows = cursor.fetchall()
if not rows:
    print("No entries found.")
else:
    print(f"{'ID':>6}  {'Captured At':>27}  {'Source':>21}  {'Destination':>21}  {'Label':<15}  {'Confidence':>10}")
    print("-" * 110)
    for r in rows:
        src_port = port_to_int(r["src_port"])
        dst_port = port_to_int(r["dst_port"])

        src = f"{r['src_ip']}:{src_port}" if src_port is not None else f"{r['src_ip']}:NA"
        dst = f"{r['dst_ip']}:{dst_port}" if dst_port is not None else f"{r['dst_ip']}:NA"

        print(
            f"{r['id']:>6}  {r['captured_at']:>27}  {src:>21}  {dst:>21}  {r['predicted_label']:<15}  {r['confidence']:>10.4f}"
        )

conn.close()
