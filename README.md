# IDPS - Graph-Based Intrusion Detection System

**Course:** AI-894

A host-based Intrusion Detection System (IDS) that leverages a Graph Neural Network (GNN) to detect network attacks. By representing network flows as nodes and communication behaviors as edges, this system applies Graph Convolutional Network (GCN) layers to effectively classify both normal (benign) traffic and malicious intrusions.

## Team
* Karla Gonzalez Caballero (kxg5613@psu.edu)
* Christopher Umbel (czu5008@psu.edu)

## Key Features
* **Graph-Based Classification:** Uses a multi-layer GCN implemented in TensorFlow to identify complex attack patterns based on communication behaviors.
* **Live Traffic Sniffing:** Captures and analyzes network packets in real-time from a specified interface using pyflowmeter.
* **Batch Analysis:** Evaluates pre-captured datasets (compatible with the CIC-IDS-2017 format) providing full evaluation metrics including Accuracy, F1 scores, Confusion Matrices, and ROC curves.
* **Interactive Dashboard:** A rich, Streamlit-powered web UI for launching real-time captures, running batch tests, and reviewing historical inference data stored in a local SQLite database.

## System Requirements

This application requires Linux to operate and can be installed natively on most RedHat or Debian-based distributions. Python 3 is required if running directly from the source code.

---

# Installing

## Debian-based (Ubuntu, Debian)
To install the pre-packaged `.deb` file:
```bash
apt install ./idps_1.0.3.deb
```

## RedHat-based (Fedora, RHEL, CentOS)
To install the pre-packaged `.rpm` file:
```bash
dnf install ./idps_1.0.3.rpm
```

## From Source (Development)
If you are developing or prefer to run the system without a package manager, ensure your dependencies are installed:
```bash
pip install -r requirements.txt
```

---

# Running

## Service Management

On all systemd-based distributions, you can manually start the background IDS service with:

```bash
systemctl start idps
```

To ensure that the IDPS service is configured to start automatically on system boot, run:

```bash
systemctl enable idps
```

## Running the Dashboard (UI)

The system includes an interactive web dashboard built with Streamlit. You can launch it using the provided shell script:

```bash
./run_dashboard.sh
```

Once executed, the UI will start a local web server (typically accessible at `http://localhost:8501`). Through the browser interface, you can navigate between **Import Test Data** (Batch Mode) and **Live Detection** (Sniffing Mode).

---

# Training

The `train.py` script can be used to train a custom model based on a labeled training CSV as specified by the `--csv-path` argument. The input dataset should follow the CIC-IDS-2017 column structure.

The `.keras` model output (which also automatically persists the feature scaler as a `.joblib` file) will be saved according to the `--model-path` argument.

```bash
python3 train.py --csv-path labelled_training_data.csv --model-path ./models/idps_model.keras
```
