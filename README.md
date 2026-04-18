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

## Project Structure

```text
├── build-deb.sh          # Script to build a Debian package (.deb).
├── build-rpm.sh          # Script to build a RedHat package (.rpm).
├── cic_to_flowmeter.py   # Translates CIC-IDS-2017 dataset columns to pyflowmeter format.
├── config.py             # Defines configuration parameters and hyperparameters for the model.
├── dashboard.py          # Streamlit web dashboard for live traffic detection and batch tests.
├── db.py                 # Handles SQLite database operations for storing flows and inferences.
├── flows_to_tensors.py   # Converts network flow data into graph tensors for the GNN.
├── idps-1.0.3.rpm        # Pre-built RPM package for RedHat-based distributions.
├── idps.service          # systemd service configuration for running the IDS in the background.
├── main.py               # Main entry point for the real-time intrusion detection system.
├── Makefile              # Build file for generating packages and other automated tasks.
├── model.py              # Defines the Graph Convolutional Network (GCN) architecture and logic.
├── models/               # Directory containing the pre-trained models and scalers.
│   ├── idps.weights.h5                  # Pre-trained GCN model weights.
│   └── idps.weights.h5.scaler.joblib    # Pre-fitted StandardScaler for feature normalization.
├── README.md             # This project documentation file.
├── requirements.txt      # Python dependencies required to run the project.
├── run_dashboard.sh      # Shell script to start the Streamlit UI dashboard.
├── run.sh                # Shell script to start the main detection system.
├── sniff.py              # Captures network traffic and extracts flow features using pyflowmeter.
├── test.py               # Evaluates a trained GCN model on a test dataset.
├── train.py              # Script to train the GCN model on a labeled dataset.
├── train.sh              # Shell script wrapper for the training pipeline.
└── VERSION               # The current version number of the IDPS software.
```

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
