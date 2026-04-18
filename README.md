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
├── data/                 # Directory for datasets and local database storage.
│   └── test_data.csv     # Sample labeled dataset for testing and evaluation.
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

# Running as a service

When running as a service, the agent auto-detects a LAN interface, continuously sniffs live network traffic, and extracts flow features using pyflowmeter. Flows are buffered and classified in batches by the pre-trained GCN, and both the raw flow features and the resulting inferences (predicted label and confidence) are persisted to the SQLite database under the configured data path. The service runs as `root` out of `/opt/idps`, logs to the systemd journal, and restarts automatically on failure. Stored results can then be reviewed at any time through the dashboard.

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

# Installing Dependencies

Training and command-line evaluation are run directly from the source tree and require the Python dependencies to be installed. From the project root:

```bash
pip install -r requirements.txt
```

Using a virtual environment or anaconda is recommended to avoid conflicts with system-wide packages. Using anaconda is an exercise left to the reader, but to use a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

# Training

The `train.py` script can be used to train a custom model based on a labeled training CSV as specified by the `--csv-path` argument. The input dataset should follow the CIC-IDS-2017 column structure.

The `.keras` model output (which also automatically persists the feature scaler as a `.joblib` file) will be saved according to the `--model-path` argument.

```bash
python3 train.py --csv-path labelled_training_data.csv --model-path ./models/idps_model.keras
```

---

# Evaluation

## Evaluating from the command line

The `test.py` script allows for the offline evaluation of a trained Graph Neural Network model against labeled datasets. This is essential for verifying model performance on hold-out test data or newly captured and labeled flows.

The script performs the following:
* **Feature Normalization:** Automatically loads the corresponding `.joblib` scaler to ensure features are scaled identically to the training phase.
* **Graph Construction:** Reconstructs the communication graph from the provided CSV.
* **Metric Reporting:** Outputs a comprehensive suite of metrics including overall Accuracy, a detailed Confusion Matrix, and a Classification Report (Precision, Recall, and F1-score for each traffic class).

```bash
python3 test.py --model-path ./models/idps.weights.h5 --csv-path ./data/test_data.csv
```

## Evaluating in the UI

Evaluation can also be performed interactively through the Streamlit dashboard without needing to invoke `test.py` directly.

1. Launch the dashboard with `./run_dashboard.sh` and open `http://localhost:8501` in a browser.
2. From the **Home** screen, click **Import Test Data** to enter Batch Mode.
3. In the **Dataset File** card, drop or select a CIC-IDS-2017 formatted CSV containing labeled flows.
4. Click **▶ Run Model** to classify every flow in the uploaded file against the trained model.
5. When processing completes, the results screen displays the full evaluation output, including overall Accuracy, the Confusion Matrix, and the per-class Classification Report (Precision, Recall, F1-score).

---

# Open Source Dependencies

This project is built on top of the following open-source packages:

* **[TensorFlow](https://www.tensorflow.org/)** — Deep learning framework used to implement and train the Graph Convolutional Network (GCN) model.
* **[scikit-learn](https://scikit-learn.org/)** — Provides the `StandardScaler` for feature normalization and evaluation utilities such as confusion matrices and classification reports.
* **[pandas](https://pandas.pydata.org/)** — Tabular data handling for loading CSV datasets, buffering live flows, and shaping inputs for inference.
* **[NumPy](https://numpy.org/)** — Underlying numerical array library used across feature processing and graph tensor construction.
* **[SciPy](https://scipy.org/)** — Supporting scientific computing routines used in graph and matrix operations.
* **[joblib](https://joblib.readthedocs.io/)** — Serialization of the fitted feature scaler alongside trained model weights.
* **[pyflowmeter](https://pypi.org/project/pyflowmeter/)** — Extracts CIC-IDS-style flow features from live or captured network traffic.
* **[scapy](https://scapy.net/)** — Low-level packet capture and parsing used by pyflowmeter under the hood.
* **[netifaces](https://pypi.org/project/netifaces/)** — Enumerates network interfaces so the agent can auto-detect the LAN interface to sniff.
* **[watchdog](https://python-watchdog.readthedocs.io/)** — Monitors the pyflowmeter CSV output for newly written flows and triggers batched inference.
* **[Streamlit](https://streamlit.io/)** — Powers the interactive web dashboard for batch evaluation and live detection.
* **[Matplotlib](https://matplotlib.org/)** — Generates plots such as ROC curves and confusion matrix visualizations.
* **[JupyterLab](https://jupyter.org/)** — Notebook environment used during model development and experimentation.
* **[python-dotenv](https://pypi.org/project/python-dotenv/)** — Loads configuration values from `.env` files at startup.
* **SQLite** (via Python's standard library `sqlite3`) — Local database backing the persisted flows and inference records shown in the dashboard.