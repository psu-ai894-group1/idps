# IDPS

A host-based IDS that uses a Graph Neural Network (GNN) to detect attacks.

## Team
Karla Gonzalez Caballero (kxg5613@psu.edu)
Christopher Umbel (czu5008@psu.edu)

## System Requirements

This application requires Linux to operate and can be installed on most RedHat or Debian-based distributions.

# Installing

## Debian

```
apt install ./idps_1.0.3.deb
```

## RedHat

```
dnf install ./idps_1.0.3.rpm
```

# Running

## Service Management

On all systemd-based distributions you can then ensure the system is running in the backgroud with

```
systemctl start idps
```

and ensure that it's configured to start on boot with

```
systemctl enable idps
```

## Running the dashboard (UI)

```
./run_dashboard.sh
```

# Training

The `train.py` script can be used to train a model based on a labelled training CSV as specified by the `--csv-path` argument. The .keras output (including the feature scaler) will be saved according to the `--model-path` argument.

```
python3 train.py --csv-path labelled_training_data.csv --model-path ./models/idps_model.keras
```
