# IDPS

A host-based IDS that uses a Graph Neural Network (GNN) to detect attacks.

## Team
Karla Gonzalez Caballero
Christopher Umbel

## System Requirements

This application requires Linux to operate and can be installed on most RedHat or Debian-based distributions.


# Installing

## Debian

```
apt install ./idps_1.0.3.deb
```

## RedHat

```
dnf install ./idps_1.0.3.deb
```

# Running

## Service Management

On all systemd-based distributions you can then ensure the system is running in the backgroud with

```
systemctl start idps
```

and ensure that it's configured to start on boot with

```
systemctl start idps
```

## Running the dashboard (UI)

```
./run_dashboard.sh
```

# Training

```
python3 train.py --csv-path /mnt/data/capstone/GeneratedLabelledFlows/TrafficLabelling/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX_cleaned_short.csv --model-path /mnt/data/models/idps_portscan.keras
```