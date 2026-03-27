Name:           idps
Version:        %{pkg_version}
Release:        1%{?dist}
Summary:        Intrusion Detection and Prevention System
License:        Proprietary
URL:            https://github.com/psu-ai894-group1/idps
BuildArch:      noarch
Requires:       python3 >= 3.11, curl

%description
GNN-based network intrusion detection system that captures
network flows and classifies them in real time.

%install
mkdir -p %{buildroot}/opt/idps/util
mkdir -p %{buildroot}/var/idps/models
mkdir -p %{buildroot}/var/idps/data
mkdir -p %{buildroot}/lib/systemd/system

cp %{_sourcedir}/main.py             %{buildroot}/opt/idps/
cp %{_sourcedir}/model.py            %{buildroot}/opt/idps/
cp %{_sourcedir}/db.py               %{buildroot}/opt/idps/
cp %{_sourcedir}/config.py           %{buildroot}/opt/idps/
cp %{_sourcedir}/flows_to_tensors.py %{buildroot}/opt/idps/
cp %{_sourcedir}/sniff.py            %{buildroot}/opt/idps/
cp %{_sourcedir}/requirements.txt    %{buildroot}/opt/idps/
cp %{_sourcedir}/util/*.py           %{buildroot}/opt/idps/util/
cp %{_sourcedir}/models/idps.weights.h5               %{buildroot}/var/idps/models/
cp %{_sourcedir}/models/idps.weights.h5.scaler.joblib %{buildroot}/var/idps/models/
cp %{_sourcedir}/idps.service        %{buildroot}/lib/systemd/system/

%files
/opt/idps
/var/idps
/lib/systemd/system/idps.service

%post
INSTALL_DIR=/opt/idps
VENV_DIR=$INSTALL_DIR/venv
DATA_DIR=/var/idps/data

mkdir -p "$DATA_DIR"

python3 -m venv --without-pip "$VENV_DIR"
"$VENV_DIR/bin/python3" -c "import ensurepip; ensurepip._main()" 2>/dev/null || true
curl -sS https://bootstrap.pypa.io/get-pip.py | "$VENV_DIR/bin/python3"
"$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

systemctl daemon-reload
systemctl enable idps.service

echo "=========================================="
echo " idps installed successfully."
echo ""
echo " Models:   /var/idps/models/"
echo " Database: /var/idps/data/idps.db"
echo ""
echo " Start the service with:"
echo "   systemctl start idps"
echo "=========================================="

%preun
systemctl stop idps.service || true
systemctl disable idps.service || true

%postun
if [ "$1" -eq 0 ]; then
    rm -rf /opt/idps
    rm -rf /var/idps
fi
systemctl daemon-reload
