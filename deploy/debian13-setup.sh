#!/usr/bin/env bash
# Prepare a Debian 13 (trixie) VPS for the Docker deployment.
# Run as root from the repository directory:   sudo deploy/debian13-setup.sh
#
# Installs Docker (official repository) and the compose plugin, keeps the clock in sync,
# enables a firewall that only allows SSH, and prepares the data directory.
set -euo pipefail
cd "$(dirname "$0")/.."

[ "$(id -u)" = 0 ] || { echo "run as root (sudo)"; exit 1; }
. /etc/os-release
[ "${VERSION_CODENAME:-}" = "trixie" ] || echo "warning: expected Debian 13 (trixie), found ${PRETTY_NAME:-unknown}"

apt-get update
apt-get install -y ca-certificates curl git ufw systemd-timesyncd

# Accurate time matters: session times, order timestamps and the journal depend on it.
timedatectl set-ntp true

# Docker from Docker's own repository (Debian's docker.io package lags behind).
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/debian ${VERSION_CODENAME} stable" \
  > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
systemctl enable --now docker

# Firewall: SSH only. (Docker publishes nothing here except VNC on 127.0.0.1.)
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw --force enable

# The agent container runs as uid 1000 and writes the journal, reviews and state here.
mkdir -p data
chown -R 1000:1000 data

for f in .env gateway.env; do
  if [ ! -f "$f" ]; then
    cp "$f.example" "$f"
    echo "created $f from the example: fill it in"
  fi
  chmod 600 "$f"
done

echo
echo "Done. Next:"
echo "  1. Fill in .env (API keys, ACCOUNT_EQUITY) and gateway.env (IBKR paper login)"
echo "  2. CODE_VERSION=\$(git rev-parse --short HEAD) docker compose up -d --build"
echo "  3. docker compose logs -f agent"
