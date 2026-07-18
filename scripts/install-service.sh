#!/usr/bin/env bash
# Install exl3-serve as a systemd service.
#
#   ./scripts/install-service.sh            # user service (default) — systemctl --user
#   ./scripts/install-service.sh --system   # system service — boots without login, needs sudo
#
# User service is the default (no root needed). To start it at boot without an active login
# session, enable lingering once:  sudo loginctl enable-linger $USER
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
MODE="${1:---user}"
TRITON_CACHE="${TRITON_CACHE_DIR:-$HOME/.triton/cache}"

[ -x .venv/bin/python ] || { echo "Run ./install.sh first (.venv missing)."; exit 1; }
[ -f tabbyAPI/main.py ] || { echo "TabbyAPI not found — run ./install.sh first."; exit 1; }
mkdir -p "$TRITON_CACHE"

render() { sed -e "s|@ROOT@|$ROOT|g" -e "s|@TRITON_CACHE@|$TRITON_CACHE|g" systemd/exl3-serve.service.template; }

case "$MODE" in
  --user)
    DEST="$HOME/.config/systemd/user"
    mkdir -p "$DEST"
    render > "$DEST/exl3-serve.service"
    systemctl --user daemon-reload
    systemctl --user enable --now exl3-serve.service
    echo
    echo "Installed as a USER service."
    echo "  status:  systemctl --user status exl3-serve"
    echo "  logs:    journalctl --user -u exl3-serve -f"
    echo "  stop:    systemctl --user stop exl3-serve"
    echo
    echo "Start at boot without an active login session (run once):"
    echo "  sudo loginctl enable-linger $(id -un)"
    ;;
  --system)
    # Add a User= line and target multi-user for a system-wide service.
    render \
      | sed "/^\[Service\]/a User=$(id -un)" \
      | sed 's/^WantedBy=default.target/WantedBy=multi-user.target/' \
      | sudo tee /etc/systemd/system/exl3-serve.service >/dev/null
    sudo systemctl daemon-reload
    sudo systemctl enable --now exl3-serve.service
    echo
    echo "Installed as a SYSTEM service (starts at boot)."
    echo "  status:  systemctl status exl3-serve"
    echo "  logs:    journalctl -u exl3-serve -f"
    ;;
  *)
    echo "usage: $0 [--user|--system]"; exit 1 ;;
esac
