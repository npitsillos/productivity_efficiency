#!/bin/sh

echo "[+] Checking for git"
if command -v git>/dev/null 2>&1 ; then
    echo "[+] Git Found"
else
    echo "[+] Git not found"
    echo "[+] Install git and run sript again"
fi

# Download pyenv in home/.pyenv
echo "[+] Downloading pyenv..."
git clone https://github.com/pyenv/pyenv.git ~/.pyenv

# Define PYENV_ROOT and add to PATH
echo "[+] Defining PYENV_ROOT and adding to PATH"
# Add new line
echo \\n >> ~/.bashrc
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc

# Add pyenv init to shell to enable shims
echo "[+] Adding pyenv init to shell"
echo 'if command -v pyenv 1>/dev/null 2>&1; then'\\n  'eval "$(pyenv init -)"'\\n'fi' >> ~/.bashrc

# Download pyenv-virtualenv
echo "[+] Downloading pyenv-virtualenv..."
git clone https://github.com/pyenv/pyenv-virtualenv.git $(pyenv root)/plugins/pyenv-virtualenv

# Add pyenv-virtualenv init to bashrc
echo "[+] Adding pyenv virtualenv-init to .bashrc"
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc

echo "[+] Installing some dependencies..."

sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

echo "[+] Finishing up..."
exec "$SHELL"

echo "[+] Installing final dependencies..."
sudo apt-get install -y make build-essential libssl-dev zlib1g-dev libbz2-dev \
libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev \
xz-utils tk-dev libffi-dev liblzma-dev python-openssl git

echo "[+] Done!"
