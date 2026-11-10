#!/data/data/com.termux/files/usr/bin/bash
# Script cài đặt môi trường trên Termux
pkg update -y
pkg install python git nodejs-lts clang cmake make -y
pip install --upgrade pip
pip install -r requirements.txt
echo "Termux setup complete."
