#!/bin/bash
# Quick launcher for Tuti Parrot Web Interface

cd "$(dirname "$0")"

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║          🦜 Tuti Parrot Web Interface                    ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Starting server..."
echo ""

python serve.py
