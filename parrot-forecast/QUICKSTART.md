# Quick Start Guide

## The Data Loading Issue

Modern browsers block loading local JSON files for security reasons (CORS policy).

There are **two solutions**:

## Solution 1: Standalone Version (Easiest)

```bash
python build_standalone_dashboard.py
open forecast_dashboard_standalone.html
```

This creates a self-contained HTML file with embedded data. No server needed!

## Solution 2: Local Server

```bash
# Start a local server
python -m http.server 8000

# Then open in your browser:
# http://localhost:8000/forecast_dashboard.html
```

Server is running on port 8000. Press Ctrl+C to stop it.

## Full Workflow

```bash
# 1. Generate forecast data
python generate_forecast_data.py

# 2. Build standalone dashboard
python build_standalone_dashboard.py

# 3. Open in browser
open forecast_dashboard_standalone.html
```

That's it! The dashboard should now display perfectly with all forecast data.

## Current Status

✓ HTTP server is running on http://localhost:8000
✓ Standalone version created
✓ Both options are available to you
