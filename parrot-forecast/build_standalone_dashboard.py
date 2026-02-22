"""
Build a standalone HTML dashboard with embedded data
No server required - just open the file in a browser
"""

import json

# Read the forecast data
with open('forecast_data.json', 'r') as f:
    forecast_data = json.load(f)

# Read the HTML template
with open('forecast_dashboard.html', 'r') as f:
    html_content = f.read()

# Embed the data directly in the HTML
standalone_html = html_content.replace(
    "fetch('forecast_data.json')",
    f"Promise.resolve({{ json: () => Promise.resolve({json.dumps(forecast_data)}) }})"
).replace(
    ".then(response => response.json())",
    ".then(response => response.json())"
)

# Write the standalone version
with open('forecast_dashboard_standalone.html', 'w') as f:
    f.write(standalone_html)

print("✓ Created forecast_dashboard_standalone.html")
print("✓ This file contains embedded data and works without a server")
print("✓ Just double-click to open in your browser")
