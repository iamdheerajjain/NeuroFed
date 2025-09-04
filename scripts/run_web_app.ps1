param(
    [int]$Port = 8501
)

$env:PYTHONPATH = "$PSScriptRoot\.."

Write-Host "🧠 Starting Brain Stroke Detection Web Interface..." -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Write-Host "Web interface will be available at: http://localhost:$Port" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Cyan

# Install streamlit if not already installed
pip install streamlit

# Run the web app
streamlit run src/web_app.py --server.port $Port
