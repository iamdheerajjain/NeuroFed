param(
    [string]$ResultsDir = "results"
)

Write-Host "Opening Brain Stroke Detection Model Report..." -ForegroundColor Green
Write-Host "Report location: $PSScriptRoot\..\$ResultsDir\report.html" -ForegroundColor Yellow

# Open the HTML report in default browser
Start-Process "$PSScriptRoot\..\$ResultsDir\report.html"

Write-Host "Report opened in your default web browser!" -ForegroundColor Green
