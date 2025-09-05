param(
    [Parameter(Mandatory=$true)]
    [string]$ImagePath,
    [string]$CheckpointPath = "checkpoints/best_epoch_62.pt",
    [string]$ConfigPath = "configs/optimized.yaml"
)

$env:PYTHONPATH = "$PSScriptRoot\.."

Write-Host "🧠 Brain Stroke Detection Prediction" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Green

# Check if image exists
if (-not (Test-Path $ImagePath)) {
    Write-Host "❌ Error: Image file not found: $ImagePath" -ForegroundColor Red
    exit 1
}

# Check if checkpoint exists
if (-not (Test-Path $CheckpointPath)) {
    Write-Host "❌ Error: Checkpoint file not found: $CheckpointPath" -ForegroundColor Red
    Write-Host "Please train the model first or specify a valid checkpoint path." -ForegroundColor Yellow
    exit 1
}

Write-Host "Running prediction..." -ForegroundColor Yellow
python -m src.predict $ImagePath --checkpoint $CheckpointPath --config $ConfigPath --use_pretrained
