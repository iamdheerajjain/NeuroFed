param(
    [string]$CheckpointPath = "checkpoints/best_epoch_64.pt",
    [string]$ConfigPath = "configs/optimized.yaml"
)

$env:PYTHONPATH = "$PSScriptRoot\.."
python -m src.evaluate_improved --checkpoint $CheckpointPath --config $ConfigPath --use_pretrained
