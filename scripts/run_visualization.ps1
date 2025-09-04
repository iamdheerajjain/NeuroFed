param(
    [string]$CheckpointPath = "checkpoints/best_epoch_64.pt",
    [string]$ConfigPath = "configs/optimized.yaml",
    [string]$OutputDir = "results"
)

$env:PYTHONPATH = "$PSScriptRoot\.."
python -m src.visualize_results --checkpoint $CheckpointPath --config $ConfigPath --use_pretrained --output_dir $OutputDir
