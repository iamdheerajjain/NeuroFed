param(
    [string]$ConfigPath = "configs/optimized.yaml"
)

$env:PYTHONPATH = "$PSScriptRoot\.."
python -m src.train.improved_centralized_train --config $ConfigPath
