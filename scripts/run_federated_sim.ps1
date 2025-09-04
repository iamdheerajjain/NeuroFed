param(
	[string]$ConfigPath = "configs/default.yaml"
)

$env:PYTHONPATH = "$PSScriptRoot\.."
python -m src.federated.simulate --config $ConfigPath
