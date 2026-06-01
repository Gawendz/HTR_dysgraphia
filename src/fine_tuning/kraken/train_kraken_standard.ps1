param(
    [string]$DataRoot = "C:\Magisterka\FineTuning\data",
    [string]$OutputDir = "C:\Magisterka\FineTuning\Kraken\outputs\standard",
    [string]$BaseModel = "",
    [int]$Epochs = 50,
    [int]$BatchSize = 4
)

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$ketos = "C:\Magisterka\.venv_ocr_gpu\Scripts\ketos.exe"
$python = "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe"
$pageRoot = Join-Path $DataRoot "kraken_pagexml"
$trainManifest = Join-Path $pageRoot "train\manifest.txt"
$valManifest = Join-Path $pageRoot "val\manifest.txt"
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null
if (-not (Test-Path -LiteralPath $trainManifest) -or -not (Test-Path -LiteralPath $valManifest)) {
    & $python "C:\Magisterka\FineTuning\Kraken\prepare_pagexml_data.py" --data-root (Join-Path $DataRoot "kraken") --output-root $pageRoot
}
if ($BaseModel -eq "") {
    $BaseModel = & $python -c "from CRNN.kraken_iam_malaysian_inverted import locate_model; print(locate_model())"
}

$args = @(
    "train",
    "-f", "page",
    "-t", $trainManifest,
    "-e", $valManifest,
    "-o", $OutputDir,
    "-B", "$BatchSize",
    "--epochs", "$Epochs",
    "--quit", "fixed",
    "--normalization", "NFC",
    "--resize", "union",
    "--weights-format", "safetensors",
    "--augment"
)

$args += @("-i", $BaseModel)

& $ketos --workers 0 @args
