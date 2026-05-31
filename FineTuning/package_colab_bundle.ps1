param(
    [string]$OutputZip = "C:\Magisterka\FineTuning\polish_forms_finetuning_colab_bundle.zip"
)

$ErrorActionPreference = "Stop"
$fineTuningRoot = [System.IO.Path]::GetFullPath("C:\Magisterka\FineTuning")
$projectRoot = [System.IO.Path]::GetFullPath("C:\Magisterka")
$stagingRoot = Join-Path $fineTuningRoot ".colab_bundle_staging"
$bundleRoot = Join-Path $stagingRoot "FineTuning"
$zip = [System.IO.Path]::GetFullPath($OutputZip)

if (-not $zip.StartsWith($fineTuningRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside FineTuning: $zip"
}
if (Test-Path -LiteralPath $stagingRoot) {
    $fullStaging = [System.IO.Path]::GetFullPath($stagingRoot)
    if (-not $fullStaging.StartsWith($fineTuningRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove outside FineTuning: $fullStaging"
    }
    Remove-Item -LiteralPath $fullStaging -Recurse -Force
}
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}

New-Item -ItemType Directory -Force -Path $bundleRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot "ocr_benchmark_utils.py") -Destination (Join-Path $stagingRoot "ocr_benchmark_utils.py") -Force

foreach ($item in @("README.md", "requirements_colab.txt")) {
    Copy-Item -LiteralPath (Join-Path $fineTuningRoot $item) -Destination (Join-Path $bundleRoot $item) -Force
}
foreach ($dir in @("Qwen3VL_LoRA", "TrOCR", "configs")) {
    Copy-Item -LiteralPath (Join-Path $fineTuningRoot $dir) -Destination (Join-Path $bundleRoot $dir) -Recurse -Force
}

$dataTarget = Join-Path $bundleRoot "data"
New-Item -ItemType Directory -Force -Path $dataTarget | Out-Null
foreach ($dir in @("images", "manifests", "qwen", "trocr")) {
    Copy-Item -LiteralPath (Join-Path $fineTuningRoot "data\$dir") -Destination (Join-Path $dataTarget $dir) -Recurse -Force
}
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "data\dataset_summary.json") -Destination (Join-Path $dataTarget "dataset_summary.json") -Force

Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $zip -CompressionLevel Optimal
Remove-Item -LiteralPath $stagingRoot -Recurse -Force
Write-Output $zip
