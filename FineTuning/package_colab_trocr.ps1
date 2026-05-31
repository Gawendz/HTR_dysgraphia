param(
    [string]$OutputZip = "C:\Magisterka\FineTuning\polish_forms_trocr_colab.zip"
)

$ErrorActionPreference = "Stop"
$fineTuningRoot = [System.IO.Path]::GetFullPath("C:\Magisterka\FineTuning")
$projectRoot = [System.IO.Path]::GetFullPath("C:\Magisterka")
$stagingRoot = Join-Path $fineTuningRoot ".trocr_colab_staging"
$bundleRoot = Join-Path $stagingRoot "FineTuning"
$zip = [System.IO.Path]::GetFullPath($OutputZip)

if (-not $zip.StartsWith($fineTuningRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside FineTuning: $zip"
}
if (Test-Path -LiteralPath $stagingRoot) {
    Remove-Item -LiteralPath $stagingRoot -Recurse -Force
}
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}

New-Item -ItemType Directory -Force -Path $bundleRoot | Out-Null
Copy-Item -LiteralPath (Join-Path $projectRoot "ocr_benchmark_utils.py") -Destination (Join-Path $stagingRoot "ocr_benchmark_utils.py") -Force
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "README.md") -Destination (Join-Path $bundleRoot "README.md") -Force
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "requirements_colab.txt") -Destination (Join-Path $bundleRoot "requirements_colab.txt") -Force
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "TrOCR") -Destination (Join-Path $bundleRoot "TrOCR") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "configs") -Destination (Join-Path $bundleRoot "configs") -Recurse -Force

$dataTarget = Join-Path $bundleRoot "data"
New-Item -ItemType Directory -Force -Path $dataTarget | Out-Null
foreach ($dir in @("images", "manifests", "trocr")) {
    Copy-Item -LiteralPath (Join-Path $fineTuningRoot "data\$dir") -Destination (Join-Path $dataTarget $dir) -Recurse -Force
}
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "data\dataset_summary.json") -Destination (Join-Path $dataTarget "dataset_summary.json") -Force

Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $zip -CompressionLevel Optimal
Remove-Item -LiteralPath $stagingRoot -Recurse -Force
Write-Output $zip
