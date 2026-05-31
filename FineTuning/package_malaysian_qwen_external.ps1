param(
    [string]$OutputZip = "C:\Magisterka\FineTuning\malaysian_qwen_external_validation_colab.zip"
)

$ErrorActionPreference = "Stop"
$fineTuningRoot = [System.IO.Path]::GetFullPath("C:\Magisterka\FineTuning")
$projectRoot = [System.IO.Path]::GetFullPath("C:\Magisterka")
$dataRoot = Join-Path $fineTuningRoot "malaysian_external_qwen"
$stagingRoot = Join-Path $fineTuningRoot ".malaysian_qwen_staging"
$bundleRoot = Join-Path $stagingRoot "FineTuning"
$zip = [System.IO.Path]::GetFullPath($OutputZip)

if (-not (Test-Path -LiteralPath $dataRoot)) {
    & "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" (Join-Path $fineTuningRoot "prepare_malaysian_external_qwen.py")
}
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
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "requirements_colab.txt") -Destination (Join-Path $bundleRoot "requirements_colab.txt") -Force
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "Qwen3VL_LoRA") -Destination (Join-Path $bundleRoot "Qwen3VL_LoRA") -Recurse -Force
Copy-Item -LiteralPath $dataRoot -Destination (Join-Path $bundleRoot "malaysian_external_qwen") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "README_MALAYSIAN_EXTERNAL_QWEN.md") -Destination (Join-Path $bundleRoot "README_MALAYSIAN_EXTERNAL_QWEN.md") -Force

Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $zip -CompressionLevel Optimal
Remove-Item -LiteralPath $stagingRoot -Recurse -Force
Write-Output $zip
