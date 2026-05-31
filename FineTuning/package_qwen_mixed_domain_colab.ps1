param(
    [string]$OutputZip = "C:\Magisterka\FineTuning\qwen_mixed_domain_colab.zip"
)

$ErrorActionPreference = "Stop"
$fineTuningRoot = [System.IO.Path]::GetFullPath("C:\Magisterka\FineTuning")
$projectRoot = [System.IO.Path]::GetFullPath("C:\Magisterka")
$mixedRoot = Join-Path $fineTuningRoot "qwen_mixed_domain"
$stagingRoot = Join-Path $fineTuningRoot ".qwen_mixed_staging"
$bundleRoot = Join-Path $stagingRoot "FineTuning"
$zip = [System.IO.Path]::GetFullPath($OutputZip)

if (-not (Test-Path -LiteralPath $mixedRoot)) {
    & "C:\Magisterka\.venv_ocr_gpu\Scripts\python.exe" (Join-Path $fineTuningRoot "prepare_qwen_mixed_domain.py")
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
Copy-Item -LiteralPath $mixedRoot -Destination (Join-Path $bundleRoot "qwen_mixed_domain") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "README_QWEN_MIXED_DOMAIN.md") -Destination (Join-Path $bundleRoot "README_QWEN_MIXED_DOMAIN.md") -Force
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "prepare_qwen_iam_eval.py") -Destination (Join-Path $bundleRoot "prepare_qwen_iam_eval.py") -Force
Copy-Item -LiteralPath (Join-Path $fineTuningRoot "README_QWEN_IAM_FINAL_CHECK.md") -Destination (Join-Path $bundleRoot "README_QWEN_IAM_FINAL_CHECK.md") -Force

Compress-Archive -Path (Join-Path $stagingRoot "*") -DestinationPath $zip -CompressionLevel Optimal
Remove-Item -LiteralPath $stagingRoot -Recurse -Force
Write-Output $zip
