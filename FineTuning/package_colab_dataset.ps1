param(
    [string]$DataRoot = "C:\Magisterka\FineTuning\data",
    [string]$OutputZip = "C:\Magisterka\FineTuning\polish_forms_finetuning_data.zip",
    [switch]$IncludeKraken
)

$ErrorActionPreference = "Stop"
$root = [System.IO.Path]::GetFullPath($DataRoot)
$zip = [System.IO.Path]::GetFullPath($OutputZip)
if (-not (Test-Path -LiteralPath $root)) {
    throw "Data root does not exist: $root"
}
if (-not $zip.StartsWith([System.IO.Path]::GetFullPath("C:\Magisterka\FineTuning"), [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside FineTuning: $zip"
}
if (Test-Path -LiteralPath $zip) {
    Remove-Item -LiteralPath $zip -Force
}
$paths = @(
    (Join-Path $root "images"),
    (Join-Path $root "manifests"),
    (Join-Path $root "qwen"),
    (Join-Path $root "trocr"),
    (Join-Path $root "dataset_summary.json")
)
if ($IncludeKraken) {
    $paths += (Join-Path $root "kraken")
}
Compress-Archive -Path $paths -DestinationPath $zip -CompressionLevel Optimal
Write-Output $zip
