import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const experimentRoot = path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1");
const resultsDir = path.join(experimentRoot, "results");
const manifestDir = path.join(experimentRoot, "manifests");
const analysisDir = path.join(experimentRoot, "analysis");
const outputPath = path.join(resultsDir, "kraken_zero_shot_metrics.xlsx");

const summaryCsv = await fs.readFile(path.join(resultsDir, "kraken_zero_shot_summary.csv"), "utf8");
const predictionsCsv = await fs.readFile(path.join(resultsDir, "kraken_zero_shot_predictions.csv"), "utf8");
const manifestCsv = await fs.readFile(path.join(manifestDir, "malaysian_eval_manifest.csv"), "utf8");
const metadata = JSON.parse(await fs.readFile(path.join(resultsDir, "kraken_zero_shot_metadata.json"), "utf8"));

const workbook = Workbook.create();
await workbook.fromCSV(summaryCsv, { sheetName: "Summary" });
await workbook.fromCSV(predictionsCsv, { sheetName: "Predictions" });
await workbook.fromCSV(manifestCsv, { sheetName: "Malaysian Manifest" });
await workbook.fromCSV(await fs.readFile(path.join(analysisDir, "group_summary.csv"), "utf8"), { sheetName: "Group Summary" });
await workbook.fromCSV(await fs.readFile(path.join(analysisDir, "malaysian_text_id_summary.csv"), "utf8"), { sheetName: "Text ID Summary" });
await workbook.fromCSV(await fs.readFile(path.join(analysisDir, "char_substitutions.csv"), "utf8"), { sheetName: "Char Substitutions" });
await workbook.fromCSV(await fs.readFile(path.join(analysisDir, "error_type_summary.csv"), "utf8"), { sheetName: "Error Types" });
const metaSheet = workbook.worksheets.add("Run Metadata");

const summary = workbook.worksheets.getItem("Summary");
const predictions = workbook.worksheets.getItem("Predictions");
const manifest = workbook.worksheets.getItem("Malaysian Manifest");
const groupSummary = workbook.worksheets.getItem("Group Summary");
const textIdSummary = workbook.worksheets.getItem("Text ID Summary");
const charSubs = workbook.worksheets.getItem("Char Substitutions");
const errorTypes = workbook.worksheets.getItem("Error Types");

summary.getRange("A1:M1").format = {
  fill: "#1F4E79",
  font: { bold: true, color: "#FFFFFF" },
};
summary.freezePanes.freezeRows(1);
summary.getRange("A:M").format.columnWidthPx = 150;
summary.getRange("G:M").format.columnWidthPx = 178;
summary.getRange("A1:M1").format.wrapText = true;
summary.getRange("B:M").format.numberFormat = "0.000";
summary.getRange("B:B").format.numberFormat = "0";
summary.getRange("I:I").format.numberFormat = "0.0%";
summary.getRange("J:L").format.numberFormat = "0.000";
summary.getRange("M:M").format.numberFormat = "0.0";

summary.getRange("O1:Q3").values = [
  ["dataset", "CER mean", "WER mean"],
  ["iam", null, null],
  ["malaysian", null, null],
];
summary.getRange("P2:Q3").formulas = [
  ["=C2", "=E2"],
  ["=C3", "=E3"],
];
summary.getRange("O1:Q1").format = {
  fill: "#D9EAF7",
  font: { bold: true },
};
const chart = summary.charts.add("bar", summary.getRange("O1:Q3"));
chart.title = "Zero-Shot Error Rates";
chart.hasLegend = true;
chart.yAxis = { numberFormatCode: "0.0%" };
chart.setPosition("S1", "AA16");

predictions.freezePanes.freezeRows(1);
predictions.getRange("A1:AJ1").format = {
  fill: "#305496",
  font: { bold: true, color: "#FFFFFF" },
};
predictions.getRange("A:AJ").format.columnWidthPx = 118;
predictions.getRange("F:F").format.columnWidthPx = 320;
predictions.getRange("N:O").format.columnWidthPx = 260;
predictions.getRange("R:AC").format.numberFormat = "0.000";
predictions.getRange("N:O").format.wrapText = true;

manifest.freezePanes.freezeRows(1);
manifest.getRange("A1:K1").format = {
  fill: "#548235",
  font: { bold: true, color: "#FFFFFF" },
};
manifest.getRange("A:K").format.columnWidthPx = 140;
manifest.getRange("E:E").format.columnWidthPx = 360;
manifest.getRange("E:E").format.wrapText = true;

for (const sheet of [groupSummary, textIdSummary, charSubs, errorTypes]) {
  const used = sheet.getUsedRange();
  used.format.columnWidthPx = 150;
  sheet.freezePanes.freezeRows(1);
}
groupSummary.getRange("A1:J1").format = { fill: "#1F4E79", font: { bold: true, color: "#FFFFFF" } };
textIdSummary.getRange("A1:G1").format = { fill: "#548235", font: { bold: true, color: "#FFFFFF" } };
charSubs.getRange("A1:C1").format = { fill: "#C65911", font: { bold: true, color: "#FFFFFF" } };
errorTypes.getRange("A1:E1").format = { fill: "#7030A0", font: { bold: true, color: "#FFFFFF" } };
charSubs.getRange("A:C").format.columnWidthPx = 120;

const metadataRows = [
  ["field", "value"],
  ["model_doi", metadata.model_doi],
  ["model_filename", metadata.model_filename],
  ["model_path", metadata.config.model_path],
  ["device", metadata.config.device],
  ["auto_invert", String(metadata.config.auto_invert)],
  ["malaysian_label_mode", metadata.config.malaysian_label_mode],
  ["malaysian_limit", metadata.config.malaysian_limit === null ? "all" : metadata.config.malaysian_limit],
  ["iam_limit", metadata.config.iam_limit],
  ["model_load_seconds", metadata.model_load_seconds],
  ["model_disk_size_mb", metadata.model_disk_size_mb],
  ["rss_model_delta_mb", metadata.rss_model_delta_mb],
  ["torch_version", metadata.torch_version],
  ["python", metadata.python],
];
metaSheet.getRangeByIndexes(0, 0, metadataRows.length, 2).values = metadataRows;
metaSheet.getRange("A1:B1").format = {
  fill: "#7030A0",
  font: { bold: true, color: "#FFFFFF" },
};
metaSheet.getRange("A:B").format.columnWidthPx = 220;
metaSheet.getRange("B:B").format.columnWidthPx = 760;
metaSheet.getRange("B:B").format.wrapText = true;

const diagnostics = workbook.worksheets.add("Notes");
diagnostics.getRange("A1:D8").values = [
  ["item", "detail", "impact", "handling"],
  ["Repository", "mittagessen/kraken + McCATMuS recognition model", "Best fit among reviewed repos for line HTR, Unicode, model repository, and later fine-tuning.", "Used DOI 10.5281/zenodo.13788177."],
  ["Malaysian paths", "CSV image_path values are relative to Dataset/.", "Portable across machines as long as Dataset folder layout is preserved.", "Resolved in code with DATASET_ROOT / image_path."],
  ["Malaysian labels", "Evaluation uses each CSV row as one sample keyed by image_path + text_id.", "This follows the dataset convention supplied by the user.", "The older one-label-per-image inference mode remains only as a diagnostic script option."],
  ["Preprocessing", "Dark-background lines are inverted before OCR.", "Improves zero-shot recognition for this dataset.", "Controlled by --no-auto-invert."],
  ["IAM", "First 200 rows of Teklia/IAM-line test split.", "CPU runtime kept practical.", "Increase --iam-limit or set it empty in the notebook/script for a larger run."],
  ["Timing", "First prediction includes warm-up overhead.", "Mean inference time is slightly inflated.", "Median time is a better steady-state proxy."],
  ["Polish characters", "McCATMuS alphabet includes combining ogonek and accents but not every precomposed Polish glyph.", "Zero-shot output can represent some diacritics after normalization, but fine-tuning should use --resize union/new.", "Notebook notes this in the repo comparison."],
];
diagnostics.getRange("A1:D1").format = {
  fill: "#C65911",
  font: { bold: true, color: "#FFFFFF" },
};
diagnostics.getRange("A:D").format.columnWidthPx = 220;
diagnostics.getRange("B:D").format.columnWidthPx = 360;
diagnostics.getRange("B:D").format.wrapText = true;
diagnostics.freezePanes.freezeRows(1);

const visuals = workbook.worksheets.add("Visuals");
visuals.showGridLines = false;
visuals.getRange("A1:H1").merge();
visuals.getRange("A1").values = [["CRNN/Kraken Zero-Shot Visual Analysis"]];
visuals.getRange("A1").format = {
  fill: "#1F4E79",
  font: { bold: true, color: "#FFFFFF", size: 16 },
};

async function addImage(sheet, title, fileName, row, col, widthPx, heightPx) {
  const bytes = await fs.readFile(path.join(analysisDir, fileName));
  const dataUrl = `data:image/png;base64,${bytes.toString("base64")}`;
  sheet.getCell(row, col).values = [[title]];
  sheet.getCell(row, col).format = { font: { bold: true, color: "#1F2937" } };
  sheet.images.add({
    dataUrl,
    anchor: {
      from: { row: row + 1, col },
      extent: { widthPx, heightPx },
    },
  });
}

await addImage(visuals, "Dataset/group CER and WER", "cer_wer_by_dataset_group.png", 2, 0, 620, 310);
await addImage(visuals, "Malaysian LPD vs PD distribution", "malaysian_error_distribution_lpd_pd.png", 2, 8, 620, 310);
await addImage(visuals, "Malaysian by text_id", "malaysian_error_by_text_id.png", 22, 0, 680, 300);
await addImage(visuals, "Character substitution heatmap", "char_substitution_heatmap_top24.png", 22, 9, 620, 470);
await addImage(visuals, "Top substitutions", "top_char_substitutions.png", 50, 0, 560, 380);
await addImage(visuals, "Error operation mix", "error_operation_mix.png", 50, 8, 620, 340);
await addImage(visuals, "Latency vs width", "latency_vs_image_width.png", 72, 0, 560, 330);

for (const [sheetName, range] of [
  ["Summary", "A1:AA16"],
  ["Predictions", "A1:O25"],
  ["Malaysian Manifest", "A1:K25"],
  ["Run Metadata", "A1:B13"],
  ["Notes", "A1:D8"],
  ["Group Summary", "A1:J10"],
  ["Visuals", "A1:Q40"],
]) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const bytes = new Uint8Array(await preview.arrayBuffer());
  await fs.writeFile(path.join(resultsDir, `${sheetName.replaceAll(" ", "_").toLowerCase()}_preview.png`), bytes);
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
  summary: "formula error scan",
});
console.log(errors.ndjson);

const summaryInspect = await workbook.inspect({
  kind: "table",
  range: "Summary!A1:M3",
  include: "values,formulas",
  tableMaxRows: 5,
  tableMaxCols: 13,
});
console.log(summaryInspect.ndjson);

const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(outputPath);
