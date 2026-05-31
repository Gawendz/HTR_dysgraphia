import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const experimentRoot = path.dirname(new URL(import.meta.url).pathname).replace(/^\/([A-Za-z]:)/, "$1");
const resultsDir = path.join(experimentRoot, "results");
const manifestDir = path.join(experimentRoot, "manifests");
const analysisDir = path.join(experimentRoot, "analysis");
const outputPath = path.join(resultsDir, "trocr_zero_shot_metrics.xlsx");

const workbook = Workbook.create();
await workbook.fromCSV(await fs.readFile(path.join(resultsDir, "trocr_zero_shot_summary.csv"), "utf8"), { sheetName: "Summary" });
await workbook.fromCSV(await fs.readFile(path.join(resultsDir, "trocr_zero_shot_predictions.csv"), "utf8"), { sheetName: "Predictions" });
await workbook.fromCSV(await fs.readFile(path.join(manifestDir, "trocr_eval_manifest.csv"), "utf8"), { sheetName: "Manifest" });
await workbook.fromCSV(await fs.readFile(path.join(analysisDir, "group_summary.csv"), "utf8"), { sheetName: "Group Summary" });
await workbook.fromCSV(await fs.readFile(path.join(analysisDir, "malaysian_text_id_summary.csv"), "utf8"), { sheetName: "Text ID Summary" });
await workbook.fromCSV(await fs.readFile(path.join(analysisDir, "best_model_char_substitutions.csv"), "utf8"), { sheetName: "Best Char Subs" });

const metadata = JSON.parse(await fs.readFile(path.join(resultsDir, "trocr_zero_shot_metadata.json"), "utf8"));
const metaSheet = workbook.worksheets.add("Run Metadata");
const metaRows = [["model_id", "status", "load_seconds", "rss_model_delta_mb", "polish_probe_exact", "polish_probe_roundtrip", "error"]];
for (const item of metadata.metadata) {
  metaRows.push([
    item.model_id,
    item.status,
    item.load_seconds,
    item.rss_model_delta_mb,
    String(item.polish_probe_exact),
    item.polish_probe_roundtrip,
    item.error,
  ]);
}
metaSheet.getRangeByIndexes(0, 0, metaRows.length, metaRows[0].length).values = metaRows;

for (const sheetName of ["Summary", "Predictions", "Manifest", "Group Summary", "Text ID Summary", "Best Char Subs", "Run Metadata"]) {
  const sheet = workbook.worksheets.getItem(sheetName);
  sheet.freezePanes.freezeRows(1);
  sheet.getUsedRange().format.columnWidthPx = 145;
  sheet.getRange("A1:Z1").format = { fill: "#4C1D95", font: { bold: true, color: "#FFFFFF" } };
}
workbook.worksheets.getItem("Predictions").getRange("O:P").format.columnWidthPx = 280;
workbook.worksheets.getItem("Predictions").getRange("O:P").format.wrapText = true;
metaSheet.getRange("F:F").format.columnWidthPx = 280;

const visuals = workbook.worksheets.add("Visuals");
visuals.showGridLines = false;
visuals.getRange("A1:H1").merge();
visuals.getRange("A1").values = [["TrOCR Zero-Shot Visual Analysis"]];
visuals.getRange("A1").format = { fill: "#4C1D95", font: { bold: true, color: "#FFFFFF", size: 16 } };

async function addImage(title, fileName, row, col, widthPx, heightPx) {
  const bytes = await fs.readFile(path.join(analysisDir, fileName));
  const dataUrl = `data:image/png;base64,${bytes.toString("base64")}`;
  visuals.getCell(row, col).values = [[title]];
  visuals.getCell(row, col).format = { font: { bold: true } };
  visuals.images.add({
    dataUrl,
    anchor: { from: { row: row + 1, col }, extent: { widthPx, heightPx } },
  });
}

await addImage("Model and dataset error", "trocr_model_dataset_error.png", 2, 0, 700, 330);
await addImage("LPD vs PD", "trocr_malaysian_lpd_pd_error.png", 2, 9, 700, 330);
await addImage("Latency by model", "trocr_latency_by_model.png", 24, 0, 600, 300);
await addImage("Best model character substitutions", "trocr_best_model_char_heatmap.png", 24, 8, 650, 500);

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 50 },
});
console.log(errors.ndjson);
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(outputPath);
