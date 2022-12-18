import { fromAreaIndexToLabel } from "../normalization/expert/values";
import { EntireDatasetValidationResult } from "../types/validationResult";
import * as fs from "fs";

export const createErrorMatrixFile = (
  filename: string,
  results: EntireDatasetValidationResult
) => {
  const filePath = process.cwd() + `/output/` + filename;

  // Clear file
  fs.writeFileSync(filePath, "");
  fs.appendFileSync(
    filePath,
    `<html><head>
  <style>
table {
  font-family: arial, sans-serif;
  border-collapse: collapse;
}

td, th {
  border: 1px solid #dddddd;
  text-align: left;
  padding: 8px;
}
</style>
  </head><body>\n`
  );
  fs.appendFileSync(filePath, "<table>\n");

  fs.appendFileSync(filePath, "<tr>\n");
  fs.appendFileSync(filePath, "<th></th>\n"); //spacer

  // top labels
  fs.appendFileSync(filePath, `<p>Predicted classes are on the top</>\n`);
  for (let i = 0; i < 5; i++) {
    const label = fromAreaIndexToLabel(i);
    fs.appendFileSync(filePath, `<th>${label}</th>\n`);
  }
  fs.appendFileSync(filePath, "</tr>\n");

  // Rows with data
  for (let index = 0; index < results.length; index++) {
    fs.appendFileSync(filePath, "<tr>\n");
    fs.appendFileSync(filePath, `<th>${fromAreaIndexToLabel(index)}</th>\n`);
    const row = results[index];
    row.forEach((cell) => {
      fs.appendFileSync(filePath, `<th>${cell}</th>\n`);
    });

    fs.appendFileSync(filePath, "</tr>\n");
  }
  results.forEach((row) => {});
  fs.appendFileSync(filePath, "<table>\n");

  fs.appendFileSync(filePath, "</table>\n");
  fs.appendFileSync(filePath, "</body></html>\n");
};
