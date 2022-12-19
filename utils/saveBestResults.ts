import * as fs from "fs";

export const saveResults = (optimal: any, trialObject: any) => {
  const filePath = process.cwd() + `/output/` + "bestResult.txt";

  // Clear file
  fs.writeFileSync(filePath, "");

  fs.appendFileSync(filePath, "Arguments for best results:\n");
  fs.appendFileSync(filePath, `${JSON.stringify(optimal)}\n`);
  fs.appendFileSync(filePath, `================================\n`);
  fs.appendFileSync(filePath, `\n`);
  fs.appendFileSync(filePath, JSON.stringify(trialObject));
};
