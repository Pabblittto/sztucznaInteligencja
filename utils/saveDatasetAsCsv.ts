import * as fs from "fs";
import * as tf from "@tensorflow/tfjs";
import { ClearedRowData } from "../types/baseTypes";

/**
 * Saves dataset to csv file in "dane directory"
 * @param filename File name
 */
export const saveDatasetAsCsv = async (
  destinationFileName: string,
  data: tf.data.Dataset<ClearedRowData>
) => {
  const fileName = process.cwd() + `/output/${destinationFileName}.csv`;

  // Clear file
  fs.writeFileSync(fileName, "");

  const headerDefinition = "X,Y,day,temp,RH,wind,rain,area\n";

  // write header line
  fs.appendFileSync(fileName, headerDefinition);

  const dataArray = await data.toArray();

  // wite rows:
  dataArray.forEach((e) => {
    fs.appendFileSync(
      fileName,
      `${e.X},${e.Y},${e.day},${e.temp},${e.RH},${e.wind},${e.rain},${e.area}\n`
    );
  });
};
