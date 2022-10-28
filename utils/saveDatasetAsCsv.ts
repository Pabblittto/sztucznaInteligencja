import * as fs from "fs";
import * as tf from "@tensorflow/tfjs";
import { DiscretizatedRowType } from "../types/baseTypes";

/**
 * Saves dataset to csv file in "dane directory"
 * @param filename File name
 */
export const saveDatasetAsCsv = async (
  destinationFileName: string,
  data: tf.data.Dataset<DiscretizatedRowType>
) => {
  const fileName = process.cwd() + `/dane/${destinationFileName}.csv`;

  // Clear file
  fs.writeFileSync(fileName, "");

  const headerDefinition =
    "X,Y,month,day,FFMC,DMC,DC,ISI,temp,RH,wind,rain,area\n";

  // write header line
  fs.appendFileSync(fileName, headerDefinition);

  const dataArray = await data.toArray();

  // wite rows:
  dataArray.forEach((e) => {
    fs.appendFileSync(
      fileName,
      `${e.X},${e.Y},${e.month},${e.day},${e.FFMC},${e.DMC},${e.DC},${e.ISI},${e.temp},${e.RH},${e.wind},${e.rain},${e.area}\n`
    );
  });
};
