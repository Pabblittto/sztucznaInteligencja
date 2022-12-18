import * as tf from "@tensorflow/tfjs";
import { ClearedRowData } from "../../types/baseTypes";
import { sample } from "simple-statistics";

export class ReasampleTool {
  static reasample = async (
    data: tf.data.Dataset<ClearedRowData>,
    recordsNumber: number,
    numbOfClasses: number
  ): Promise<tf.data.Dataset<ClearedRowData>> => {
    const sortedData: Array<Array<ClearedRowData>> = new Array(
      numbOfClasses
    ).fill([]);

    const originalData = await data.toArray();

    originalData.forEach((row) => {
      sortedData[row.area].push(row);
    });

    const result: Array<ClearedRowData> = [];
    // Do the reasampling  (will do oversalmpling if there is few elements, and undersampling when there is too many elements)

    sortedData.forEach((sortedRow) => {
      const reassambledData = sample(sortedRow, recordsNumber, Math.random);
      result.push(...reassambledData);
    });

    console.log(console.log(result.length));

    return tf.data.array(result);
  };
}
