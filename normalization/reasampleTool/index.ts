import * as tf from "@tensorflow/tfjs";
import { ClearedRowData } from "../../types/baseTypes";
import { sample } from "simple-statistics";

export class ReasampleTool {
  static reasample = async (
    data: tf.data.Dataset<ClearedRowData>,
    sampleSize: number,
    numbOfClasses: number
  ): Promise<tf.data.Dataset<ClearedRowData>> => {
    const sortedData: Array<Array<ClearedRowData>> = new Array(numbOfClasses);

    for (let i = 0; i < sortedData.length; i++) {
      sortedData[i] = [];
    }

    const originalData = await data.toArray();
    originalData.forEach((row) => {
      sortedData[row.area].push(row);
    });

    const result: Array<ClearedRowData> = [];
    // Do the reasampling  (will do oversalmpling if there is few elements, and undersampling when there is too many elements)

    sortedData.forEach((sortedRow) => {
      const tmpAggregateArray = [];

      while (tmpAggregateArray.length != sampleSize) {
        if (sortedRow.length >= sampleSize) {
          //there is much more data in original array than is expected
          tmpAggregateArray.push(...sample(sortedRow, sampleSize, () => 0.5));
        } else {
          const required = sampleSize - tmpAggregateArray.length;
          const amountToTake = Math.min(sortedRow.length, required);
          tmpAggregateArray.push(...sample(sortedRow, amountToTake, () => 0.5));
        }
      }
      result.push(...tmpAggregateArray);
    });

    return tf.data.array(result);
  };
}
