import * as tf from "@tensorflow/tfjs";
import { DiscretizatedRowType } from "../../types/baseTypes";

export class DuplicateDealer {
  hashedDataList: string[] = [];

  /**
   * Removed duplicates from data
   * @param data Data
   */
  dealWithDuplicates = async (data: tf.data.Dataset<DiscretizatedRowType>) => {
    const result: DiscretizatedRowType[] = [];
    const nonProcessedData = await data.toArray();

    nonProcessedData.forEach((row) => {
      const { area, ...restRow } = row;
      const hash = Object.values(restRow).join(""); // Creates string with values from

      if (!this.hashedDataList.includes(hash)) {
        // if an array doesn't have this hash - add this hash to list and add this row to the result
        this.hashedDataList.push(hash);
        result.push(row);
      }
      // if our array contains hash, ignore it and do not add it to the result
    });

    return tf.data.array(result);
  };
}
