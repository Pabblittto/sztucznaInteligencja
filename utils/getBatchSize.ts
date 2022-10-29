/**
 * Returns batch size, assigns dataset size or returns overwrite number
 * @param datasetSize Size of dataset
 * @param overwriteNumber Batch number that overwrites dataset size
 */
export const getBatchSize = (datasetSize: number, overwriteNumber?: number) => {
  return overwriteNumber ?? datasetSize;
};
