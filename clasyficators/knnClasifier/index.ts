import { defineTheLargestValueInArray } from "./../nauralNetwork/index";
import { datasetDivider } from "./../../utils/datasetDivider";
import { linearNormalizeFn } from "./../../normalization/linear/index";
import { ClearedRowData } from "./../../types/baseTypes";
import knnClasifier from "@tensorflow-models/knn-classifier";
import { data as tfData } from "@tensorflow/tfjs";
import { ErrorMatrixCreator } from "../utils/matrixErrorCreator";

export const divideDataIntoTrainingAndTestingSets = async (
  data: tfData.Dataset<ClearedRowData>,
  trainingPercentage: number,
  numbOfClasses: number
): Promise<{
  training: tfData.Dataset<ClearedRowData>;
  testing: tfData.Dataset<ClearedRowData>;
}> => {
  const sortedData: Array<Array<ClearedRowData>> = new Array(numbOfClasses);

  for (let i = 0; i < sortedData.length; i++) {
    sortedData[i] = [];
  }
  const originalData = await data.toArray();

  originalData.forEach((row) => {
    sortedData[row.area].push(row);
  });

  const training: ClearedRowData[] = [];
  const testing: ClearedRowData[] = [];

  for (const sortedRow of sortedData) {
    const numbOfElements = sortedRow.length;

    const trainingCount = Math.ceil(
      (numbOfElements * trainingPercentage) / 100
    );

    const tmpTraining = sortedRow.splice(0, trainingCount);
    const tmpTesting = sortedRow; // The rest goes as a training

    training.push(...tmpTraining);
    testing.push(...tmpTesting);
  }

  return { training: tfData.array(training), testing: tfData.array(testing) };
};

export const createKnnClasifier = async (
  training: tfData.Dataset<ClearedRowData>,
  numberOfClasses: number
) => {
  const clasifier = knnClasifier.create();

  const trainingData = await Promise.all(
    (
      await training.toArray()
    ).map(async (x) => await datasetDivider(x, numberOfClasses))
  );

  trainingData.forEach((row) => {
    const rowClass = defineTheLargestValueInArray(row.ys.arraySync());

    clasifier.addExample(row.xs, rowClass);
  });

  return clasifier;
};

export const validateKnnClasifier = async (
  clasifier: knnClasifier.KNNClassifier,
  testing: tfData.Dataset<ClearedRowData>,
  numberOfClasses: number,
  k: number
) => {
  const errorMatrix = new ErrorMatrixCreator(5);

  const testingData = await Promise.all(
    (
      await testing.toArray()
    ).map(async (x) => await datasetDivider(x, numberOfClasses))
  );

  testingData.forEach((row) => {
    const expectedClass = defineTheLargestValueInArray(row.ys.arraySync());

    const predicted = clasifier.predictClass(row.xs, k);
    console.log(predicted);

    // errorMatrix.increaseaCell(expectedClass, predicted);
  });
};
