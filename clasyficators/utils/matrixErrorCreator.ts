import { EntireDatasetValidationResult } from "./../../types/validationResult";

export class ErrorMatrixCreator {
  result: EntireDatasetValidationResult;
  constructor(numbOfClasses: number) {
    this.result = new Array(numbOfClasses);

    for (let i = 0; i < numbOfClasses; i++) {
      this.result[i] = new Array(numbOfClasses).fill(0);
    }
  }

  public increaseaCell(
    expectedClassIndex: number,
    predictedClassIndex: number
  ) {
    this.result[expectedClassIndex][predictedClassIndex] += 1;
  }
}
