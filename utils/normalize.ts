import { linearNormalizeFn } from "./../normalization/linear/index";
import { expertNormalizeFn } from "./../normalization/expert/index";
import {
  discMonth,
  discDay,
  discFFMC,
  discDMC,
  discDC,
  discISI,
  discRH,
  discWind,
  discRain,
  discTemp,
  discArea,
} from "../normalization/expert/methods";
import { DiscretizatedRowType } from "../types/baseTypes";
import * as tf from "@tensorflow/tfjs";
import { BaseRowType } from "../types/baseTypes";

export enum NormalizationType {
  EXPERT = "EXPERT",
  LINEAR = "LINEAR",
}

/**
 * Returns particular type of normalizing function based of passed type. Used for discretizing data.
 * @param type
 * @returns
 */
export const getNormalizingFunction = (type: NormalizationType) => {
  switch (type) {
    case NormalizationType.EXPERT:
      return expertNormalizeFn;
    case NormalizationType.LINEAR:
      return linearNormalizeFn;
    default:
      throw "not supported normalizint func";
  }
};

/**
 * @deprecated do not use this
 */
export const discretization = async (data: tf.data.CSVDataset) => {
  const discreditedData = data.mapAsync(async (row) => {
    const typedRow: BaseRowType = row as BaseRowType;

    const result: DiscretizatedRowType = {
      X: typedRow.X,
      Y: typedRow.Y,
      month: await discMonth(typedRow.month),
      day: await discDay(typedRow.day),
      FFMC: await discFFMC(typedRow.FFMC),
      DMC: await discDMC(typedRow.DMC),
      DC: await discDC(typedRow.DC),
      ISI: await discISI(typedRow.ISI),
      temp: await discTemp(typedRow.temp),
      RH: await discRH(typedRow.RH),
      wind: await discWind(typedRow.wind),
      rain: await discRain(typedRow.rain),
      area: await discArea(typedRow.area),
    };

    return result;
  });

  return discreditedData;
};
