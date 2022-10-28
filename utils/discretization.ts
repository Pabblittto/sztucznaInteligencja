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
} from "./../discretization/methods";
import { DiscretizatedRowType } from "./../types/baseTypes";
import { FFMCthresholds } from "./../discretization/values";
import * as tf from "@tensorflow/tfjs";
import { BaseRowType } from "../types/baseTypes";

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

    // FFMC discretization:
    const FFMCValue = tf.tensor1d([typedRow.FFMC]);
    tf.upperBound(FFMCthresholds, FFMCValue);
    // tf.lowerBound()

    return typedRow;
  });

  // const seq = tf.tensor1d([0, 76, 84, 8]);
  // const values = tf.tensor([900]);

  // const res = tf.upperBound(FFMCthresholds, values);
  // console.log(Number(res.arraySync().toString()));

  return discreditedData;
};
