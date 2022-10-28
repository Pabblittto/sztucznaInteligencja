import {
  monthMap,
  dayMap,
  FFMCthresholds,
  DMCthresholds,
  DCthresholds,
} from "./values";
import * as tf from "@tensorflow/tfjs";
// File that contains helper methods for discretization

import { Month, Day } from "../types/baseTypes";
/**
 * WorkAround
 */
const tensorToNumber = (tensor: tf.Tensor) => {
  return Number(tensor.arraySync().toString());
};

export const discMonth = async (month: Month) => {
  return monthMap(month);
};

export const discDay = async (day: Day) => {
  return dayMap(day);
};

export const discFFMC = async (FFMC: number) => {
  const FFMCValue = tf.tensor1d([FFMC]);
  const res = tf.upperBound(FFMCthresholds, FFMCValue);

  return tensorToNumber(res);
};

export const discDMC = async (DMC: number) => {
  const DMCValue = tf.tensor1d([DMC]);
  const res = tf.upperBound(DMCthresholds, DMCValue);

  return tensorToNumber(res);
};

export const discDC = async (DC: number) => {
  const DCValue = tf.tensor1d([DC]);
  const res = tf.upperBound(DCthresholds, DCValue);

  return tensorToNumber(res);
};
export const discISI = async (ISI: number) => {};
export const discTemp = async (temp: number) => {};
export const discRH = async (RH: number) => {};
export const discWind = async (wind: number) => {};
export const discRain = async (rain: number) => {};
export const discArea = async (area: number) => {};
