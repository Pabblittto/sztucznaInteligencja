import * as tf from "@tensorflow/tfjs";

export enum Month {
  jan = "jan",
  feb = "feb",
  mar = "mar",
  apr = "apr",
  may = "may",
  jun = "jun",
  jul = "jul",
  aug = "aug",
  sep = "sep",
  oct = "oct",
  nov = "nov",
  dec = "dec",
}

export enum Day {
  mon = "mon",
  tue = "tue",
  wed = "wed",
  thu = "thu",
  fri = "fri",
  sat = "sat",
  sun = "sun",
}

/**
 * Type of raw row from the csv file, before discetization
 */
export type BaseRowType = {
  X: number;
  Y: number;
  month: Month;
  day: Day;
  FFMC: number;
  DMC: number;
  DC: number;
  ISI: number;
  temp: number;
  RH: number;
  wind: number;
  rain: number;
  area: number;
};

/**
 * Row of discretizated data
 */
export type DiscretizatedRowType = {
  X: number;
  Y: number;
  month: number;
  day: number;
  FFMC: number;
  DMC: number;
  DC: number;
  ISI: number;
  temp: number;
  RH: number;
  wind: number;
  rain: number;
  area: number;
};

/**
 * Row of data after discretization and correlated rows removal.
 * Rows that were removed:
 * - month - it correlates with a lot of other columns
 * - FFMC
 * - DMC
 * - DC
 * - ISI
 *
 */
export type ClearedRowData = Omit<
  DiscretizatedRowType,
  "month" | "FFMC" | "DMC" | "DC" | "ISI"
>;

export const columnConfigs: {
  [key: string]: tf.data.ColumnConfig;
} = {
  X: { dtype: "int32" },
  Y: { dtype: "int32" },
  month: { dtype: "string" },
  day: { dtype: "string" },
  FFMC: { dtype: "float32" },
  DMC: { dtype: "float32" },
  ISI: { dtype: "float32" },
  temp: { dtype: "float32" },
  RH: { dtype: "int32" },
  wind: { dtype: "float32" },
  rain: { dtype: "float32" },
  area: { dtype: "float32" },
};

export enum Optimizer {
  sgd = "sgd",
  adam = "adam",
}

export enum LossFunction {
  categoricalCrossentropy = "categoricalCrossentropy",
  meanSquaredError = "meanSquaredError",
}
