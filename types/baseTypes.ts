export enum Month {
  jan = "jan",
  feb = "feb",
  mar = "mar",
  apr = "apr",
  may = "may",
  jun = "jun",
  jul = "jul",
  aug = "aug",
  oct = "oct",
  nov = "nov",
  dec = "dec",
}

export enum Day {
  sun = "sun",
  mon = "mon",
  tue = "tue",
  wed = "wed",
  thu = "thu",
  fri = "fri",
  sat = "sat",
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
