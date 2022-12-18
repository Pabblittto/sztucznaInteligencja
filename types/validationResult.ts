/**
 * Defines how validation looks like. First index of the table defines which real
 * group the record was assigned to and the second index defines which group was predicted by model
 *
 */
export type EntireDatasetValidationResult = Array<Array<number>>;
