import { ExtendedTubDataApi } from "./tubDataApi";
import {
  Car as _Car,
  Tub as _Tub,
  TubDataPoint as _TubDataPoint,
  CarApi,
  TubApi,
  TrainingApi
} from "./apiClient";

export const carApi = new CarApi();

export const tubApi = new TubApi();

export const tubDataApi = new ExtendedTubDataApi();

export const trainingApi = new TrainingApi();

// Because "isolatedModules" = true config in tsconfig we can't
// directly re-export these
export type Car = _Car;
export type Tub = _Tub;
export type TubDataPoint = _TubDataPoint;
