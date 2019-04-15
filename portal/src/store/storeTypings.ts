import { CarsState } from "./cars/carsTypings";
import { TubsState } from "./tubs/tubsTypings";
import { ThunkDispatch } from "redux-thunk";
import { AnyAction } from "redux";
import { UIState } from "./ui/uiTypings";
import { TubDataPointsState as TubDataState } from "./tubData/tubDataTypings";

export type ID = string;

export interface RootState {
  cars: CarsState;
  tubs: TubsState;
  tubData: TubDataState;
  ui: UIState;
}

export type StoreDispatch<E = any> = ThunkDispatch<RootState, E, AnyAction>;
