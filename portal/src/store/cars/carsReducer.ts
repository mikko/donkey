import { CarsState } from "./carsTypings";
import { createReducer, Reducer, createAction } from "redux-starter-kit";
import * as actions from "./carsActions";
// import { getType, ActionType } from "typesafe-actions";
import { from } from "fromfrom";
import { setTubs } from "../tubs/tubsActions";
import { getType } from "typesafe-actions";

export const initialState: CarsState = {
  cars: {},
  carTubs: {},
  isLoading: false
} as const;

// export type CarsAction = <typeof actions>;

export const carsReducer = createReducer<CarsState, any>(initialState, {
  [actions.setCars.toString()]: (
    state,
    action: ReturnType<typeof actions.setCars>
  ) => {
    state.cars = from(action.payload.cars).toObject(car => car.id);
  },

  [actions.setIsLoadingCars.toString()]: (
    state,
    action: ReturnType<typeof actions.setIsLoadingCars>
  ) => {
    state.isLoading = action.payload.isLoading;
  },

  [getType(setTubs)]: (state, action: ReturnType<typeof setTubs>) => {
    state.carTubs[action.payload.carId] = action.payload.tubs.map(
      tub => tub.id
    );
  }
});
