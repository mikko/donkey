import { UIState } from "./uiTypings";
import { createReducer } from "redux-starter-kit";
import {
  setSelectedCar,
  setSelectedTub,
  setSelectedDataPoint
} from "./uiActions";
import { getType } from "typesafe-actions";

export const initialState: UIState = {
  selectedCarId: undefined,
  selectedTubId: undefined,
  selectedDataPointId: undefined
};

export const uiReducer = createReducer<UIState, any>(initialState, {
  [getType(setSelectedCar)]: (
    state,
    action: ReturnType<typeof setSelectedCar>
  ) => {
    state.selectedCarId = action.payload.carId;
    state.selectedTubId = undefined;
  },

  [getType(setSelectedTub)]: (
    state,
    action: ReturnType<typeof setSelectedTub>
  ) => {
    state.selectedTubId = action.payload.tubId;
    state.selectedDataPointId = undefined;
  },

  [getType(setSelectedDataPoint)]: (
    state,
    action: ReturnType<typeof setSelectedDataPoint>
  ) => {
    state.selectedDataPointId = action.payload.dataPointId;
  }
});
