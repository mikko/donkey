import { TubDataPointsState } from "./tubDataTypings";
import { createReducer } from "redux-starter-kit";
import * as actions from "./tubDataActions";
import { ActionType, getType } from "typesafe-actions";

export const initialState: TubDataPointsState = {
  tubDataPoints: {},
  isLoadingForTubs: []
} as const;

export type TubsAction = ActionType<typeof actions>;

export const tubDataReducer = createReducer<TubDataPointsState, any>(
  initialState,
  {
    [getType(actions.setTubDataPoint)]: (
      state,
      action: ReturnType<typeof actions.setTubDataPoint>
    ) => {
      if (!state.tubDataPoints[action.payload.tubId]) {
        state.tubDataPoints[action.payload.tubId] = [];
      }

      state.tubDataPoints[action.payload.tubId][action.payload.dataPointId] =
        action.payload.tubDataPoint;
    },

    [getType(actions.setTubDataPoints)]: (
      state,
      action: ReturnType<typeof actions.setTubDataPoints>
    ) => {
      state.tubDataPoints[action.payload.tubId] = action.payload.tubDataPoints;
    },

    [getType(actions.setLoadingTubData)]: (
      state,
      action: ReturnType<typeof actions.setLoadingTubData>
    ) => {
      if (state.isLoadingForTubs.includes(action.payload.tubId)) {
        throw new Error(
          "Invariant violation. Already loading for " + action.payload.tubId
        );
      }

      state.isLoadingForTubs.push(action.payload.tubId);
    },

    [getType(actions.setFinishLoadingTubData)]: (
      state,
      action: ReturnType<typeof actions.setFinishLoadingTubData>
    ) => {
      state.isLoadingForTubs = state.isLoadingForTubs.filter(
        x => x === action.payload.tubId
      );
    }
  }
);
