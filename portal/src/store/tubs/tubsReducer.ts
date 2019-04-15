import { TubsState } from "./tubsTypings";
import { createReducer } from "redux-starter-kit";
import * as actions from "./tubsActions";
import { from } from "fromfrom";
import { ActionType, getType } from "typesafe-actions";

export const initialState: TubsState = {
  tubs: {},
  isLoading: false
} as const;

export type TubsAction = ActionType<typeof actions>;

export const tubsReducer = createReducer<TubsState, any>(initialState, {
  [getType(actions.setTubs)]: (
    state,
    action: ReturnType<typeof actions.setTubs>
  ) => {
    state.tubs = from(action.payload.tubs).toObject(tub => tub.id);
  },

  [getType(actions.setIsLoadingTubs)]: (
    state,
    action: ReturnType<typeof actions.setIsLoadingTubs>
  ) => {
    state.isLoading = action.payload.isLoading;
  }
});
