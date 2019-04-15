import { RootState } from "../storeTypings";

export const getSelectedCarId = (state: RootState) => state.ui.selectedCarId;

export const getSelectedTubId = (state: RootState) => state.ui.selectedTubId;

export const getSelectedDataPointId = (state: RootState) =>
  state.ui.selectedDataPointId;
