import { RootState } from "../storeTypings";

export const getTubsById = (state: RootState, tubId: string) =>
  state.tubs.tubs[tubId];

export const getIsLoadingTubs = (state: RootState) => state.tubs.isLoading;
