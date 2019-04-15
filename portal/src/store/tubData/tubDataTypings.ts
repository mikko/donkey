export interface TubDataPoint {
  timestamp: Date;
  angle: number;
  throttle: number;
  imageName: string;
}

export interface TubDataPointsState {
  tubDataPoints: {
    readonly [tubId: string]: ReadonlyArray<TubDataPoint>;
  };
  /**
   * Ids of tubs for which we are currently loading data
   */
  readonly isLoadingForTubs: ReadonlyArray<string>;
}
