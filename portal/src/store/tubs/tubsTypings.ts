export interface Tub {
  id: string;
  name: string;
  timestamp: Date;
  numDataPoints: number;
}

export interface TubsState {
  tubs: {
    readonly [tubId: string]: Tub;
  };
  readonly isLoading: boolean;
}
