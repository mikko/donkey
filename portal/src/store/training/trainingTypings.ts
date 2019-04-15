export interface TrainingParams {}

export interface TrainingStatus {
  readonly status: string;
}

export interface TrainingState {
  readonly isLoading: boolean;
  readonly params: TrainingParams;
  readonly status: TrainingStatus;
}
