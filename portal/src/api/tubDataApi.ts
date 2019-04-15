import { TubDataApi, BASE_PATH, GetTubDataPointsRequest } from "./apiClient";

export class ExtendedTubDataApi extends TubDataApi {
  /**
   *
   * @summary Returns all data points in a tub
   * @param {string} carId ID of car to return
   * @param {string} tubId ID of tub to return
   * @throws {RequiredError}
   * @memberof TubDataApi
   */
  public async getTubDataPoints(requestParameters: GetTubDataPointsRequest) {
    const dataPoints = await super.getTubDataPoints(requestParameters);

    for (const dataPoint of dataPoints) {
      dataPoint.imageName = BASE_PATH + dataPoint.imageName;
    }

    return dataPoints;
  }
}
