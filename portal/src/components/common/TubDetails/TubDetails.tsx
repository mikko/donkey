import * as React from "react";
import styles from "./TubDetails.module.css";
import Typography from "antd/lib/typography";
import { TubStats } from "../TubStats/TubStats";
import { TubPlayer } from "../TubPlayer/TubPlayer";
import { Tub, TubDataPoint } from "../../../api";

const { Title } = Typography;

export interface TubDetailsProps {
  tub: Tub;
  dataPoint: TubDataPoint;
  // data: TubDataPoint[];
}

export const TubDetails: React.FunctionComponent<TubDetailsProps> = ({
  // data,
  tub
}) => (
  <div className={styles.tubDetails}>
    {/* <Title>{tub.name}</Title>
    <TubStats numDataPoints={tub.numDataPoints} timestamp={tub.timestamp} />
    <TubPlayer data={data} /> */}
  </div>
);
