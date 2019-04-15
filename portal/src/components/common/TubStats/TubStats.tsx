import styles from "./TubStats.module.css";
import * as React from "react";
import Statistic from "antd/lib/statistic";
import { Row, Col } from "antd/lib/grid";
import { formatTimestamp } from "../../../utils/date";

export interface TubStatsProps {
  timestamp: Date;
  numDataPoints: number;
}

export const TubStats: React.FunctionComponent<TubStatsProps> = ({
  numDataPoints,
  timestamp
}) => (
  <Row gutter={16} className={styles.tubStats}>
    <Col span={12}>
      <Statistic title="Records" value={numDataPoints} />
    </Col>
    <Col span={12}>
      <Statistic
        title="Recorded"
        value={formatTimestamp(timestamp)}
        precision={2}
      />
    </Col>
  </Row>
);
