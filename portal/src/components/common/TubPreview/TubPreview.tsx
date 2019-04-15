import * as React from "react";
import Card from "antd/lib/card";
import { formatTimestamp } from "../../../utils/date";
import { Tub } from "../../../api";

export interface TubPreviewProps {
  tub: Tub;
  isSelected: boolean;
}

export const TubPreview: React.FunctionComponent<TubPreviewProps> = ({
  tub
}) => {
  return (
    <Card title={tub.name} hoverable>
      {formatTimestamp(tub.timestamp)}
    </Card>
  );
};
