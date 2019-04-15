import * as React from "react";
import Menu from "antd/lib/menu";
import Icon from "antd/lib/icon";
import { Link } from "react-router-dom";

export enum CarDataPage {
  Data = "data",
  Train = "train"
}

export interface NavigationProps {
  selectedCarId: string;
  selectedPage: CarDataPage;
  onSubPageSelected?: (subPage: CarDataPage) => void;
}

export const Navigation: React.FunctionComponent<NavigationProps> = ({
  onSubPageSelected,
  selectedCarId,
  selectedPage
}) => (
  <Menu
    theme="dark"
    mode="horizontal"
    selectedKeys={[selectedPage]}
    style={{ lineHeight: "64px" }}
  >
    <Menu.Item
      key="data"
      onClick={() => onSubPageSelected && onSubPageSelected(CarDataPage.Data)}
    >
      <Link to={`/car/${selectedCarId}/data`}>
        <Icon type="mail" />
        Data
      </Link>
    </Menu.Item>
    <Menu.Item
      key="train"
      onClick={() => onSubPageSelected && onSubPageSelected(CarDataPage.Train)}
    >
      <Link to={`/car/${selectedCarId}/train`}>
        <Icon type="appstore" />
        Train
      </Link>
    </Menu.Item>
    <Menu.Item key="analysis">
      <Icon type="appstore" />
      Analysis
    </Menu.Item>
    <Menu.Item key="drive">
      <Icon type="appstore" />
      Drive
    </Menu.Item>
  </Menu>
);
