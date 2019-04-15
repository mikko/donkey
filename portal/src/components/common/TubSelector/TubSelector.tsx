import * as React from "react";
import List from "antd/lib/list";
import Menu from "antd/lib/menu";
import { TubPreview } from "../TubPreview/TubPreview";
import { Tub } from "../../../api";

// const SubMenu = Menu.SubMenu;
// const MenuItemGroup = Menu.ItemGroup;

export interface TubSelectorProps {
  tubs: Tub[];
  selectedTub?: Tub;
  onTubSelected: (tub: Tub) => void;
}

export const TubSelector: React.FunctionComponent<TubSelectorProps> = props => (
  // <Menu selectedKeys={props.selectedTub ? [props.selectedTub.id] : []}>
  //   {props.tubs.map(tub => (
  //     <Menu.Item key={tub.id}>
  //       <h2>{tub.name}</h2>
  //       <span>{tub.timestamp}</span>
  //     </Menu.Item>
  //   ))}
  //   {/* <Menu.Item key={props.}>
  //     <Icon type="pie-chart" />
  //     <span>Option 1</span>
  //   </Menu.Item> */}
  // </Menu>
  <List
    itemLayout="vertical"
    size="large"
    dataSource={props.tubs}
    renderItem={item => (
      <List.Item key={item.name} onClick={() => props.onTubSelected(item)}>
        {/* <List.Item.Meta title={item.name} description={item.timestamp} /> */}
        <TubPreview isSelected={item === props.selectedTub} tub={item} />
      </List.Item>
    )}
  />
);
