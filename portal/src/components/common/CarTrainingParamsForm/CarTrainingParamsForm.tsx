import styles from "./CarTrainingParamsForm.module.css";
import * as React from "react";
import Form from "antd/lib/form";
import Input from "antd/lib/input";
import Checkbox from "antd/lib/checkbox";
import Button from "antd/lib/button";
import { Row, Col } from "antd/lib/grid";
import { WrappedFormUtils } from "antd/lib/form/Form";
import { Tub } from "../../../store/tubs/tubsTypings";

export interface CarTrainingFormData {
  modelName: string;
  selectedTubIds: string[];
  selectedAugmentations: string[];
}

export interface CarTrainingParamsFormProps {
  form: WrappedFormUtils;
  tubs: Tub[];
  onFormSubmitted?: (formData: CarTrainingFormData) => void;
}

const formItemLayout = {
  labelCol: {
    xs: { span: 24 },
    sm: { span: 8 }
  },
  wrapperCol: {
    xs: { span: 24 },
    sm: { span: 16 }
  }
};

const tailFormItemLayout = {
  wrapperCol: {
    xs: {
      span: 24,
      offset: 0
    },
    sm: {
      span: 16,
      offset: 8
    }
  }
};

const CarTrainingParamsForm: React.FunctionComponent<
  CarTrainingParamsFormProps
> = props => {
  const [modelName, setModelName] = React.useState("");
  const { getFieldDecorator } = props.form;

  const onFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    props.form.validateFieldsAndScroll((err, values) => {
      if (!err) {
        console.log("Received values of form: ", values);
      }
    });
  };

  return (
    <Form {...formItemLayout} onSubmit={onFormSubmit}>
      <Form.Item label="Model name">
        {getFieldDecorator("modelName", {
          rules: [
            {
              required: true,
              message: "Please give model a name"
            }
          ]
        })(<Input />)}
      </Form.Item>

      <Form.Item label="Tubs">
        {getFieldDecorator("tubs", {
          rules: [
            {
              required: true,
              message: "Please select at least one tub"
            }
          ]
        })(
          <Checkbox.Group style={{ width: "100%" }}>
            <Row>
              {props.tubs.map(tub => (
                <Checkbox
                  key={tub.id}
                  className={styles.carTrainingParamsFormCheckboxGroupItem}
                  value={tub.id}
                >
                  {tub.name}
                </Checkbox>
              ))}
            </Row>
          </Checkbox.Group>
        )}
      </Form.Item>

      <Form.Item label="Augmentations">
        {getFieldDecorator("augmentations")(
          <Checkbox.Group style={{ width: "100%" }}>
            <Row>
              {/* <Col span={6}> */}
              <Checkbox
                className={styles.carTrainingParamsFormCheckboxGroupItem}
                value="flip"
              >
                Flip
              </Checkbox>
              {/* </Col>
              <Col span={6}> */}
              <Checkbox
                className={styles.carTrainingParamsFormCheckboxGroupItem}
                value="brightness"
              >
                Brightness
              </Checkbox>
              {/* </Col>
              <Col span={6}> */}
              <Checkbox
                className={styles.carTrainingParamsFormCheckboxGroupItem}
                value="shadow"
              >
                Shadow
              </Checkbox>
              {/* </Col>
              <Col span={6}> */}
              <Checkbox
                className={styles.carTrainingParamsFormCheckboxGroupItem}
                value="boogieman"
              >
                Boogieman
              </Checkbox>
              {/* </Col> */}
            </Row>
          </Checkbox.Group>
          //   style={{ width: "100%" }}
          //   options={[
          //     { label: "Flip", value: "flip" },
          //     { label: "Brightness", value: "brightness" },
          //     { label: "Shadow", value: "shadow" },
          //     { label: "Boogieman", value: "boogieman" }
          //   ]}
          // />
        )}
      </Form.Item>

      <Form.Item {...tailFormItemLayout}>
        {/*
        // @ts-ignore */}
        <Button type="primary" htmlType="submit">
          Train
        </Button>
      </Form.Item>
    </Form>
  );
};

const WrapperCarTrainingParamsForm = Form.create({ name: "train_car" })(
  CarTrainingParamsForm
);

export { WrapperCarTrainingParamsForm as CarTrainingParamsForm };
