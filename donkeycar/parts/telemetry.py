import paho.mqtt.client as mqtt
import json

class Telemetry:

  def on_message(self, client, userdata, message):
      print("message received " ,str(message.payload.decode("utf-8")))
      print("message topic=",message.topic)
      print("message qos=",message.qos)
      print("message retain flag=",message.retain)

  def send_telemetry(self, *args):
    assert len(self.inputs) == len(args)
    self.client.publish(self.topic, json.dumps(dict(zip(self.inputs, args))))

  def __init__(self, inputs, broker='broker.hivemq.com', topic='donkey-telemetry-data', subscribe=False):
    self.on = True
    self.topic = topic
    self.inputs = inputs
    self.args = None
    self.client = mqtt.Client()
    self.client.connect(broker)
    self.client.loop_start()
    if subscribe:
      self.client.on_message = self.on_message
      self.client.subscribe(topic)

  def run(self, *args):
    self.send_telemetry(args)

  def run_threaded(self, args):
    self.args = args

  def update(self):
    self.send_telemetry(self.args)

  def shutdown(self):
    # indicate that the thread should be stopped
    self.on = False
    self.client.loop_stop()
    print('stopping telemetry')

if __name__ == "__main__":
  import time
  iter = 0
  p = Telemetry(inputs=['temperature'], subscribe=True)
  while iter < 100:
    p.run(iter)
    time.sleep(0.1)
    iter += 1
    