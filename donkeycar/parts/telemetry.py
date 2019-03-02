import paho.mqtt.client as mqtt

class Telemetry:

  def on_message(self, client, userdata, message):
      print("message received " ,str(message.payload.decode("utf-8")))
      print("message topic=",message.topic)
      print("message qos=",message.qos)
      print("message retain flag=",message.retain)

  def __init__(self, inputs, broker='broker.hivemq.com', subscribe=False):
    self.on = True
    self.inputs = inputs
    self.args = None
    self.client = mqtt.Client()
    if subscribe:
      self.client.on_message = self.on_message
    self.client.connect(broker)
    self.client.loop_start()
    for input in self.inputs:
      self.client.subscribe(input)

  def run_threaded(self, args):
    self.args = args

  def run(self, *args):
    assert len(self.inputs) == len(args)
    for index, arg in enumerate(args):
      self.client.publish(self.inputs[index], arg)

  def shutdown(self):
    # indicate that the thread should be stopped
    self.on = False
    self.client.loop_stop()
    print('stopping telemetry')

  def update(self):
    assert len(self.inputs) == len(self.args)
    for index, arg in enumerate(self.args):
      self.client.publish(self.inputs[index], arg)

if __name__ == "__main__":
  import time
  iter = 0
  p = Telemetry(inputs=['temperature'], subscribe=True)
  while iter < 100:
    p.run(iter)
    time.sleep(0.1)
    iter += 1
