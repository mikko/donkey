import os
import pygame

os.environ["DISPLAY"] = "0"
os.environ["home"] = "/home/pi"
os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

dir_path = os.path.dirname(os.path.realpath(__file__))


class Subwoofer:
  def play(self, file, stop_others=True, loop=True):
    loops = -1 if loop else 0
    try:
      if stop_others:
        for _, media in self.media.items():
          media.stop()
      self.media[file].play(loops=loops)
    except:
      print("Error playing", file)

  def __init__(self):
    self.on = True
    self.mode = ''
    self.recording = False
    self.emergency_brake = False
    pygame.mixer.init()
    print("starting subwoofer")
    try:
      self.media = {
        "ai": pygame.mixer.Sound("%s/../../media/ai.ogg" % dir_path),
        "emergency": pygame.mixer.Sound("%s/../../media/emergency.ogg" % dir_path),
        "idle": pygame.mixer.Sound("%s/../../media/idle.ogg" % dir_path),
        "recording": pygame.mixer.Sound("%s/../../media/recording.ogg" % dir_path)
      }
    except Exception as e:
      print("Error loading sound files")
      print(e)
      self.on = False
      self.media = {}

    self.play("idle")

  def run(self, mode, recording, emergency_brake):
    if self.on:
        prev_mode = self.mode
        prev_recording = self.recording
        prev_emergency = self.emergency_brake
        self.mode = mode
        self.recording = recording
        self.emergency_brake = emergency_brake
        # Start recording music if recording changed and now on
        if (prev_recording != recording and recording):
            self.play("recording")
        # Start AI music if mode changed and is now "local"
        elif (prev_mode != mode and mode == "local"): # local mode means AI driven
            self.play("ai")
        # Start idle music if mode changed to something else and not recording
        elif (prev_mode != mode and not recording ):
            self.play("idle")
        # Play emergency sound on top of other music and do not loop
        if (prev_emergency != emergency_brake and emergency_brake):
            self.play("emergency", False, False)

  def shutdown(self):
    # indicate that the thread should be stopped
    self.on = False
    print('stopping subwoofer')

if __name__ == "__main__":
    import time
    iter = 0
    p = Subwoofer()
    print("user mode")
    p.run("user", False, False)
    time.sleep(5)
    print("user mode recording")
    p.run("user", True, False)
    time.sleep(3)
    print("emergency brake")
    p.run("user", True, True)
    time.sleep(3)
    print("AI mode")
    p.run("local", True, False)
    time.sleep(5)
