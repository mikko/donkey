import os
import pygame
import random

os.environ["DISPLAY"] = "0"
os.environ["home"] = "/home/pi"
os.environ["XDG_RUNTIME_DIR"] = "/run/user/1000"

dir_path = os.path.dirname(os.path.realpath(__file__))

def get_random():
    return random.randint(100, 300)

class Subwoofer:
  def play_random(self):
    try:
        sound_file = random.choice(os.listdir("%s/../../media/random" % dir_path))
        sound = pygame.mixer.Sound("%s/../../media/random/%s" % (dir_path, sound_file))
        sound.set_volume(1.0)
        sound.play(loops=0)
    except:
        print("Error playing random sound clip")

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
    self.until_random = get_random()
    print("starting subwoofer")
    try:
      pygame.mixer.init()
      self.media = {
        "ai": pygame.mixer.Sound("%s/../../media/ai.ogg" % dir_path),
        "emergency": pygame.mixer.Sound("%s/../../media/emergency.ogg" % dir_path),
        "idle": pygame.mixer.Sound("%s/../../media/idle.ogg" % dir_path),
        "recording": pygame.mixer.Sound("%s/../../media/recording.ogg" % dir_path),
        "cruising": pygame.mixer.Sound("%s/../../media/cruising.ogg" % dir_path)
      }
      self.media["idle"].set_volume(0.3)
      self.media["cruising"].set_volume(0.3)
      self.media["ai"].set_volume(0.6)

      self.play("idle")

    except Exception as e:
      print("Error loading sound files")
      print(e)
      # self.on = False
      self.on = False
      self.media = {}


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
        elif (prev_mode != mode and mode == "local_angle"): # local_angle mode means AI driven steering
            self.play("cruising")
        # Start idle music if mode changed to something else and not recording
        elif ((prev_mode != mode or prev_recording != recording) and not recording):
            self.play("idle")
        # Play emergency sound on top of other music and do not loop
        if (prev_emergency != emergency_brake and emergency_brake):
            self.play("emergency", False, False)

        if (mode == "local"):
            self.until_random = self.until_random - 1
            if self.until_random < 1:
#                self.play_random()
                self.until_random = get_random()

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
    time.sleep(2)
    print("user mode recording")
    p.run("user", True, False)
    time.sleep(3)
    print("emergency brake")
    p.run("user", True, True)
    time.sleep(3)
    print("AI mode")
    p.run("local", True, False)
    time.sleep(5)

    for i in range(10000):
        p.run('user', False, False)
        time.sleep(0.1)
