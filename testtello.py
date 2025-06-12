
from djitellopy import Tello
tello = Tello()
tello.connect()
tello.streamon()
print("Tello connected and streaming video.")