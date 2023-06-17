from enum import Enum

class Rotation(Enum):
    Left = 0
    Up = 1
    Right = 2
    Down = 3

class Type(Enum):
    Berserker = 0
    Runner = 1

class Profession():
  def __init__(self, power, health, speed):
    self.power = power
    self.health = health
    self.speed = speed

PROFESSIONS = {}
PROFESSIONS[Type.Berserker] = Profession(50,200,1)
PROFESSIONS[Type.Runner] = Profession(50,100,2)

MAX_ROTATIONS = len(Rotation) - 1
MAX_TYPES = len(Type) - 1
MAX_TEAMS = 1
MAX_HEALTH = 200
MAX_SPEED = 2
MAX_POWER = 200
AGENT_FIELDS = 6


class Agent:
  def __init__(self, location, rotation, type, team, size):
    self.location = location
    self.size = size
    self.rotation = rotation
    self.type = type
    self.team = team
    self.health = PROFESSIONS[type].health

  def normalize(self):
    return self.rotation.value / MAX_ROTATIONS, self.type.value / MAX_TYPES, self.team / MAX_TEAMS, self.health / MAX_HEALTH, self.location[0] / (self.size-1), self.location[1] / (self.size-1)


  



