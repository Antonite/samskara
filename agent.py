from enum import Enum

class Type(Enum):
    FIGHTER = 0

class Profession():
  def __init__(self, power, health, speed, range):
    self.power = power
    self.health = health
    self.speed = speed
    self.range = range

PROFESSIONS = {}
PROFESSIONS[Type.FIGHTER] = Profession(20,100,1)

AGENT_FIELDS = 7
# MAX_TYPES = len(Type) - 1
MAX_HEALTH = 200
MAX_SPEED = 2
MAX_POWER = 100
MAX_ROW = 8
MAX_COL = 8
MAX_RANGE = 5


class Agent:
  def __init__(self, cell_id, type, team, health, power, speed, range):
    # internal
    self.cell_id = cell_id

    # state
    self.type = type
    self.team = team
    self.health = health
    self.power = power
    self.speed = speed
    self.range = range
    self.is_active = False

  def normalize(self):
    return float(self.is_active), self.type.value, self.team, self.health / MAX_HEALTH, self.power / MAX_POWER, self.speed / MAX_SPEED, self.range / MAX_RANGE


  



