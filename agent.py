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
PROFESSIONS[Type.FIGHTER] = Profession(20,100,1,1)

AGENT_FIELDS = 6
# MAX_TYPES = len(Type) - 1
MAX_HEALTH = 200
MAX_POWER = 100
MAX_ROW = 8
MAX_COL = 8
MAX_RANGE = 5


class Agent:
  def __init__(self, id, cell_id, type, team, health, power, range):
    # internal
    self.cell_id = cell_id
    self.id = id

    # state
    self.type = type
    self.team = team
    self.health = health
    self.power = power
    self.range = range
    self.is_active = False

  def normalize(self, active_team):
    t = 0 if active_team == self.team else 1
    return float(self.is_active), self.type.value, t, self.health / MAX_HEALTH, self.power / MAX_POWER, self.range / MAX_RANGE


  



