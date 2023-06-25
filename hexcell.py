from enum import Enum

class Direction(Enum):
    LEFT = 0
    TOP_LEFT = 1
    TOP_RIGHT = 2
    RIGHT = 3
    BOTTOM_RIGHT = 4
    BOTTOM_LEFT = 5

id_counter = 0

class HexCell:
    def __init__(self, data=None):
        self.data = data
        self.id = id_counter
        id_counter += 1
        self.neighbors = {direction: None for direction in Direction}

class HexGrid:
    def __init__(self):
        self.size = 5
        self.top_left_corner = None
        self.bottom_right_corner = None
        self.initialize_grid()

    def initialize_grid(self):
        # top half
        last_row = []
        current_row = []
        id = 0
        for row in range(0,self.size):
            last_row = current_row
            current_row = []
            lastCell = None
            for col in range(self.size + row):
                cell = HexCell()
                # remember top left corner
                if row == 0 and col == 0:
                    self.top_left_corner = cell

                # connect to top rows
                # nothing to do for top row
                if row > 0:
                    # if not first element in row
                    if col > 0:
                        # top left
                        top_left_cell = last_row[col-1]
                        cell.neighbors[Direction.TOP_LEFT] = top_left_cell
                        top_left_cell.neighbors[Direction.BOTTOM_RIGHT] = cell
                    # if not last element in row
                    if col < self.size + row - 1:
                        # top right
                        top_right_cell = last_row[col]
                        cell.neighbors[Direction.TOP_RIGHT] = top_right_cell
                        top_right_cell.neighbors[Direction.BOTTOM_LEFT] = cell

                # connect to left rows
                # if not first element in row
                if col > 0:
                    cell.neighbors[Direction.LEFT] = lastCell
                    lastCell.neighbors[Direction.RIGHT] = cell
                # set the last cell either way. this ensures that second cell in row connects to first, not last of previous row
                lastCell = cell
                current_row.append(cell)

        # bottom half
        for row in range(self.size, self.size*2 - 1):
            last_row = current_row
            current_row = []
            lastCell = None
            for col in range(self.size*2 - 2 - (row - self.size)):
                cell = HexCell()
                # remember bottom right corner
                if row == self.size*2 - 2 and col == self.size - 1:
                    self.bottom_right_corner = cell

                # connect to top rows
                # top left
                top_left_cell = last_row[col]
                cell.neighbors[Direction.TOP_LEFT] = top_left_cell
                top_left_cell.neighbors[Direction.BOTTOM_RIGHT] = cell
                # top right
                top_right_cell = last_row[col+1]
                cell.neighbors[Direction.TOP_RIGHT] = top_right_cell
                top_right_cell.neighbors[Direction.BOTTOM_LEFT] = cell

                # connect to left rows
                # if not first element in row
                if col > 0:
                    cell.neighbors[Direction.LEFT] = lastCell
                    lastCell.neighbors[Direction.RIGHT] = cell
                # set the last cell either way. this ensures that second cell in row connects to first, not last of previous row
                lastCell = cell
                current_row.append(cell)

