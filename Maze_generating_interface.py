import numpy
import tkinter

# Class dealing with the cells
class Cell():
    FILLED_COLOR_BG = "blue"
    EMPTY_COLOR_BG = "white"
    FILLED_COLOR_BORDER = "blue"
    EMPTY_COLOR_BORDER = "black"
    EXIT_COLOR_BG = "red"  # Define the exit color

    def __init__(self, master, x, y, size, fill=False, is_exit=False):
        self.master = master
        self.abs = x  # Coordinates
        self.ord = y
        self.size = size
        self.fill = fill
        self.is_exit = is_exit  # Track if the cell is the exit

    def _switch(self):
        self.fill = not self.fill
        self.is_exit = False  # Reset exit status if switched back

    def mark_as_exit(self):
        # Mark this cell as the exit
        self.is_exit = True
        self.fill = False  # Exit should not be filled

    def draw(self):
        if self.master != None:
            if self.is_exit:
                fill = Cell.EXIT_COLOR_BG  # Red color for exit
                outline = Cell.EMPTY_COLOR_BORDER
            else:
                fill = Cell.FILLED_COLOR_BG if self.fill else Cell.EMPTY_COLOR_BG
                outline = Cell.FILLED_COLOR_BORDER if self.fill else Cell.EMPTY_COLOR_BORDER

            xmin = self.abs * self.size
            xmax = xmin + self.size
            ymin = self.ord * self.size
            ymax = ymin + self.size

            self.master.create_rectangle(xmin, ymin, xmax, ymax, fill=fill, outline=outline)

class CellGridCanvas(tkinter.Canvas):
    def __init__(self, master, rows_cnt, columns_cnt, cell_size, *args, **kwargs):
        kwargs["width"] = cell_size * columns_cnt
        kwargs["height"] = cell_size * rows_cnt
        tkinter.Canvas.__init__(self, master, *args, **kwargs)

        self.rows_cnt = rows_cnt
        self.columns_cnt = columns_cnt
        self.cell_size = cell_size

        self.grid = []
        for row in range(rows_cnt):
            line = []
            for column in range(columns_cnt):
                line.append(Cell(self, column, row, cell_size))
            self.grid.append(line)

        self.switched = []

        self.bind("<Button-1>", self.handleMouseClick)
        self.bind("<Button-3>", self.handleRightClick)  # Right-click to mark the exit
        self.bind("<B1-Motion>", self.handleMouseMotion)
        self.bind("<ButtonRelease-1>", lambda event: self.switched.clear())

        self.draw()

    def draw(self):
        for row in self.grid:
            for cell in row:
                cell.draw()

    def _eventCoords(self, event):
        row = int(event.y / self.cell_size)
        column = int(event.x / self.cell_size)
        return row, column

    def handleMouseClick(self, event):
        row, column = self._eventCoords(event)
        cell = self.grid[row][column]
        cell._switch()
        cell.draw()
        self.switched.append(cell)

    def handleRightClick(self, event):
        # Right-click to mark the exit
        row, column = self._eventCoords(event)
        for line in self.grid:
            for cell in line:
                cell.is_exit = False
        cell = self.grid[row][column]
        cell.mark_as_exit()
        self.draw()

    def handleMouseMotion(self, event):
        row, column = self._eventCoords(event)
        try:
            cell = self.grid[row][column]
        except IndexError:
            return
        if cell not in self.switched:
            cell._switch()
            cell.draw()
            self.switched.append(cell)

    def clear(self):
        for row in self.grid:
            for cell in row:
                cell.fill = False
                cell.is_exit = False
        self.draw()

    def to_array(self):
        array = numpy.zeros((self.rows_cnt, self.columns_cnt))
        for i, row in enumerate(self.grid):
            for j, cell in enumerate(row):
                if cell.fill:
                    array[i, j] = 1
                if cell.is_exit:
                    array[i, j] = 2  # Mark exit as 2
        return array

class CellGrid(tkinter.Frame):
    CELL_SIZE = 20
    def __init__(self, *args, **kwargs):
        tkinter.Frame.__init__(self, *args, **kwargs)
        self.grid_canvas = None

    def draw(self, rows_cnt, columns_cnt, cell_size=None):
        if self.grid_canvas:
            self.grid_canvas.destroy()

        cell_size = cell_size or CellGrid.CELL_SIZE
        self.grid_canvas = CellGridCanvas(self, rows_cnt, columns_cnt, cell_size)
        self.grid_canvas.pack()

    def clear(self):
        self.grid_canvas.clear()

    def to_array(self):
        return self.grid_canvas.to_array()

class App(tkinter.Tk):
    HMARGIN = 10
    DEFAULT_ROWS = 10
    DEFAULT_COLUMNS = 10

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Get out of the maze!")

        frame = tkinter.Frame(self)
        frame.pack(padx=self.HMARGIN)
        lbl = tkinter.Label(frame, text="Set the maze dimensions:")
        lbl.grid(row=0, pady=5)
        lbl = tkinter.Label(frame, text="Rows")
        lbl.grid(row=1, column=0, sticky=tkinter.W)
        self.rows_entry = tkinter.Entry(frame)
        self.rows_entry.insert(tkinter.END, str(self.DEFAULT_ROWS))
        self.rows_entry.grid(row=1, column=1)
        lbl = tkinter.Label(frame, text="Columns", anchor="w")
        lbl.grid(row=2, column=0, sticky=tkinter.W)
        self.columns_entry = tkinter.Entry(frame)
        self.columns_entry.insert(tkinter.END, str(self.DEFAULT_COLUMNS))
        self.columns_entry.grid(row=2, column=1)

        self.grid = CellGrid(self)
        btn = tkinter.Button(self, text="Refresh dimensions", command=self.onRefresh)
        btn.pack(pady=5)
        lbl = tkinter.Label(self, text="Draw the path, right-click to mark the exit")
        lbl.pack(padx=self.HMARGIN)
        self.grid.draw(self.DEFAULT_ROWS, self.DEFAULT_COLUMNS)
        self.grid.pack(padx=self.HMARGIN, pady=10)

        btn_frame = tkinter.Frame(self)
        btn_frame.pack(pady=10)

        btn = tkinter.Button(btn_frame, text="Create Matrix", command=self.onSave)
        btn.pack(side=tkinter.RIGHT)

        btn = tkinter.Button(btn_frame, text="Clear", command=self.onClear)
        btn.pack(side=tkinter.RIGHT, padx=5)

    def onSave(self):
        self.A = self.grid.to_array()
        self.destroy()

    def onClear(self):
        self.grid.clear()

    def onRefresh(self):
        try:
            rows = int(self.rows_entry.get())
        except ValueError:
            pass
        try:
            columns = int(self.columns_entry.get())
        except ValueError:
            pass
        self.grid.draw(rows, columns)

if __name__ == "__main__":
    app = App()
    app.mainloop()
