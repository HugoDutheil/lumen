import numpy as np

class Parser:

    def parse(self, file: str)-> np.ndarray:
        matrix = []
        with open(file, "r") as f:
            data = f.read()
            lines = data.split("\n")
            lines = [line for line in lines if len(line) > 0]
            for line in lines:
                matrix.append(line.split(" "))
        return np.array(matrix)
    
    def save(self, file:str, matrix: np.ndarray) -> None:
        string = ""
        for row in matrix:
            for value in row:
                string += f"{value} "
            string = string[:-1]
            string += "\n"

        with open(file, "w") as f:
            f.write(string)
        return
