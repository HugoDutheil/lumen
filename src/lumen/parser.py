import numpy as np
import os

# TODO: define what a bad path is
def invalid_path(path:os.Pathlike) -> bool:
    if not path.endswith(".mtx"):
        logger.info("Expecting a *.mtx file path")
        return True
    if os.is_dir(path):
        logger.info("Path given is directory. Please provide a file name.")
        return True
    return False

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
    
    def save(self, file: os.Pathlike, title:str, U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> None:
        if not os.isfile(file):
            logger.error(f"No such file found: {file}")
            return False

        if os.path.getsize(file) > 0:
            confirmation: str = ""
            while confirmation not in ('y', 'n', 'c'):
                confirmation = input("Output file is not empty. Do you want to erase content or change file path? [y/n/c]")
            match(confirmation):
                case 'y': 
                    pass
                case 'n': 
                    logger.warning("Unable to save file. User aborted action")
                    return
                case 'c':
                    new_path = "/"
                    while invalid_path(new_path):
                        new_path = input("Enter new path to save data : ")
                    return self.save(new_path, title, U, S, Vt) 
                    
                    


        string = f'{title}\n'
        for matrix in (U, S, Vt):
            for row in matrix:
                for value in row:
                    string += f"{value} "
                string = string[:-1]
                string += "\n"
            string+="&\n"
        string = string[:-1]

        with open(file, "w") as f:
            f.write(string)
        return True
