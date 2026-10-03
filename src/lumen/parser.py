import numpy as np
import os
import logging

logger: logging.Logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

def is_writable(path: os.PathLike) -> bool:
    result = split_dir_file(path)
    if result is None:
        return False
    
    return os.access(path, os.W_OK)

def split_dir_file(path: os.PathLike) -> tuble[os.PathLike, os.PathLike]:
    print("Attempting to split ", path)
    if os.path.isdir(path):
        dir_name = path
        file_name = f"PCA_matrix.mtx"
        print("Was directory: created a file here")
    
    elif path.endswith(".mtx"):
        print("Path ended with .mtx, checking if dir exits")
        dir_name, file_name = os.path.split(path)
        if not os.path.isdir(dir_name):
            logger.error(f"No such directory: {dir_name}")
            return save_to_new_path(title, U, S, Vt)
        if not os.path.isfile(path):
            print("File didn't exist yet. Creating it...")
            with open(path, 'w'): # This creates a file if it doesn't exist yet
                pass
    else:
        print("Path was neither a valid directory, nor a valid file path")
        return None
    print("Validated path!\n\n")
    return dir_name, file_name


# TODO: define what a bad path is
def invalid_path(path:os.PathLike) -> bool:

    if not is_writable(path):
        logger.info(f"Can't write to location {path}. Insufficient privilege")
        return True
    
    result = split_dir_file(path)
    if result is None:
        logger.info("Invalid location given, expecting either a directory or a file with the *.mtx extension.")
        return True
        
    return False

class Parser:

    def ask_for_new_path(self):
        new_path = "/"
        # Doing it this way so it doesn't throw an error on the tmp path 
        # I'm setting which looks weird to the user since they never call it 
        while True: 
            new_path = input("Enter new path to save data : ")
            if not invalid_path(new_path):
                break
        return new_path
    
    def save_to_new_path(self, title:str, U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> None:
            new_path = self.ask_for_new_path()
            return self.save(new_path, title, U, S, Vt)

    def save(self, path: os.PathLike, title:str, U: np.ndarray, S: np.ndarray, Vt: np.ndarray) -> None:
        logger.info(f"Attempting to write to {path}")

        if (result := split_dir_file(path)) is None:
            logger.error("Invalid path given. The given path either doesn't exist or isn't a valid file. Expecting either a directory or a *.mtx file")
            return self.save_to_new_path(title, U, S, Vt)
        
        path = os.path.join(*result)
        if not is_writable(path):
            logger.error(f"Can't write to location {path} : Insufficient privilege")
            logger.info(f'Would you like to enter a new location?')
            return self.save_to_new_path(title, U, S, Vt) 

        if os.path.getsize(path) > 0:
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
                    return self.save_to_new_path(title, U, S, Vt) 
                    
                    


        string = f'{title}\n'
        for matrix in (U, S, Vt):
            for row in matrix:
                for value in row:
                    string += f"{value} "
                string = string[:-1]
                string += "\n"
            string+="&\n"
        string = string[:-3] #This deletes the final three symbols which are \n&\n, the newline of the final line, the last seperator and it's newline symbol

        with open(path, 'w') as f:
            f.write(string)
        return True


    def parse(self, file: str)-> np.ndarray:
        matrix = []
        with open(file, "r") as f:
            text = f.read()
        
        lines = text.split("\n")
        title = lines[0]
        data = lines[1:]

        U, S, Vt = [], [], [] 
        line_index = 0
        for matrix in (U, S, Vt):
            while line_index < len(data):
                if data[line_index] == '&':
                    line_index += 1
                    break
                matrix.append([float(value) for value in data[line_index].split(" ")])
                line_index += 1


        return np.array(U), np.array(S), np.array(Vt)
    


