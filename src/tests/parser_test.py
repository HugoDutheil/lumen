import numpy as np
from lumen.parser import Parser

parser: Parser = Parser()

def test_saving(true: str, expected: str) -> None:
    assert true == expected, "Contents were not properly saved"
    print("==== SUCCESSFULL SAVING ====")
    return

def test_parsing(name: str, true: np.array, expected: np.array) -> None:
    assert np.all(true == expected), f"{name} failed\n"
    print("==== SUCCESSFULL PARSING ====")
    return
    


U_test = np.array(
         [[1.,2.,3.], 
          [4.,5.,6.],
          [7.,8.,9.]]
         )

S_test = np.array(
         [[10.,11.,12.], 
          [13.,14.,15.],
          [16.,17.,18.]]
         )

Vt_test = np.array(
         [[19.,20.,21.], 
          [22.,23.,24.],
          [25.,26.,27.]]
         )

title_test = "~/path/to/source/data.csv"
save_file_path_test = "/tmp/matrix_test.mtx"

parser.save(save_file_path_test, title_test, U_test, S_test, Vt_test)

with open(save_file_path_test, "r") as f:
    contents: str = f.read()

expected = (
    f"{title_test}\n"
    "1.0 2.0 3.0\n"
    "4.0 5.0 6.0\n"
    "7.0 8.0 9.0\n"
    "&\n"
    "10.0 11.0 12.0\n"
    "13.0 14.0 15.0\n"
    "16.0 17.0 18.0\n"
    "&\n"
    "19.0 20.0 21.0\n"
    "22.0 23.0 24.0\n"
    "25.0 26.0 27.0"
)
test_saving(contents, expected)

U_recup, S_recup, Vt_recup = parser.parse(save_file_path_test)

test_parsing("U", U_recup, U_test)
test_parsing("S", S_recup, S_test)
test_parsing("Vt", Vt_recup, Vt_test)

print("===== PASSED ALL TESTS ====")

os.remove(save_file_path_test)
