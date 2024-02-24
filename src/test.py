import os
file_path = os.path.realpath(__file__)
print(file_path)
print(os.path.dirname(file_path))
print("".join(file_path.split("src/")[:-1]) + "data/ddd")
print(__file__)
