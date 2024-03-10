""""""
import numpy as np  # for data type


# Constantes class
class Constantes():
    """Define a class containing"""
    def __init__(self, column, value, name=None, **kwargs) -> None:
        self.column = column
        self.value = value
        self.name = name
        self.kwargs = kwargs
        if self.name is None:
            self.name = self.value

        # Set data accessing dictionnary
        self.__set_dict__()

    def __set_dict__(self):
        """Set the dictionnary of str and associated index,
        and the respective attributes from **kwargs"""
        self.attributes_ = {
            "column": self.column,
            "value": self.value,
            "name": self.name,
            **self.kwargs
        }
        self.key_index_ = {
            idx: arg for idx, arg in enumerate(self.attributes_.values())
        }

        # Set attributes
        for key, val in self.kwargs.items():
            setattr(self, key, val)

        self.n_key = max(self.key_index_.keys())

    def __getitem__(self, key):
        """Allow the user to retrieve element with str or int key."""
        if isinstance(key, str):
            return self.attributes_.get(key, None)
        elif isinstance(key, int):
            return self.key_index_.get(key)
        elif isinstance(key, Constantes):
            print("hey")
            return self.__getitem__(key.value)
        else:
            return None

    def __bool__(self):
        """Define the behavior of if(self)"""
        return bool(self.value)

    def __contains__(self, value):
        """Defines (in) behavior"""
        return self.__eq__(value)

    def __eq__(self, other):
        """Define equality (==) behavior"""
        if isinstance(other, Constantes):
            return self.__eq__(other.value)

        if isinstance(self.value, bool):
            condition = other != 0
            return (
                condition if self.value else
                ~condition
            )

        return self.value == other

    def __gt__(self, other):
        """Define greater than (>) behavior"""
        if isinstance(other, Constantes):
            return self.__gt__(other.value)

        if isinstance(self.value, bool):
            condition = 0 < other
            return (
                condition if self.value else
                ~condition
            )
        return self.value > other

    def __ge__(self, other):
        """Define greater or equal (>=) behavior"""
        if isinstance(other, Constantes):
            return self.__ge__(other.value)

        if isinstance(self.value, bool):
            condition = 0 <= other
            return (
                condition if self.value else
                ~condition
            )
        return self.value >= other

    def __lt__(self, other):
        """Define less than (<) behavior"""
        if isinstance(other, Constantes):
            return self.__lt__(other.value)

        if isinstance(self.value, bool):
            condition = 0 > other
            return (
                condition if self.value else
                ~condition
            )
        return self.value < other

    def __le__(self, other):
        """Define less or equal (<=) behavior"""
        if isinstance(other, Constantes):
            return self.__le__(other.value)

        if isinstance(self.value, bool):
            condition = 0 >= other
            return (
                condition if self.value else
                ~condition
            )
        return self.value <= other

    def __repr__(self) -> str:
        """Representation value of the object"""
        return f"{self.value}"

    def __str__(self) -> str:
        """Printable value of the object"""
        return f"Constantes: {self.column} = {self.value}"


# Directory path
OUTPUT_DIRNAME = "out"
DATA_DIRNAME = "data"
SOURCE_DIRNAME = "src"

# Column, Val association
WT = Constantes("Condition", "WT")
KI = Constantes("Condition", "KI")
CD3 = Constantes("Type", "CD3")
LY6 = Constantes("Type", "Ly6", "LY6")
IN_TUMOR = Constantes("Mask", 1, "IN-TUMOR")
OUT_TUMOR = Constantes("Mask", 0, "OUT-TUMOR")

IN_FIBER = Constantes("Density20", True, "FIBER")
OUT_FIBER = Constantes("Density20", False, "NO-FIBER")

CELLS = Constantes("Cells", True)
CELLS100UM = Constantes("Cells100um", True)

# Classes column
T_PLUS = Constantes("t_plus", 1, "t-plus")
T_ENRICHED = Constantes("t_enrich", 1, "t-enrich")
T_ENRICHED_2 = Constantes("t_enrich_2", 1, "t-enrich-2")

# Columns definition
str_columns = ("Condition", "FileName", "Type")
int_columns = ("Mask",)
unsigned_columns = ("X", "Y")

# With angle
float20_columns = [
    "Angle20", "Coherency20", "Energy20", "MeanInt20",
   "VarInt20", "Density20", "VarDensity20", "OrientationRef20"
]
float60_columns = [
    "Angle60", "Coherency60", "Energy60", "MeanInt60",
   "VarInt60", "Density60", "VarDensity60", "OrientationRef60"
]
float100_columns = [
    "Angle100", "Coherency100", "Energy100", "MeanInt100",
    "VarInt100", "Density100", "VarDensity100", "OrientationRef100"
]
float140_columns = [
    "Angle140", "Coherency140", "Energy140", "MeanInt140",
    "VarInt140", "Density140", "VarDensity140", "OrientationRef140"
]

# Without angle
float20_columns_unloc = float20_columns.copy()
float60_columns_unloc = float60_columns.copy()
float100_columns_unloc = float100_columns.copy()
float140_columns_unloc = float140_columns.copy()

angle_columns = (
    float20_columns_unloc.pop(float20_columns.index("Angle20")),
    float60_columns_unloc.pop(float60_columns_unloc.index("Angle60")),
    float100_columns_unloc.pop(float100_columns_unloc.index("Angle100")),
    float140_columns_unloc.pop(float140_columns_unloc.index("Angle140"))
)

dist_columns = ("Dist",)
shape_columns = ("Frac",)
cells_shape_columns = ("CellArea", "CellEcc")
cells_dist_columns = ("MinDist", "MedDist")
cells_100um_columns = ("MinDist100um", "MedDist100um", "CellArea100um", "CellEcc100um")
cells_columns = ("Cells", "Cells100um")  # "Cells" converted to float due to NaN exception

float_columns = (
    *float20_columns, *float60_columns, *float100_columns, *float140_columns,
    *dist_columns, *shape_columns, *cells_shape_columns, *cells_dist_columns,
    *cells_100um_columns, *cells_columns
)

aberrant_columns = (
    *float20_columns,
    *float60_columns,
    *float100_columns,
    *float140_columns
)
aberrant_columns = [
    column for column in aberrant_columns
    if column not in ("Density60", "Density100", "Density140")
]

# Type for each column
data_type = {
    **dict.fromkeys(str_columns, object),
    **dict.fromkeys(unsigned_columns, np.uint32),
    **dict.fromkeys(float_columns, np.float64),
    **dict.fromkeys(int_columns, np.int32)
}

if __name__ == "__main__":
    hey = Constantes("Condition", "KI")
    print(f"column = {hey[0]}")
    print(f"value = {hey[1]}")
