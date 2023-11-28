
# Column, Val association
WT = ("Condition", "WT")
KI = ("Condition", "KI")
CD3 = ("Type", "CD3")
LY6 = ("Type", "Ly6")
IN_TUMOR = ("Mask", 1)
OUT_TUMOR = ("Mask", 0)

# Columns definition
str_columns = ("Condition", "FileName", "Type")
int_columns = ("Mask",)
unsigned_columns = ("X", "Y")
float20_columns = (
    "Angle20", "Coherency20", "Energy20", "MeanInt20",
   "VarInt20", "Density20", "VarDensity20", "OrientationRef20"
)
float60_columns = (
    "Angle60", "Coherency60", "Energy60", "MeanInt60",
   "VarInt60", "Density60", "VarDensity60", "OrientationRef60"
)
float100_columns = (
    "Angle100", "Coherency100", "Energy100", "MeanInt100",
    "VarInt100", "Density100", "VarDensity100", "OrientationRef100"
)
float140_columns = (
    "Angle140", "Coherency140", "Energy140", "MeanInt140",
    "VarInt140", "Density140", "VarDensity140", "OrientationRef140"
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

if __name__ == "__main__":
    pass
