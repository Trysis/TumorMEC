from . import auxiliary

FORMAT_STR = "-"
FORMAT_REPEAT = 20
FORMAT = lambda x: f"{'-'*FORMAT_REPEAT} {x} {'-'*FORMAT_REPEAT}\n"


def select_padding(padding, values=None):
    """"""
    if isinstance(padding, int):
        return padding
    elif isinstance(padding, str):
        if padding == "adaptive":
            if values is None:
                raise Exception(
                    "{values} needs to be specified"
                    ", when {padding=adaptive}"
                )
            else:
                if isinstance(values, (list, tuple, set)):
                    padding = max(values, key=lambda x: len(str(x)))
                else:
                    padding = len(str(values))
    return padding
                
def generic_summary(to_write, filepath=None, mode="a"):
    """"""
    if filepath is not None:
        with open(filepath, mode):
            filepath.write(to_write)
    return to_write

def arg_summary(
    argument, value="", padding=None, sep=":",
    filepath=None, mode="a"
):
    """
    argument: str
    value: bool, int, str, float or writable to str
    sep: str
    padding: int
    """
    if padding is None or padding < 0:
        padding = 0
    # Writing
    to_write = f"{argument:<{padding}s} {sep} {value}\n"
    return generic_summary(to_write, filepath=filepath, mode=mode)


def mapped_summary(
    map, map_sep="<==>", 
    label=None, label_sep=":", label_padding="adaptive",
    filepath=None, mode="a"
):
    if map is None or not isinstance(map, dict):
        return ""
    #
    to_write = ""
    if label is None:
        for key, val in map.items():
            to_write += f"{key} {map_sep} {val}\n"
    else:
        for key, val in map.items():
            to_write += arg_summary(
                label, f"{key} {map_sep} {val}",
                padding=select_padding(label_padding, map.values()),
                sep=label_sep
            )
    return generic_summary(to_write, filepath=filepath, mode=mode)

def xy_summary(
        x, y, unique_groups=None,
        x_label="x shape", y_label="y shape", groups_label="groups",
        padding=10, filepath=None, mode="a"
    ):
    """"""
    if x_label is None:
        x_label = "x"
    if y_label is None:
        y_label = "y"
    #
    x_shape_str = f"{x.shape[0]} lines, {x.shape[1]} columns"
    y_shape_str = f"{y.shape[0]} lines, {y.shape[1]} columns"
    groups_str = ""
    if unique_groups is not None:
        groups_formated = auxiliary.format_by_rows(
            unique_groups, ncol=len(unique_groups), spacing=2
        )
        groups_str = arg_summary(groups_label, groups_formated, padding=padding)
    #
    to_write = arg_summary(x_label, x_shape_str, padding=padding)
    to_write += arg_summary(y_label, y_shape_str, padding=padding)
    to_write += groups_str

    return generic_summary(to_write, filepath=filepath, mode=mode)

def df_summary(
        x, y, unique_groups=None, x_columns=None, y_columns=None, groups_columns=None,
        title="DataFrame", filepath=None, mode="a"
):
    to_write = FORMAT(title) + "\n"
    to_write += xy_summary(x, y, unique_groups, padding=10)
    to_write += f"{arg_summary('x_columns')}{auxiliary.format_by_rows(x_columns, ncol=6)}\n\n"
    to_write += f"{arg_summary('y_columns')}{auxiliary.format_by_rows(y_columns, ncol=6)}\n\n"
    to_write += f"{arg_summary('grouped by')}{auxiliary.format_by_rows(groups_columns, ncol=6)}\n\n"

    return generic_summary(to_write, filepath=filepath, mode=mode)


if __name__ == "__main__":
    df_summary("", "", "")
    print(arg_summary("hey", 5, padding=11))
