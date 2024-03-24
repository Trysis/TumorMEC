from . import auxiliary

FORMAT_STR = "-"
FORMAT_REPEAT = 40
FORMAT = lambda x: f"{'-'*FORMAT_REPEAT} {x} {'-'*FORMAT_REPEAT}\n"
SUB_FORMAT = lambda x: f"{'-'*(FORMAT_REPEAT//2)} {x} {'-'*(FORMAT_REPEAT//2)}\n"


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
                    padding = max(len(str(x)) for x in values)
                else:
                    padding = len(str(values))
    elif padding is None or padding < 0:
        return 0
    return padding
                

def write_summary(to_write, filepath=None, mode="a"):
    """"""
    if filepath is not None:
        with open(filepath, mode) as file_in:
            file_in.write(to_write)
    return to_write


def arg_summary(
    argument, value="", padding=None, sep=":",
    padding_left=None, new_line=True, filepath=None, mode="a"
):
    """
    argument: str
    value: bool, int, str, float or writable to str
    sep: str
    padding: int
    """
    padding = select_padding(padding=padding)
    padding_left = select_padding(padding=padding_left)
    # Writing
    to_write = f"{'':<{padding_left}s}{argument:<{padding}s} {sep} {value}"
    to_write += "\n" if new_line else ""
    return write_summary(to_write, filepath=filepath, mode=mode)


def mapped_summary(
    map, map_sep="==>",
    padding="adaptive", padding_left=None,
    label=None, label_sep=":",
    label_padding=10, label_padding_left=None,
    new_line=False, filepath=None, mode="a"
):
    """"""
    if map is None or not isinstance(map, dict):
        return ""
    #
    to_write = ""
    padding = select_padding(padding, list(map.keys()))
    padding_left = select_padding(padding_left)
    for key, val in map.items():
        line_value = f"{'':<{padding_left}s}{key:<{padding}s} {map_sep} {val}\n"
        if label is None:
            to_write += line_value
        else:
            to_write += arg_summary(
                label, line_value,
                padding=label_padding,
                padding_left=label_padding_left,
                sep=label_sep, new_line=new_line
            )
    return write_summary(to_write, filepath=filepath, mode=mode)


def summarize(
        *args, title=None, subtitle=None,
        args_space=True, filepath=None, mode="a"
):
    """"""
    to_write = ""
    to_write += FORMAT(title) + "\n" if title is not None else ""
    to_write += SUB_FORMAT(subtitle) + "\n" if subtitle is not None else ""
    for v in args:
        to_write += str(v)
        if args_space: to_write += "\n"
    return write_summary(to_write, filepath=filepath, mode=mode)


def xy_summary(
        x, y, unique_groups=None, title=None,
        x_label="x shape", y_label="y shape",
        groups_label="groups", group_spacing=2,
        padding=10, new_line=False,
        filepath=None, mode="a"
    ):
    """"""
    if x_label is None:
        x_label = "x"
    if y_label is None:
        y_label = "y"
    #
    to_write = ""
    x_shape_str = ""
    y_shape_str = ""
    groups_str = ""
    if x is not None:
        x_shape_formated = f"{x.shape[0]} lines, {x.shape[1]} columns"
        x_shape_str = arg_summary(x_label, x_shape_formated, padding=padding)
    if y is not None:
        y_shape_formated = f"{y.shape[0]} lines, {y.shape[1]} columns"
        y_shape_str = arg_summary(y_label, y_shape_formated, padding=padding)
    if unique_groups is not None:
        groups_formated = auxiliary.format_by_rows(
            unique_groups, ncol=len(unique_groups), spacing=group_spacing
        )
        groups_str = arg_summary(groups_label, groups_formated, padding=padding)
    #
    to_write += SUB_FORMAT(title) + "\n" if title is not None else ""
    to_write += x_shape_str
    to_write += y_shape_str
    to_write += groups_str
    to_write += "\n" if new_line else ""
    return write_summary(to_write, filepath=filepath, mode=mode)


def df_summary(
        x, y, unique_groups=None,
        x_columns=None, y_columns=None, groups_columns=None,
        mapped_groups=None, new_line=False,
        title="DataFrame", filepath=None, mode="a"
):
    to_write = FORMAT(title) + "\n"
    to_write += xy_summary(x=x, y=y, unique_groups=unique_groups, padding=10)
    to_write += f"\n{mapped_summary(mapped_groups)}"
    to_write += "" if x_columns is None else \
        f"\n{arg_summary('x_columns')}{auxiliary.format_by_rows(x_columns, ncol=5)}\n" 
    to_write += "" if y_columns is None else \
        f"\n{arg_summary('y_columns')}{auxiliary.format_by_rows(y_columns, ncol=5)}\n"
    to_write += "" if groups_columns is None else \
        f"\n{arg_summary('grouped by')}{auxiliary.format_by_rows(groups_columns, ncol=5)}\n"
    to_write += "\n" if new_line else ""
    return write_summary(to_write, filepath=filepath, mode=mode)


if __name__ == "__main__":
    df_summary("", "", "")
    print(arg_summary("hey", 5, padding=11))
