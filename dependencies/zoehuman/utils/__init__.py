def str_or_empty(input_str):
    """If the input string equals to "empty", return "", else return itself.

    Args:
        input_str (str): any string

    Returns:
        str: input string or ""
    """
    if input_str.lower() == 'empty':
        return ''
    return input_str
