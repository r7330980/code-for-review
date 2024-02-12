import re


def find_variable_indices(input_str, variable_name):
    variable_name_pattern = re.compile(
        r'[^a-zA-Z0-9](%s)[^a-zA-Z0-9]' % variable_name)
    indices = []
    for match in variable_name_pattern.finditer(input_str):
        indices.append(match.span(1))
    return indices