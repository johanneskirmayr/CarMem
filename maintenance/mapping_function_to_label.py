def function_to_label(tool_call):
    mapping = {
        "insert_preference": 0,
        "pass_preference": 1,
        "update_preference": 2,
        "append_preference": 3
    }
    if tool_call in mapping:
        return mapping[tool_call]
    else:
        return KeyError