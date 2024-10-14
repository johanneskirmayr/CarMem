import os


def load_chat_history(memory):
    conversation = ""
    for msg in memory.chat_memory.messages:
        if msg.type == "human":
            conversation += f"Human: {msg.content}\n"
        else:
            conversation += f"AI: {msg.content}\n"
    return conversation


def get_project_root():
    """
    Get the project root by finding the directory that contains .git
    """
    current_dir = os.getcwd()

    # Define the markers that signify the root of the project
    markers = [".git"]

    # Start from the current directory and move up until we find a marker
    root = current_dir
    while True:
        if any(os.path.exists(os.path.join(root, marker)) for marker in markers):
            # We found the root
            return root
        parent_dir = os.path.dirname(root)
        if parent_dir == root:
            # We've reached the root of the filesystem without finding a marker
            raise FileNotFoundError("Could not find project root")
        root = parent_dir


def stringify_conversations(messages) -> str:
    formatted_messages = []
    for msg in messages:
        if msg["role"] == "User":
            formatted_messages.append(f"{msg['role']} {msg['name']}: {msg['content']}")
        else:
            formatted_messages.append(f"{msg['role']}: {msg['content']}")

    result_string = "\n".join(formatted_messages)
    return result_string
