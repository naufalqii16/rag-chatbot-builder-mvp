import os

EXCLUDE_DIRS = {"myvenv", ".git"}

def print_tree(start_path=".", prefix=""):
    items = sorted(os.listdir(start_path))
    items = [i for i in items if i not in EXCLUDE_DIRS]

    for index, item in enumerate(items):
        path = os.path.join(start_path, item)
        connector = "└── " if index == len(items) - 1 else "├── "

        print(prefix + connector + item)

        if os.path.isdir(path):
            extension = "    " if index == len(items) - 1 else "│   "
            print_tree(path, prefix + extension)

if __name__ == "__main__":
    print_tree(".")
