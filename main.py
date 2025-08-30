from typing import NamedTuple

from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")

print(ds.shape)
print(ds['train'][0])



PageDimensions = NamedTuple("PageDimensions", [('width', int), ('height', int)])


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.

    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
