# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from __future__ import print_function
import torch
import numpy as np
import pandas as pd


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+8 to toggle the breakpoint.
    x = torch.empty(5, 3)
    print(x)
    x = torch.rand(5, 3)
    print(x)

    x = torch.zeros(5, 3, dtype=torch.long)
    print(x)

    x = torch.tensor([5.5, 3])
    print(x)

    x = torch.rand(5, 3)
    x = x.new_ones(5, 2, dtype=torch.double)
    print(x)

    x = torch.randn_like(x, dtype=torch.float)
    print(x)
    print(x.size())

    y = torch.empty(5, 2, dtype=torch.float)
    torch.add(x, y, out=y)
    print(y)
    




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
