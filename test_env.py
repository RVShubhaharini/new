try:
    import pandas
    print("Pandas OK")
except ImportError as e:
    print(f"Pandas Error: {e}")

try:
    import numpy
    print("Numpy OK")
except ImportError as e:
    print(f"Numpy Error: {e}")

try:
    import sklearn
    print("Sklearn OK")
except ImportError as e:
    print(f"Sklearn Error: {e}")

try:
    import scipy
    print("Scipy OK")
except ImportError as e:
    print(f"Scipy Error: {e}")

try:
    import torch
    print("Torch OK")
except ImportError as e:
    print(f"Torch Error: {e}")
