import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils.load_data import load_vix


def main() -> None:
    vix = load_vix()
    print("VIX head:")
    print(vix.head())
    print("\nVIX tail:")
    print(vix.tail())
    print("\nVIX describe():")
    print(vix.describe())


if __name__ == "__main__":
    main()
