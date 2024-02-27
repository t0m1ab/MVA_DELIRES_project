import os
import matplotlib.pyplot as plt


def plot_sequence(seq: list, path: str = None, title: str = None):
    path = path if path is not None else os.getcwd()
    title = title if title is not None else "sequence"
    plt.plot(seq)
    plt.xlabel("index")
    plt.ylabel("value")
    plt.title(f"{title} (length={len(seq)})")
    plt.savefig(os.path.join(path, title + ".png"))
    plt.close()


def main():
    plot_sequence(seq=[1,2,2,2,3,4,5,5,5,6], path=None, title="my_sequence")


if __name__ == '__main__':
    main()