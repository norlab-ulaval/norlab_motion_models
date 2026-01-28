import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import numpy as np




def main():
    # --- Define two shapes ---
    poly1 = Polygon([
        (0, 0),
        (4, 0),
        (4, 3),
        (0, 3)
    ])

    poly2 = Polygon([
        (2, 1),
        (6, 1),
        (6, 4),
        (2, 4)
    ])

    # --- Compute intersection ---
    intersection = poly1.intersection(poly2)

    # --- Plot ---
    fig, ax = plt.subplots()

    plot_polygon(ax, poly1, color="blue", label="Polygon 1")
    plot_polygon(ax, poly2, color="green", label="Polygon 2")

    if not intersection.is_empty:
        plot_polygon(ax, intersection, color="red", label="Intersection", alpha=0.8)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend()
    ax.set_title("Shapely Polygon Intersection")

    plt.show()


if __name__ == "__main__":
    main()
