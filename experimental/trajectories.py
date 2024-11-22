import tyro
from mvdatasets.geometry.trajectories import (
    generate_spiral_path,
    generate_ellipse_path_z,
    generate_ellipse_path_y,
    generate_interpolated_path,
)
from config import Args

# TODO: write tests for the functions in mvdatasets.geometry.trajectories


def main(args: Args):
    pass


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
