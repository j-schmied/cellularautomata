"""
Plot logic for cellular automata simulations.

Contains various functions for all kinds of plots and data loading.
When called as main, it will generate plots for cell count, cell area,
cell density and cell size median for multiverse runs.
The plots will be saved in the simulation's plots folder.

Authors: Jannik Schmied, Florian Freier
"""
import argparse
import csv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import re
import seaborn as sns
import shutil

from matplotlib.patches import Rectangle
from scipy.interpolate import interp1d
from sys import exit

from .conversion import time_conversion


def plot_linear(ncell_list, path: str, name: str, parameters: dict, log: bool = False, exp_data_path: str = None):
    """
    Plots linear time series, e.g. cell count, colony area, cell density, cell size median.

    Parameters:
        ncell_list (numpy.ndarray)
        path (str)
        name (str)
        parameters (dict)
        log (bool)
        exp_data_path (str)
    """
    plt.figure(figsize=(4*0.9, 3*0.9))
    plt.rc("font", size=7)
    sns.lineplot(x=((np.array(range(len(ncell_list)))*parameters["Allgemein"]["dt"])/parameters["Allgemein"]["dayConst"]), y=ncell_list, markers=True, color="red", label=("CA Simulation" if exp_data_path else None))

    if exp_data_path and "area" in name:
        areas, _, _ = load_exp_set(exp_data_path)
        m = min(areas[0]["x"])
        sns.lineplot(x=areas[0]["x"] - m, y=areas[0]["y"], markers=True, color="black", label="Experiment")

    if exp_data_path and "density" in name:
        _, densities, _ = load_exp_set(exp_data_path)
        m = min(densities[0]["x"])
        sns.lineplot(x=densities[0]["x"] - m, y=densities[0]["y"], markers=True, color="black", label="Experiment")

    if exp_data_path and "median" in name:
        _, _, medians = load_exp_set(exp_data_path)
        m = min(medians[0]["x"])
        sns.lineplot(x=medians[0]["x"] - m, y=medians[0]["y"], markers=True, color="black", label="Experiment")

    if log:
        plt.semilogy()

    plt.xlabel("t [d]")
    plt.ylabel("Count [cells]" if "count" in name else "Colony area [$\mu$m$^2$]" if "area" in name else "Cell density [$10^3$ cells/mm$^2$]" if "density" in name else "Median size [$\mu$m$^2$]")
    plt.title(name.replace('_', ' ') + " development")
    plt.savefig(fname=f"{path}/{name}.pdf", bbox_inches="tight")
    plt.show(block=False)


def plot_multi(attribute: str, yunit: str, output_path: str, output_file_type: str, sim_data: list, exp_data: list = None, log: bool = False, grid: bool = False, title: bool = False, boxshift: bool = False):
    """
    Plots multiple mean time series + SD for colony area, cell density, cell count and cell size median  between experimental data and simulation results

    Parameters:
        attribute (str)
        yunit (str)
        output_path (str)
        output_file_type (str)
        sim_data (list)
        exp_data (list)
        log (bool)
        grid (bool)
        title (bool)
        boxshift (bool)
    """
    fig = plt.figure(figsize=(4*0.9, 3*0.9))
    plt.rc("font", size=7)
    plt.xlabel("t [d]")
    plt.ylabel(f"{'Colony' if attribute == 'area' else 'Cell'} {attribute} [{yunit}]")

    if boxshift:
        shift_width: float = 1.
        plt.xlim(-0.75, 20)

    if boxshift and attribute == "median size":
        plt.ylim(-0.5, 550)

    if grid:
        plt.grid()

    if title:
        plt.title(f"{'Colony' if attribute == 'area' else 'Cell'} {attribute} development")

    if exp_data:
        m = min(exp_data[0]["x"])

        for i in range(len(exp_data)):
            plt.plot(
                exp_data[i]["x"] - m,
                exp_data[i]["y"],
                "o",
                markersize=2,
                color="black",
                label=(f"Experiment" if i == 0 else None),
                zorder=2
            )

    python_lines: list = list()

    for idx, pysim in enumerate(sim_data):
        if f"Cell {attribute}" in pysim:
            python_lines.append(pysim[f"Cell {attribute}"])

    # calc avg
    py_avg: list = list()

    for i in range(len(python_lines[0])):
        sum_value = np.sum([simulation[i] for simulation in python_lines]) / len(python_lines)
        py_avg.append(sum_value)

    # calc std deviation
    py_avg_minus: list = list()
    py_avg_plus: list = list()

    for i in range(len(python_lines[0])):
        std = np.sum([simulation[i] ** 2 for simulation in python_lines]) / len(python_lines)
        py_avg_minus.append(py_avg[i] - np.sqrt(std - (py_avg[i] * py_avg[i])))
        py_avg_plus.append(py_avg[i] + np.sqrt(std - (py_avg[i] * py_avg[i])))

    plt.plot(
        (np.array(pysim["time"]) - shift_width) if boxshift else pysim["time"],
        py_avg,
        "r-",
        linewidth=1.5,
        markeredgewidth=1,
        label="CA Simulation",
        zorder=1
    )
    plt.fill_between(
        (np.array(pysim["time"]) - shift_width) if boxshift else pysim["time"],
        py_avg_minus,
        py_avg_plus,
        facecolor="red",
        alpha=0.25,
        zorder=1
    )

    if log:
        plt.semilogy()

    if attribute == "density":
        # Instead of 10^0, ..., use 1, 2, 5 as yticks
        plt.yticks([1, 2, 5], ['1', '2', '5'])

    plt.legend(frameon=False)
    plt.savefig(os.path.join(output_path, f"{attribute.replace(' ', '_')}.{output_file_type}"), bbox_inches="tight")
    plt.show(block=False)


def plot_grid(by, path, t, parameters=None, cells_pos=None, ncells=None, grid_field_sizes=None, show_plot=True, display=False):
    """
    Plot colony grid by deme sizes or density.

    Parameters:
        by (str)
        path (str)
        t (int)
        parameters (dict)
        cells_pos (numpy.ndarray)
        ncells (int)
        grid_field_sizes (numpy.ndarray)
        show_plot (bool)

    Returns:
        file_name (str)
    """
    if by not in ["size", "density"]:
        print("Error: Invalid option for grid plot")
        exit(1)

    if by == "density":
        if parameters["Experiment"]["colonial"]:
            xs = [x for x, y in cells_pos if x != 0]
            ys = [y for x, y in cells_pos if y != 0]
        else:
            """
            Unlike colony experiment, in box experiment (-1, 0) is a valid point for a cell to rest. 
            However, after initialising the grid with np.nan, these nan values will be interpreted as -1 in mpl.
            To fix this issue, all cells in (-1, 0) will be filtered initially and the difference of 
            ncells and length xs/ys is added again afterwards.
            """
            xs = [x for x, y in cells_pos if not (x == 0 and y == 0)]
            ys = [y for x, y in cells_pos if not (x == 0 and y == 0)]

            while len(xs) < ncells or len(ys) < ncells:
                if len(xs) < ncells:
                    xs.append(0)
                if len(ys) < ncells:
                    ys.append(0)

    N_grid = parameters["Allgemein"]["grid_size"]
    pixel_size_um = parameters["Allgemein"]["deme_size"]
    t = round(t, 1)

    plt.figure(figsize=(8, 8))

    if by == "density":
        plt.hist2d(xs, ys, cmap="viridis", norm=matplotlib.colors.LogNorm(), bins=range(int(N_grid + 1)))

    if by == "size":
        plt.imshow(grid_field_sizes, cmap="RdYlGn_r", norm=matplotlib.colors.LogNorm())

    # Time stamp
    cm_in_um = 10000  # 1 cm in µm
    scale_length_pixels = cm_in_um / pixel_size_um

    if display:
        unit_scalar = 1
        plt.text(5, N_grid-12, f"{t}d", color="white", fontsize=16, fontdict={'family': 'serif'})  # display

        # Scale bar
        plt.text(N_grid-20, N_grid-10, f"{unit_scalar} cm", color="white", va="center")
        plt.gca().add_patch(Rectangle((N_grid-20, N_grid-15), scale_length_pixels*unit_scalar, 2, color="white"))

    else:
        unit_scalar = 2
        plt.text(5, N_grid-30, f"{t}d", color="white", fontsize=64, fontdict={'family': 'serif'})  # print

        # Scale bar
        plt.text(N_grid-42, N_grid-15, f"{unit_scalar} cm", color="white", va="center", fontsize=32)
        plt.gca().add_patch(Rectangle((N_grid-35, N_grid-30), scale_length_pixels*unit_scalar, 5, color="white"))

    # Isolate grid
    plt.gca().set_facecolor("black")
    plt.gcf().set_facecolor('black')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.xticks([])
    plt.yticks([])
    plt.xlim((0, N_grid))
    plt.ylim((0, N_grid))

    file_name = os.path.join(path, f"colony_{by}_t-{t}.png")
    plt.savefig(fname=file_name)

    if show_plot:
        plt.show(block=False)

    plt.close()

    return file_name


def plot_collage(parameters, colony_dev_dens, path):
    """
    Plots a collage of colony density plots for 8 different time steps.

    Parameters:
        parameters (dict)
        colony_dev_dens (np.ndarray)
        path (str)
    """
    N_grid: int = int(parameters["Allgemein"]["grid_size"])
    N_rows: int = 8
    plot_timesteps: list = [1.0, 3.3, 5.0, 6.2, 7.3, 8.3, 8.8, 9.8]
    pixel_size_um: float = parameters["Allgemein"]["deme_size"]
    cm_in_um: int = 10000  # 1 cm in µm
    scale_length_pixels: float = cm_in_um / pixel_size_um

    fig, axs = plt.subplots(1, N_rows, figsize=(80, 10))

    current_ax: int = 0

    for t, stage in enumerate(colony_dev_dens):
        d = time_conversion(t, parameters["Allgemein"]["tmax"], parameters["Allgemein"]["dt"], parameters["Allgemein"]["dayConst"], parameters["Allgemein"]["plot_interval"])
        d = round(d, 1)

        if d not in plot_timesteps:
            continue

        ax = axs[current_ax]
        xs = [x for x, y in stage if x != 0]
        ys = [y for x, y in stage if y != 0]

        h = ax.hist2d(xs, ys, cmap="viridis", norm=matplotlib.colors.LogNorm(), bins=range(N_grid + 1))

        ax.text(5, N_grid - 30, f"{d}d", color="white", fontsize=64, fontdict={'family': 'serif'})  # print

        # Add Scale bar, only for last plot
        if d == plot_timesteps[-1]:
            unit_scalar = 2
            ax.text(N_grid - 42, N_grid - 15, f"{unit_scalar} cm", color="white", va="center", fontsize=32)
            ax.add_patch(Rectangle((N_grid - 35, N_grid - 30), scale_length_pixels * unit_scalar, 5, color="white"))

        ax.set_aspect("equal", "box")  # Make each subplot square
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((0, N_grid))
        ax.set_ylim((0, N_grid))

        current_ax += 1

    cbar_ax = fig.add_axes([0.02, 0.15, 0.2, 1])
    cbar = fig.colorbar(h[3], ax=cbar_ax, orientation="horizontal")  # , fraction=0.1, pad=0.075)
    cbar.set_label("Density [$10^3$ cells/mm$^2$]", color="white", fontsize=48)
    cbar.ax.xaxis.set_tick_params(color="white")
    plt.setp(plt.getp(cbar.ax.axes, "xticklabels"), color="white", fontsize=40)

    # Set the background color of each subplot to black
    for ax in axs:
        ax.set_facecolor("black")

    cbar_ax.set_facecolor("none")
    cbar_ax.set_xticks([])
    cbar_ax.set_yticks([])
    cbar_ax.spines["top"].set_visible(False)
    cbar_ax.spines["right"].set_visible(False)
    cbar_ax.spines["bottom"].set_visible(False)
    cbar_ax.spines["left"].set_visible(False)

    plt.gcf().set_facecolor('black')
    plt.tight_layout()
    plt.savefig(fname=os.path.join(path, "plots", "colony_density_collage.png"))
    plt.show(block=False)


def plot_dist(parameters, path):
    """
    Plots cell size distributions over time. Recreation of Fig. 4D from Puliafito et al. (2012)

    Parameters:
        colony_dev_size (np.ndarray)
        parameters (dict)
        path (str)
    """
    data_path = os.path.join(path, "colony_dev_cells.csv")

    with open(data_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        cell_counts = list()

        for i, row in enumerate(reader):
            if i == 0:
                time_steps = row
                time_steps.pop(0)
                time_steps = [float(x) for x in time_steps]
                continue

            if row:
                row = [float(x) for x in row]
                cell_counts.append(row)

    sns.set_palette("Spectral_r", n_colors=len(cell_counts), desat=0.5)
    ts = np.array([x for x in range(0, int(parameters["Allgemein"]["tmax"]) + 1, int(parameters["Allgemein"]["dt"])) if x % parameters["Allgemein"]["plot_interval"] == 0])
    ts = np.array(range(len(ts))) * parameters["Allgemein"]["dt"] * parameters["Allgemein"]["plot_interval"]

    fig, ax = plt.subplots(figsize=(4*0.9, 3*0.9))
    plt.rc("font", size=7)

    for t, state in enumerate(cell_counts):
        sns.kdeplot(state, ax=ax, linewidth=2, label=(f"t = {round(ts[t], 2)}" if t < 3 or t > len(ts) - 4 else None), zorder=(len(cell_counts) - t))

    # handles, labels = ax.get_legend_handles_labels()

    plt.title("cell area distribution")
    # plt.legend(handles=handles[::-1], frameon=False)
    plt.xlim(0, 1000)
    plt.xlabel("A [$\mu$m$^2$]")
    plt.ylabel("P(A)")
    plt.savefig(os.path.join(path, "plots", "cell_area_distro.pdf"), bbox_inches="tight")
    plt.show(block=False)


def load_csv(file: str) -> tuple:
    """
    Loads experimental data from csv file.

    Parameters:
        file (str): Path to csv file
    """
    x: list = list()
    y: list = list()

    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")

        for row in reader:
            x.append(float(row[0]))
            y.append(float(row[1]))

    return np.array(x), np.array(y)


def load_transposed_csv(file: str) -> dict:
    """
    Loads python simulation results from csv file

    Parameters:
        file (str): Path to csv file
    """
    y: dict = dict()

    with open(file) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")

        for row in reader:

            length: int = int(len(row)) - 1

            if length < 1:
                continue

            name: str = row[0]
            ctr: int = 1

            while name in y:
                name = row[0] + str(ctr)
                ctr = ctr

            y[name] = np.zeros(length)

            for i in range(length):
                y[name][i] = float(row[(i + 1)])

    return y


def load_set(python_sims: list, ca_python_sim_results_path: str) -> list:
    """
    Loads python simulations results.

    Parameters:
        python_sims (list): List of python simulation names
        ca_python_sim_results_path (str): Path to python simulation results
    """
    py_sims: list = list()

    for idx, py_sim in enumerate(python_sims):
        python_sim_path = os.path.join(os.getcwd(), ca_python_sim_results_path, py_sim)
        py_sims.append(
            load_transposed_csv(
                os.path.join(python_sim_path, "colony_dev_cell_count.csv")
            )
        )
        py_sims.append(
            load_transposed_csv(
                os.path.join(python_sim_path, "colony_dev_cell_density.csv")
            )
        )
        py_sims.append(
            load_transposed_csv(
                os.path.join(python_sim_path, "colony_dev_area.csv")
            )
        )
        py_sims.append(
            load_transposed_csv(
                os.path.join(python_sim_path, "colony_dev_median_size.csv")
            )
        )

    return py_sims


def load_exp_set(exp_dir: str) -> tuple:
    """
    Loads experimental data. exp_dir should point to cellularautomata_experimentaldata/Puliafito_2012_01_17 folder.

    Parameters:
        exp_dir (str): Path to experimental data
    """
    densities: list = list()
    files: list = [os.path.join(exp_dir, file) for file in os.listdir(exp_dir) if re.match(r"Pul_2012_01_celldensity(_[A-Za-z]*)?\.csv", file)]

    for file in files:
        x, y = load_csv(file)

        obj = dict()
        obj["x"] = x
        obj["y"] = 10**y
        obj["yi"] = interp1d(x, y, kind="linear", fill_value=(0, 0), bounds_error=False)

        densities.append(obj)

    areas: list = list()
    files: list = [os.path.join(exp_dir, file) for file in os.listdir(exp_dir) if re.match(r"Pul_2012_01_cellarea(_[A-Za-z]*)?\.csv", file)]

    for file in files:
        file = os.path.join(exp_dir, file)
        x, y = load_csv(file)

        obj = dict()
        obj["x"] = x
        obj["y"] = 10**y
        obj["yi"] = interp1d(x, y, kind="linear", fill_value=(0, 0), bounds_error=False)

        areas.append(obj)

    medians: list = list()
    files: list = [os.path.join(exp_dir, file) for file in os.listdir(exp_dir) if re.match(r"Pul_2012_01_cellmedian(_[A-Za-z]*)?\.csv", file)]

    for file in files:
        file = os.path.join(exp_dir, file)
        x, y = load_csv(file)

        obj = dict()
        obj["x"] = x
        obj["y"] = y
        obj["yi"] = interp1d(x, y, kind="linear", fill_value=(0, 0), bounds_error=False)

        medians.append(obj)

    return areas, densities, medians


def parse_args():
    """
    Parses command line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path",
        "-p",
        help="Specify path where multiverse data is located",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--expdata",
        "-e",
        help="Specify path where experimental data is located",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Specify type of generated files (Default: pdf)",
        choices=["pdf", "png", "svg"],
        default="pdf",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--box",
        help="Specify whether simulation was a box experiment",
        action="store_true",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print("Error: Path to simualtion data is invalid")
        exit(1)

    if not os.path.exists(args.expdata):
        print("Error: Path to experimental data is invalid")
        exit(1)

    output_file_type: str = args.output

    exp_data_path: str = args.expdata
    ca_python_sim_results_path: str = args.path
    outputpath: str = os.path.join(ca_python_sim_results_path, "plots")

    # Delete output dir
    if os.path.exists(outputpath):
        shutil.rmtree(outputpath)

    ca_python_sim_ids: list = [
        dir
        for dir in os.listdir(ca_python_sim_results_path)
        if os.path.isdir(os.path.join(ca_python_sim_results_path, dir))
    ]

    # Create output dir
    os.makedirs(outputpath)

    # Load experimental data
    exparea, expdensities, expmedians = load_exp_set(exp_data_path)  # , expdistros

    # Load simulation data
    simulation_data: list = load_set(ca_python_sim_ids, ca_python_sim_results_path)

    # --- Cell Area ---
    plot_multi("area", "$\mu$m$^2$", outputpath, output_file_type, simulation_data, exparea, log=True, boxshift=args.box)

    # --- Cell density ---
    plot_multi("density", "$10^3$ cells/mm$^2$", outputpath, output_file_type, simulation_data, expdensities, log=True, boxshift=args.box)

    # --- Cell Count ---
    plot_multi("count", "cells", outputpath, output_file_type, simulation_data, log=True, boxshift=args.box)

    # --- Cell Median ---
    plot_multi("median size", "$\mu$m$^2$", outputpath, output_file_type, simulation_data, expmedians, log=False, boxshift=args.box)


if __name__ == "__main__":
    main()
