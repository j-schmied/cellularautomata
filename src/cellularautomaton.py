import argparse
import csv
import datetime
import h5py
import json
import numpy as np
import os
import time
import traceback

from numba import njit, prange
from skimage import io

from .utils.conversion import convert_to_numba_dict, time_conversion
from .utils.experimental import aW_exp3
from .utils.parameters import get_parameters
from .utils.plot import plot_grid, plot_linear, plot_collage, plot_dist


@njit(parallel=True)
def calc_colony_density(grid, parameters):
    """
    Calculates the colonies cell density by dividing the total number of cells by the number of demes
    holding at least one cell times the deme_size

    Parameters:
        grid (numpy.ndarray)
        parameters (dict)

    Returns:
        density (float)
    """
    count_cells = 0
    demes_non_zero = 0
    N_grid = int(parameters["grid_size"])

    for x in prange(N_grid):
        for y in prange(N_grid):
            if np.count_nonzero(np.isnan(grid[x][y])) != len(grid[x][y]):
                count_cells += (len(grid[x][y]) - np.count_nonzero(np.isnan(grid[x][y])))
                demes_non_zero += 1

    return count_cells / (demes_non_zero * parameters["deme_size"] * 10 ** -3)


@njit()
def calc_colony_median_size(cells, ncells, cells_pos, parameters):
    """
    Calculates the median of all cell sizes

    Parameters:
        cells (numpy.ndarray)
        ncells (int)
        cells_pos (numpy.ndarray)
        parameters (dict)

    Returns:
        median (float)
    """
    alive_cells_list = list()
    gridsize = parameters["grid_size"]

    for i, cell in enumerate(cells):
        if i == ncells:
            break

        # field_of_view = np.sqrt((cells_pos[i][0] - (gridsize / 2)) ** 2 + (cells_pos[i][1] - (gridsize / 2)) ** 2)

        # if field_of_view <= 7:
        alive_cells_list.append(cell[0])

    return np.median(np.array(alive_cells_list)) if len(alive_cells_list) > 0 else 0


@njit(parallel=True)
def calc_colony_area2(grid_field_sizes, parameters):
    """
    Calculates the colony area by summing up the sizes of all cells

    Parameters:
        grid_field_sizes (numpy.ndarray)
        parameters (dict)

    Returns:
        area (float)
    """
    area = 0
    N_grid = int(parameters["grid_size"])

    for x in prange(N_grid):
        for y in prange(N_grid):
            area += grid_field_sizes[x][y]

    return area


@njit(parallel=True)
def calc_colony_area(grid, parameters):
    """
    Calculates the colony area by multiplying the number of demes holding at least one cell with the deme_size

    Parameters:
        grid (numpy.ndarray)
        parameters (dict)

    Returns:
        area (float)
    """
    demes_non_zero = 0
    N_grid = int(parameters["grid_size"])

    for x in prange(N_grid):
        for y in prange(N_grid):
            if np.count_nonzero(np.isnan(grid[x][y])) != len(grid[x][y]):
                demes_non_zero += 1

    return demes_non_zero * parameters["deme_size"]


@njit()
def calc_ncells(cells):
    """
    Returns the total number of cells

    Parameters:
        cells (numpy.ndarray)

    Returns:
        ncells (int)
    """
    return len(cells) - np.count_nonzero(np.array([np.sum(cell) for cell in cells]) == 0)


@njit()
def calc_asymmetric_proliferation(parameters):
    """
    Draws proliferation factor from normal distribution

    Parameters:
        parameters (dict)

    Returns:
        p (float)
    """
    p_min = parameters["p_cutoff"]
    p_max = 1 - p_min

    p0 = parameters["p0"]
    std = parameters["sigma_p"]

    while True:
        p = np.random.normal(p0, std)
        if p_min <= p <= p_max:
            break

    return p


@njit()
def field_has_enough_space(grid_field_sizes, x, y, new_cell_size, parameters):
    """
    Check if current deme has enough space for new cell

    Parameters:
        grid_field_sizes (numpy.ndarray)
        x (int)
        y (int)
        new_cell_size (float)
        parameters (dict)

    Returns:
        bool
    """
    return (grid_field_sizes[x][y] + new_cell_size) <= parameters["deme_size"]


@njit()
def get_neighbors_with_enough_space(grid_field_sizes, x, y, new_cell_size, parameters):
    # Using Moore neighborhood
    neighbors = [[x, y + 1], [x + 1, y + 1], [x - 1, y], [x + 1, y], [x - 1, y - 1], [x, y - 1], [x + 1, y - 1],
                 [x - 1, y + 1]]
    targets = list()
    i = 0

    for xi, yi in neighbors:
        if xi < 0 or xi > parameters["grid_size"] or yi < 0 or yi > parameters["grid_size"]:
            continue

        if field_has_enough_space(grid_field_sizes, xi, yi, new_cell_size, parameters):
            targets.append(neighbors[i])

        i += 1

    return targets


@njit()
def experiment(grid, grid_field_sizes, cells, cells_pos, event_list, parameters):
    """
    Running Experiment (universal function, can do both, colonial and box experiment)

    Parameters:
        grid (numpy.ndarray)
        grid_field_sizes (numpy.ndarray)
        cells (numpy.ndarray)
        cells_pos (numpy.ndarray)
        event_list (list)
        parameters (dict)

    Returns:
        t (int)
        cells (numpy.ndarray)
        ncell_list (list)
        cells_pos (numpy.ndarray)
        colony_dev_size (list)
        colony_dev_dens (list)
        area_list (list)
        density_list (list)
        median_size_list (list)
    """
    area_list: list = list()  # Track colony area over time
    colony_dev_dens: list = list()  # Track colony density development over time
    colony_dev_size: list = list()  # Track colony size development over time
    colony_dev_cells: list = list()  # Track colony cell size development over time
    density_list: list = list()  # Track colony density over time
    median_size_list: list = list()  # Track median size of colony over time
    ncell_list: list = list()  # Track cell count over time
    nevent_list: list = list()  # Track event count over time

    cell_divisions: list = list()

    t = 0
    dt = parameters["dt"]
    max_field_size = parameters["deme_size"]
    ncells = calc_ncells(cells)
    tmax = parameters["tmax"]

    # Index of last inserted cell
    NEXT_IDX = ncells

    reached_initial_area = False if parameters["area"] < parameters["initial_area"] else True

    while t < tmax:
        if t == 0 and not reached_initial_area:
            print("-" * 40)
            print("Phase 1: Growth")
            print("-" * 40)

        if parameters["area"] >= parameters["initial_area"] and not reached_initial_area:
            reached_initial_area = True
            print("-" * 40)
            print("Finished Phase 1 at t =", t)
            print("Updating tmax:", tmax, "->", tmax + t)
            print("-" * 40)
            print("\n")
            tmax += t
            print("-" * 40)
            print("Phase 2: Colonisation")
            print("-" * 40)

        t += 1

        # Handle cases where tmax is reached before initial area
        # This can happen e.g. with short experiments
        if t == tmax and not reached_initial_area:
            tmax += 1

        if not t % dt == 0:
            continue

        days = t / parameters["dayConst"]

        # Print status twice a day
        if days % 0.5 == 0:
            # Print info message
            if parameters["multiverse"]:
                print("--- Well", int(parameters["well"]), "---")
            print("Time (days):\t\t", np.round(days, 2))
            print("Colony Area:\t\t", parameters["area"], "µm^2")
            print("Number of Cells:\t", ncells, '\n')

        if t % parameters["plot_interval"] == 0 and reached_initial_area:
            # Track colony development
            colony_dev_cells.append(np.copy(cells))
            colony_dev_dens.append(np.copy(cells_pos))
            colony_dev_size.append(np.copy(grid_field_sizes))

        for _ in range(ncells * len(event_list)):
            next_cell = np.random.randint(ncells)
            x, y = cells_pos[next_cell]

            a = cells[next_cell][0]
            a0 = parameters["a0"]
            m = parameters["m"]
            gamma0 = parameters["gamma0"]

            # Calculate Proliferation (see Paper, 2. B)
            event_list[1] = gamma0 * np.exp(-(np.power(a0 / a, m)))

            # Draw Event uniformly random
            event = np.random.randint(len(event_list))  # draw between migration and proliferation
            r = event_list[event]  # get rate of drawn event

            # Calculate if event will happen
            if np.random.uniform(0, 1) <= r * dt:
                # Migration
                if event == 0:
                    # Check neighbors for enough space (using Moore neighborhood)
                    neighbors = [[x, y + 1], [x + 1, y + 1], [x - 1, y], [x + 1, y], [x - 1, y - 1], [x, y - 1],
                                 [x + 1, y - 1], [x - 1,
                                                  y + 1]]  # get_neighbors_with_enough_space(grid_field_sizes, x, y, a, parameters)

                    # If no neighbor has enough space, skip migration
                    if len(neighbors) == 0:
                        continue

                    while True:
                        # Choose random neighbor to migrate to
                        target = np.random.randint(len(neighbors))

                        # Get target coordinates
                        x_new, y_new = neighbors[target]

                        # Check if drawn neighbor is inside the grid (especially important for box experiment)
                        if 0 <= x_new < parameters["grid_size"] and 0 <= y_new < parameters["grid_size"]:
                            break

                    if field_has_enough_space(grid_field_sizes, x_new, y_new, a, parameters):
                        app_success = False
                        rem_success = False

                        # Move cell to target field and increase field size
                        for i, cell in enumerate(grid[x_new][y_new]):
                            if np.isnan(cell):
                                grid[x_new][y_new][i] = float(next_cell)
                                break

                        if next_cell in grid[x_new][y_new]:
                            app_success = True
                            grid_field_sizes[x_new][y_new] += a

                        # Remove cell from initial field and reduce field size
                        for i, cell in enumerate(grid[x][y]):
                            if np.isnan(cell):
                                continue

                            if int(cell) == next_cell:
                                grid[x][y][i] = np.nan
                                break
                        if next_cell not in grid[x][y]:
                            rem_success = True
                            grid_field_sizes[x][y] -= a

                        # Update position in cells pos array
                        if app_success and rem_success:
                            cells_pos[next_cell][0] = x_new
                            cells_pos[next_cell][1] = y_new

                        # Reset position for further processing
                        x, y = x_new, y_new

                        nevent_list.append(event)

                # Proliferation
                if event == 1:
                    if parameters["symmetric"]:
                        ds = a * 0.5

                        # Create new cell
                        cells[NEXT_IDX][0] = ds

                    else:  # Using asymmetric proliferation
                        asymm_factor = calc_asymmetric_proliferation(parameters)
                        ds = a * asymm_factor

                        # Create new cell
                        cells[NEXT_IDX][0] = a - ds

                    # Shrink cell size of initial cell
                    cells[next_cell][0] -= ds

                    # Set time of last division
                    cells[next_cell][1] = t

                    # Append cell to field
                    for i, cell in enumerate(grid[x][y]):
                        if np.isnan(cell):
                            grid[x][y][i] = float(NEXT_IDX)
                            break

                    cell_divisions.append((next_cell, t, cells[next_cell][0] + ds))
                    cell_divisions.append((NEXT_IDX, t, ds))

                    # If proliferation succeeded
                    if NEXT_IDX in grid[x][y]:
                        cells_pos[NEXT_IDX][0] = x
                        cells_pos[NEXT_IDX][1] = y

                        if NEXT_IDX + 1 < parameters["max_cells"]:
                            NEXT_IDX += 1

                    nevent_list.append(event)

                # Growth
                if event == 2:
                    free_space = max_field_size - grid_field_sizes[x][y]

                    Amax_factor = 1.0
                    da = parameters["alphaw"] * a * (1 - (a / (Amax_factor * parameters["aM"])))

                    # 0 <= da <= free_space
                    da = max(0, min(da, free_space))  # , parameters["aM"] - a)

                    # Update cell size
                    cells[next_cell][0] += da
                    grid_field_sizes[x][y] += da

                    nevent_list.append(event)

        parameters["area"] = calc_colony_area(grid, parameters)

        if reached_initial_area:
            median = calc_colony_median_size(cells, ncells, cells_pos, parameters)

            # Track progress
            ncell_list.append(ncells)
            density_list.append(calc_colony_density(grid, parameters))
            area_list.append(parameters["area"])
            median_size_list.append(median)

        ncells = calc_ncells(cells)

    nevents = len(nevent_list)

    if t % parameters["plot_interval"] != 0:
        # Track colony development
        colony_dev_cells.append(np.copy(cells))
        colony_dev_dens.append(np.copy(cells_pos))
        colony_dev_size.append(np.copy(grid_field_sizes))

    print("\n")
    print("-" * 40)
    print("Simulation Results")
    print("-" * 40)
    print(f"Number of Events:\t {nevents}")

    return t, cells, ncell_list, cells_pos, colony_dev_size, colony_dev_dens, area_list, density_list, median_size_list, colony_dev_cells, cell_divisions


def init_ca(parameters):
    """
    Initialize Cellular Automata

    Parameter:
        parameters (dict)

    Returns:
        grid (numpy.ndarray)
        cells (numpy.ndarray)
        cells_pos (numpy.ndarray)
        parameters (dict)
        event_list (numpy.ndarray)
        grid_field_sizes (numpy.ndarray)
    """
    NEXT_IDX = 0

    N_grid = int(parameters["Allgemein"]["grid_size"])  # grid size per axis

    max_cells_per_field = int(parameters["Allgemein"]["d0"])
    max_cells = N_grid * N_grid * max_cells_per_field

    parameters["Allgemein"]["max_cells"] = float(max_cells)

    N_cells = np.power(N_grid, 2) * max_cells_per_field  # cell count for two dimensions

    # 2D Grid, N_grid x N_grid -> [i] = (x, y, [cells: indices pointing to cells -> empty by default])
    grid = np.empty((N_grid, N_grid, max_cells_per_field))
    grid.fill(np.nan)

    grid_field_sizes = np.zeros((N_grid, N_grid), dtype=float)

    # Grid indices of cell (x, y)
    cells_pos = np.zeros((N_cells, 2), dtype=int)

    # Attributes (size, time of last division)
    cells = np.zeros((N_cells, 2), dtype=float)

    area = 0

    # Initialize grid with on max size cell at center
    if parameters["Experiment"]["colonial"]:
        center_x = int(N_grid / 2)
        center_y = int(N_grid / 2)

        cell_size = parameters["Allgemein"]["aM"]

        grid[center_x][center_y][0] = float(NEXT_IDX)
        grid_field_sizes[center_x][center_y] += cell_size
        cells[NEXT_IDX][0] = cell_size
        cells_pos[NEXT_IDX][0] = center_x
        cells_pos[NEXT_IDX][1] = center_y
        area += cell_size
        NEXT_IDX += 1

    if parameters["Experiment"]["box"]:
        place_cell = False
        for x in range(N_grid):
            for y in range(N_grid):
                place_cell = not place_cell

                if not place_cell:
                    continue

                cell_size = parameters["Allgemein"]["aM"]

                grid[x][y][0] = float(NEXT_IDX)
                grid_field_sizes[x][y] += cell_size
                cells[NEXT_IDX][0] = cell_size
                cells_pos[NEXT_IDX][0] = x
                cells_pos[NEXT_IDX][1] = y
                area += cell_size
                NEXT_IDX += 1

    if not parameters["Experiment"]["multiverse"] or parameters["Experiment"]["multiverse"] and parameters["Allgemein"][
        "well"] == 1:
        print("-" * 40)

        if parameters["Experiment"]["colonial"]:
            print("Running Colony Experiment")

        if parameters["Experiment"]["box"]:
            print("Running Box Experiment")

        print("-" * 40)

    parameters["Allgemein"]["area"] = area
    cells_n0 = len(cells) - np.count_nonzero(np.array([np.sum(cell) for cell in cells]) == 0)

    print("Parameters")
    print("-" * 20)
    print(f"Initial area:\t\t {round(area, 2)} µm^2")
    print(f"Cells at t0:\t\t {cells_n0}")

    # Events: [Migration, Proliferate*, Growth]
    # * calculated at runtime
    event_list = np.array([parameters["Allgemein"]["mu"], 0.0, (1.0 / parameters["Allgemein"]["dt"])])

    return grid, cells, cells_pos, parameters, event_list, grid_field_sizes


def run_ca(cells, cells_pos, event_list, grid, grid_field_sizes, parameters):
    """
    Run Cellular Automata

    Parameters:
        cells (numpy.ndarray)
        cells_pos (numpy.ndarray)
        event_list (numpy.ndarray)
        grid (numpy.ndarray)
        grid_field_sizes (numpy.ndarray)
        parameters (dict)

    Returns:
        0
    """
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

    # Construct output path based on parameters
    path = parameters["Export"]["output_directory"]
    path += "/colonial/" if parameters["Experiment"]["colonial"] else "/box/"
    path += f"multiverse_{int(parameters['Allgemein']['multiverse_id'])}/" if parameters["Experiment"][
        "multiverse"] else ""
    path += "single_" if not parameters["Experiment"]["multiverse"] else ""
    path += str(int(parameters['Allgemein']['multiverse_id'])) if parameters["Experiment"]["multiverse"] else str(
        timestamp)
    path += f"_well{int(parameters['Allgemein']['well']):03}" if parameters["Experiment"]["multiverse"] else ""

    if not os.path.exists(path):
        os.makedirs(path)

    if parameters["Experiment"]["plot"] or parameters["Experiment"]["plothist"]:
        plot_dir = "plots"

        if not os.path.exists(os.path.join(path, plot_dir)):
            os.makedirs(os.path.join(path, plot_dir))

    if parameters["Experiment"]["multiverse"]:
        print(f"Well:\t\t\t {parameters['Allgemein']['well']:03}")
    print(f"Seed:\t\t\t {parameters['Allgemein']['seed']}")
    print(f"Grid Size:\t\t {int(parameters['Allgemein']['grid_size'])}x{int(parameters['Allgemein']['grid_size'])}")
    print(f"Steps:\t\t\t {int(parameters['Allgemein']['tmax'] / parameters['Allgemein']['dt'])}")
    print(f"Step length (dt):\t {parameters['Allgemein']['dt']} min")
    print(f"Speed:\t\t\t {parameters['Allgemein']['vc']} µm/h")
    print(f"Proliferation mode:\t {'symmetric' if parameters['Allgemein']['symmetric'] else 'asymmetric'}")
    print("-" * 40, "\n")

    # --- Run Experiment ---
    custom_parameters = dict(parameters["Allgemein"], **parameters["Experiment"])
    custom_parameters = convert_to_numba_dict(custom_parameters)

    t0 = time.time()
    t, cells, ncell_list, cells_pos, colony_dev_size, colony_dev_dens, area_list, density_list, median_size_list, colony_dev_cells, cell_divisions = experiment(
        grid, grid_field_sizes, cells, cells_pos, event_list, custom_parameters)
    runtime = time.time() - t0

    parameters["Results"]["ComputationTime"] = runtime

    print(f'Computation Time:\t {np.round(runtime, 2)} seconds')
    print(f'Simulation Time:\t {np.round(t / parameters["Allgemein"]["dayConst"], 2)} days')

    print(f'Number of Cells:\t {ncell_list[-1]}')
    parameters["Results"]["FinalCellCount"] = ncell_list[-1]

    print(f'Final Area:\t\t {round(area_list[-1], 2)} µm^2\n\n')
    parameters["Results"]["FinalArea"] = area_list[-1]

    with open(f"{path}/parameters.json", 'w') as f:
        json.dump(parameters, f)

    t0 = time.time()

    print("Exporting data...", end="\r", flush=True)

    image_files = list()

    np.save(f"{path}/cells.npy", colony_dev_cells)
    np.save(f"{path}/cells_pos.npy", colony_dev_dens)

    # --- Export images, colony area, cell count and density to hdf ---
    if parameters["Experiment"]["exporthdf"]:
        # Only use values created at plot_interval
        density_list_filtered = [val for i, val in enumerate(density_list) if
                                 i * parameters["Allgemein"]["dt"] % parameters["Allgemein"]["plot_interval"] == 0]
        area_list_filtered = [val for i, val in enumerate(area_list) if
                              i * parameters["Allgemein"]["dt"] % parameters["Allgemein"]["plot_interval"] == 0]
        ncell_list_filtered = [val for i, val in enumerate(ncell_list) if
                               i * parameters["Allgemein"]["dt"] % parameters["Allgemein"]["plot_interval"] == 0]

        hdf_file_name = path + "/"
        hdf_file_name += "Colonial_Experiment" if parameters["Experiment"]["colonial"] else "Box_Experiment"
        hdf_file_name += "_" + timestamp
        hdf_file_name += ".hdf5"

        with h5py.File(hdf_file_name, "w") as _hdf5_file:
            dataset_group = _hdf5_file.create_group("Colony")

            # Image group
            images_dataset = dataset_group.create_dataset("Image", (len(ncell_list_filtered), 600, 800, 4),
                                                          dtype="uint8")

            for t, file in enumerate(image_files):
                image = io.imread(file)
                images_dataset[t] = image  # file

            # Colony area group
            dataset_group.create_dataset("ColonyArea", data=area_list_filtered)

            # Cell count group
            dataset_group.create_dataset("CellCount", data=ncell_list_filtered)

            # Colony density group
            dataset_group.create_dataset("ColonyDensity", data=density_list_filtered)

    # --- Export colony development to csv ---
    if parameters["Experiment"]["exportcsv"]:
        tmax_by_dt = parameters["Allgemein"]["tmax"] / parameters["Allgemein"]["dt"]

        transformIterationToDays = lambda t: np.round(
            t / parameters["Allgemein"]["dayConst"] * parameters["Allgemein"]["dt"], 2)

        # Using ncell_list generally is okay because all list have the same length
        time_list = np.arange(0, tmax_by_dt + 1, parameters["Allgemein"]["plot_interval"])

        # If last time step is not evenly dividable by plot_interval, manually append last time step
        if time_list[-1] != tmax_by_dt:
            time_list = np.concatenate((time_list, [int(tmax_by_dt - 1)]))

        time_list_to_days = np.concatenate((["time"], transformIterationToDays(time_list)))

        with open(f"{path}/colony_dev_cell_count.csv", "a") as cc_file:
            countcsv = csv.writer(cc_file)

            filtered_cellCount_list = [ncell_list[int(i if i < tmax_by_dt else i - 1)] for i in time_list]
            filtered_cellCount_list.insert(0, "Cell count")

            countcsv.writerow(time_list_to_days)
            countcsv.writerow(filtered_cellCount_list)

        with open(f"{path}/colony_dev_cell_density.csv", "a") as cdens_file:
            denscsv = csv.writer(cdens_file)

            filtered_cellDensity_list = [density_list[int(i if i < tmax_by_dt else i - 1)] for i in time_list]
            filtered_cellDensity_list.insert(0, "Cell density")

            denscsv.writerow(time_list_to_days)
            denscsv.writerow(filtered_cellDensity_list)

        with open(f"{path}/colony_dev_median_size.csv", "a") as med_size_file:
            medsizecsv = csv.writer(med_size_file)

            filtered_median_size_list = [median_size_list[int(i if i < tmax_by_dt else i - 1)] for i in time_list]
            filtered_median_size_list.insert(0, "Cell median size")

            medsizecsv.writerow(time_list_to_days)
            medsizecsv.writerow(filtered_median_size_list)

        with open(f"{path}/colony_dev_area.csv", "a") as carea_file:
            areacsv = csv.writer(carea_file)

            filtered_cellArea_list = [area_list[int(i if i < tmax_by_dt else i - 1)] for i in time_list]
            filtered_cellArea_list.insert(0, "Cell area")

            areacsv.writerow(time_list_to_days)
            areacsv.writerow(filtered_cellArea_list)

        # Dump entire colony_dev_cells array to csv, filter out cells with size 0
        with open(f"{path}/colony_dev_cells.csv", "a") as grid_file:
            gridcsv = csv.writer(grid_file)

            gridcsv.writerow(time_list_to_days)

            for state in colony_dev_cells:
                gridcsv.writerow([size for size, tld in state if size != 0])

        # Dump cell division times
        np.save(f"{path}/cell_divisions.npy", cell_divisions)

    # --- Plot ---
    # Plot final stage
    if parameters["Experiment"]["plot"]:
        # Plot final stage
        # plot_grid(by="size", parameters=parameters, grid_field_sizes=grid_field_sizes, path=os.path.join(path, plot_dir), t=(parameters["Allgemein"]["tmax"] / parameters["Allgemein"]["dayConst"]))
        plot_grid(by="density", parameters=parameters, cells_pos=cells_pos, path=os.path.join(path, plot_dir),
                  t=(parameters["Allgemein"]["tmax"] / parameters["Allgemein"]["dayConst"]), ncells=ncell_list[-1],
                  display=True)
        plot_linear(ncell_list, os.path.join(path, plot_dir), name="cell_count", parameters=parameters,
                    exp_data_path=parameters["Export"]["expdata"])
        plot_linear(area_list, os.path.join(path, plot_dir), name="colony_area", parameters=parameters, log=True,
                    exp_data_path=parameters["Export"]["expdata"])
        plot_linear(median_size_list, os.path.join(path, plot_dir), name="cell_size_median", parameters=parameters,
                    exp_data_path=parameters["Export"]["expdata"])
        plot_linear(density_list, os.path.join(path, plot_dir), name="colony_density", parameters=parameters, log=True,
                    exp_data_path=parameters["Export"]["expdata"])
        plot_dist(parameters, path)

    # Plot colony development, but do not display them, only save
    if parameters["Experiment"]["plothist"]:
        # for t, stage in enumerate(colony_dev_size):
        #     d = time_conversion(t, parameters["Allgemein"]["tmax"], parameters["Allgemein"]["dt"], parameters["Allgemein"]["dayConst"], parameters["Allgemein"]["plot_interval"])
        #     plot_grid(by="size", parameters=parameters grid_field_sizes=stage, path=os.path.join(path, plot_dir), t=d, show_plot=False)

        plot_collage(parameters, colony_dev_dens, path)

        for t, stage in enumerate(colony_dev_dens):
            d = time_conversion(t, parameters["Allgemein"]["tmax"], parameters["Allgemein"]["dt"],
                                parameters["Allgemein"]["dayConst"], parameters["Allgemein"]["plot_interval"])
            file_name = plot_grid(by="density", parameters=parameters, cells_pos=stage,
                                  path=os.path.join(path, plot_dir), t=d, show_plot=False, ncells=ncell_list[
                    max(0, min(int(t * parameters["Allgemein"]["plot_interval"]), len(ncell_list)) - 1)])
            image_files.append(file_name)

    print(f"Exported data in {round(time.time() - t0, 2)} seconds")

    return 0


def build_parameters_dict(args):
    """
    Merges command line and config file parameters,
    re-calculates values to match units

    Parameters:
        args (Namespace)

    Returns:
        parameters (dict)
    """
    # Parameters parsed from command line arguments
    parameters = get_parameters(args.colonial)

    # Numpy random generator seed
    parameters["Allgemein"]["seed"] = args.seed if args.seed != -1 else np.random.randint(low=0, high=np.power(2, 24))

    # Number of demes per edge (= M, see Paper 2.)
    parameters["Allgemein"]["grid_size"] = args.gridsize if args.gridsize > 0.0 else (256.0 if args.colonial else 12.0)

    # Monte-Carlo step width
    # FIXME: PEP Unitless-ness
    parameters["Allgemein"]["dt"] = parameters["Allgemein"]["dt"] * 60  # Convert to min

    # Controls units based on different time steps
    # FIXME: PEP Unitless-ness
    parameters["Allgemein"]["dayConst"] = 1440  # steps/day
    parameters["Allgemein"]["hourConst"] = 60 / parameters["Allgemein"]["dt"]  # steps/hour

    # Max simulation steps
    # FIXME: PEP Unitless-ness
    parameters["Allgemein"]["tmax"] = parameters["Allgemein"]["tmax"] * 60  # Convert to min

    # Max possible area based on grid size
    parameters["Allgemein"]["areamax"] = parameters["Allgemein"]["grid_size"] ** 2 * parameters["Allgemein"][
        "deme_size"]  # µm^2

    # Toggles the use of Symmetric/Asymmetric proliferation
    parameters["Allgemein"]["symmetric"] = args.symmetric  # bool

    # Rate at which a cell migrates to a neighboring cell (see Paper 2. C)
    # FIXME: PEP Unitless-ness
    L = np.sqrt(parameters["Allgemein"]["deme_size"])  # µm
    parameters["Allgemein"]["mu"] = parameters["Allgemein"]["vc"] / L / 60  # float

    # Based on dt, re-calculate parameters
    # FIXME: PEP Unitless-ness
    parameters["Allgemein"]["gamma0"] = parameters["Allgemein"]["gamma0"] / parameters["Allgemein"]["dayConst"]

    if args.experimental:
        # FIXME: PEP Unitless-ness
        parameters["Allgemein"]["alphaw"] = aW_exp3(parameters["Allgemein"]["vc"]) / parameters["Allgemein"][
            "hourConst"]
    else:
        if parameters["Allgemein"]["vc"] == 50:
            # FIXME: PEP Unitless-ness
            parameters["Allgemein"]["alphaw"] = parameters["Allgemein"]["alphaw_50"] / parameters["Allgemein"][
                "hourConst"]

        if parameters["Allgemein"]["vc"] == 100:
            # FIXME: PEP Unitless-ness
            parameters["Allgemein"]["alphaw"] = parameters["Allgemein"]["alphaw_100"] / parameters["Allgemein"][
                "hourConst"]

    if args.multiverse:
        parameters["Allgemein"]["well"] = int(args.well) + 1
        parameters["Allgemein"]["multiverse_id"] = args.multiverse
        parameters["Experiment"]["multiverse"] = True
    else:
        parameters["Experiment"]["multiverse"] = False

    parameters["Experiment"]["plot"] = args.plot
    parameters["Experiment"]["plothist"] = args.plothist or args.exporthdf
    parameters["Experiment"]["colonial"] = args.colonial
    parameters["Experiment"]["box"] = args.box
    parameters["Experiment"]["exportcsv"] = args.exportcsv
    parameters["Experiment"]["exporthdf"] = args.exporthdf
    parameters["Export"]["expdata"] = args.expdata if args.expdata != "" else None
    parameters["Export"]["output_directory"] = args.output if os.path.exists(args.output) else "output"

    return parameters


def parse_args():
    """
    Parses command line arguments

    Returns:
        args (Namespace)
    """
    parser = argparse.ArgumentParser()

    # Functional
    parser.add_argument("--box", help="Run box experiment", action="store_true", default=False)
    parser.add_argument("--colonial", help="Run colonial experiment", action="store_true", default=False)
    parser.add_argument("--gridsize", help="Grid edge size (default: 256 for Colonial and 12 for Box experiment)",
                        type=float, default=0.0)
    parser.add_argument("--seed", "-s", help="Simulation seed", type=int, default=-1)
    parser.add_argument("--symmetric", help="Toggle symmetric/asymmetric proliferation (asymmetric by default)",
                        action="store_true", default=False)
    parser.add_argument("--experimental", help="Enable experimental features", action="store_true", default=False)
    parser.add_argument("--multiverse", help="Specify multiverse id of current simulation", type=float)
    parser.add_argument("--well", help="Specify well index of current simulation", type=float)

    # Data export
    parser.add_argument("--expdata", help="Path to experimental data", type=str, default="")
    parser.add_argument("--output", "-o", help="Specify output directory", type=str, default="")
    parser.add_argument("--exportcsv", help="Export colony development history to csv", action="store_true",
                        default=False)
    parser.add_argument("--exporthdf", help="Export colony development history to hdf5", action="store_true",
                        default=False)
    parser.add_argument("--plot", help="Create and save plots during experiment (final stage only)",
                        action="store_true", default=False)
    parser.add_argument("--plothist", help="Save plots during colony development", action="store_true", default=False)

    parsed_args = parser.parse_args()

    # Combine parsed args with parameters from json
    parsed_args = build_parameters_dict(parsed_args)

    return parsed_args


def main():
    parameters = parse_args()

    if parameters["Experiment"]["box"] and parameters["Experiment"]["colonial"]:
        print("Error: only one experiment can be run at a time.")
        return 1

    # Set seed for numpy random generator
    np.random.seed(parameters["Allgemein"]["seed"])

    try:
        grid, cells, cells_pos, parameters, event_list, grid_field_sizes = init_ca(parameters)
        run_ca(cells, cells_pos, event_list, grid, grid_field_sizes, parameters)
    except Exception as _:
        traceback.print_exc()
        return -1
    except KeyboardInterrupt:
        print("Experiment interrupted by User.")

    return 0


if __name__ == "__main__":
    main()
