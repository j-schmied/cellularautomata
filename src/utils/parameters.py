def get_parameters(is_colonial_experiment: bool):
    parameter_set = "steffen"  # Choose: "steffen", "paul"

    parameters = {
        "Allgemein": {
            "dt": 0.1,  # h
            # Set default experiment duration to 10 days for colonial and 15 days for box experiments
            "tmax": 240.0 if is_colonial_experiment else 360,  # h
            # Area of a single grid point (x, y) (= deme/node, L^2)
            "deme_size": 990.0,  # µm^2
            "plot_interval": 100.0,  # h
            # Initial area (threshold for phase transition)
            "initial_area": 17552.0,  # µm^2
            # Maximum number of division events per unit time of an individual cell
            "gamma0": 1.3 if parameter_set == "steffen" else 1.62,
            # Turning point of the proliferation rate curve at which a
            # single proliferation event is e^-1 ≈ 36% as likely as gamma0
            "a0": 80.0 if parameter_set == "steffen" else 105.29,  # µm^2
            # Controls the slope smoothing the curve for smaller values
            "m": 3.0 if parameter_set == "steffen" else 1.697,
            # Maximum number of cells per unit area
            "d0": 33.0,  # cells
            # Maximum size an individual cell can reach
            "aM": 990.0,  # µm^2
            # Movement speed of cells
            "vc": 100.0,  # µm/h
            # Maximum growth rate of an individual cell per unit area
            # Separate values for different movement speeds
            # "WR-aw-30": 0.095,  # h^-1
            "alphaw_50": 0.054 if parameter_set == "steffen" else 0.065,  # h^-1
            "alphaw_100": 0.054 if parameter_set == "steffen" else 0.054,  # h^-1
            # Parameters for asymmetric proliferation
            "p0": 0.5,
            "sigma_p": 0.125,
            "p_cutoff": 0.25
        },
        "Experiment": {},
        "Export": {},
        "Results": {}
    }
    return parameters
