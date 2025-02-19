import ert3


def clean(workspace, experiment_names, clean_all):
    if clean_all:
        non_existant = []
    else:
        non_existant = [
            name
            for name in experiment_names
            if name not in ert3.storage.get_experiment_names(workspace=workspace)
        ]

    ert3.engine.clean(workspace, experiment_names, clean_all)

    if non_existant:
        print("Following experiment(s) did not exist:")
        for name in non_existant:
            print(f"    {name}")
        print("Perhaps you mistyped an experiment name?")
