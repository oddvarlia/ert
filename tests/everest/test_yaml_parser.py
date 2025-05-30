import os
from pathlib import Path
from textwrap import dedent

import pytest
from ruamel.yaml import YAML

from ert.config.parsing import ConfigKeys as ErtConfigKeys
from everest.config import EverestConfig
from everest.simulator.everest_to_ert import (
    everest_to_ert_config_dict,
)
from tests.everest.utils import MockParser, skipif_no_everest_models


@pytest.mark.parametrize("random_seed", [None, 1234])
def test_random_seed(tmp_path, monkeypatch, random_seed):
    monkeypatch.chdir(tmp_path)
    config = {"model": {"realizations": [0]}}
    if random_seed:
        config["environment"] = {"random_seed": random_seed}
    ever_config = EverestConfig.with_defaults(**config)
    dictionary = everest_to_ert_config_dict(ever_config)

    if random_seed is None:
        assert ever_config.environment.random_seed > 0
        assert dictionary[ErtConfigKeys.RANDOM_SEED] > 0
    else:
        assert ever_config.environment.random_seed == random_seed
        assert dictionary[ErtConfigKeys.RANDOM_SEED] == random_seed


def test_read_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("config.yml").write_text(
        dedent("""
    model: {"realizations": [0]}
    controls:
      -
        name: my_control
        type: well_control
        min: 0
        max: 0.1
        variables:
          - { name: test, initial_guess: 0.1 }
    objective_functions:
      - {name: my_objective}
    """),
        encoding="utf-8",
    )
    everest_config = EverestConfig.load_file("config.yml")
    keys = ["config_path", "controls", "model", "objective_functions"]
    assert sorted(everest_config.to_dict().keys()) == sorted(keys)

    exp_dir, exp_fn = os.path.split(os.path.realpath("config.yml"))
    assert exp_dir == everest_config.config_directory
    assert exp_fn == everest_config.config_file


def test_valid_config_file(copy_test_data_to_tmp, monkeypatch):
    monkeypatch.chdir("valid_config_file")
    parser = MockParser()

    config = EverestConfig.load_file_with_argparser(
        "valid_yaml_config.yml", parser=parser
    )
    # Check no error is generated when loading a valid config file
    assert parser.get_error() is None

    yaml = YAML(typ="safe", pure=True)
    with open("test", "w", encoding="utf-8") as f:
        yaml.dump(config.to_dict(), f)

    assert EverestConfig.load_file_with_argparser("test", parser=parser) is not None

    config_dict = config.to_dict()
    del config_dict["objective_functions"]
    yaml = YAML(typ="safe", pure=True)
    with open("test", "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f)

    # Check a valid config file is also linted
    assert EverestConfig.load_file_with_argparser("test", parser=parser) is None
    assert "objective_functions" in parser.get_error()
    assert "Field required" in parser.get_error()

    # Check a invalid yaml errors are reported to the parser
    assert (
        EverestConfig.load_file_with_argparser(
            config_path="invalid_yaml_config.yml", parser=parser
        )
        is None
    )

    assert (
        "The config file: <invalid_yaml_config.yml> contains invalid YAML syntax:"
        in parser.get_error()
    )
    assert "could not find expected ':'" in parser.get_error()


@pytest.mark.skip_mac_ci
@skipif_no_everest_models
@pytest.mark.everest_models_test
def test_valid_forward_model_config_files(copy_test_data_to_tmp, monkeypatch):
    monkeypatch.chdir("valid_config_file/forward_models")
    parser = MockParser()
    EverestConfig.load_file_with_argparser(
        "valid_config_maintained_forward_models.yml", parser=parser
    )

    assert parser.get_error() is None


@skipif_no_everest_models
@pytest.mark.everest_models_test
@pytest.mark.skip_mac_ci
def test_invalid_forward_model_config_files(copy_test_data_to_tmp, monkeypatch):
    monkeypatch.chdir("valid_config_file/forward_models")
    parser = MockParser()
    next((Path.cwd() / "input" / "templates").glob("*")).unlink()
    EverestConfig.load_file_with_argparser(
        "valid_config_maintained_forward_models.yml", parser=parser
    )
    template_config_path = "configs/template_config.yml"
    config_file = "valid_config_maintained_forward_models.yml"
    template_path = "./templates/wellopen.jinja"
    assert f"""Loading config file <{config_file}> failed with:
Found 1 validation error:


    * Value error, job = 'add_templates'\t-c/--config = {template_config_path}
\t\ttemplates: {template_path} -> Path does not point to a file (type=value_error)""" in parser.get_error()  # noqa: E501
