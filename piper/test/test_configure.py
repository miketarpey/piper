import pytest
import json
import pandas as pd
import numpy as np
from piper.configure import get_config
from piper.configure import get_dict_df


@pytest.fixture
def get_complete_config():

    config = {"meta": {"project": "Datagon CRP"},
              "folders": {"report": "reports", "sql": "sql",
                          "input": "inputs", "output": "outputs"},
              "variables": {"schema": "eudta", "schema_ctl": "eudta_ctl"}}

    return config


@pytest.fixture
def get_missing_project():

    config = {"folders": {"report": "reports", "sql": "sql",
                          "input": "inputs", "output": "outputs"},
              "variables": {"schema": "", "schema_ctl": ""}}

    return config


def test_get_config_file_name_invalid_returns_NONE():

    assert get_config(file_name='invalid', info=False) is None


def test_get_config_valid_with_info(get_complete_config):

    file_name = 'piper/temp/test_config.json'

    # Convert dictionary to json format, write to test temp config file
    json_data = json.dumps(get_complete_config)
    with open(file_name, 'w') as f:
        f.write(json_data)

    # Read the data back and load to dict object
    with open(file_name, 'r') as f:
        expected = json.load(f)

    actual = get_config(file_name, info=True)

    assert expected == actual


def test_get_meta_folders(get_complete_config):

    expected = {"report": "reports", "sql": "sql",
                "input": "inputs", "output": "outputs"}
    actual = get_complete_config['folders']

    assert expected == actual


def test_get_meta_project_name_missing(get_missing_project):

    with pytest.raises(Exception):
        assert get_missing_project['meta']['project'] is None


def test_get_meta_project_name_valid(get_complete_config):

    expected = "Datagon CRP"
    actual = get_complete_config['meta']['project']

    assert expected == actual


def test_get_meta_variables_valid(get_complete_config):

    expected = {"schema": "eudta", "schema_ctl": "eudta_ctl"}
    actual = get_complete_config['variables']

    assert expected == actual


def test_get_dict_df(get_complete_config):

    expected = pd.DataFrame.from_dict(get_complete_config, orient='index')
    expected = expected.loc['meta', 'project']
    actual = get_dict_df(get_complete_config).loc['meta', 'project']
    assert expected == actual


def test_get_dict_df_with_colnames(get_complete_config):

    expected = pd.DataFrame.from_dict(get_complete_config, orient='index')
    expected = expected.loc['folders', 'report']
    actual = (get_dict_df(get_complete_config, colnames=['report'])
              .loc['folders', 'report'])
    assert expected == actual
