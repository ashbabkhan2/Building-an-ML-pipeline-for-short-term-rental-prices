"""
This module used as a storage for the test_data.py module this module 
contain lots of fixtures that will be used in the test_data.py module
and also function which read the parameters pass from MLProject. 

Date: 06/March/2023
Developer: ashbab khan

"""

# Importing packages
import pytest
import pandas as pd
import wandb


def pytest_addoption(parser):

    """
    this function used to get parameters coming from MLProject
    we use parser.adoption() to store the parameters our other functions can
    easily access this parameters using the request.config.option.parameter_name
    
    """
    
    parser.addoption("--csv", action="store")
    parser.addoption("--ref", action="store")
    parser.addoption("--kl_threshold", action="store")
    parser.addoption("--min_price", action="store")
    parser.addoption("--max_price", action="store")

#============================================ data fixture =====================================

@pytest.fixture(scope='session')
def data(request):

    """
    this function will fetch the latest clean_data artifact and then read 
    the artifact using pd.read_csv() and then return the dataframe

    Input: request 
    Output: df (dataframe)
    """
    
    # Initializing the run
    run = wandb.init(job_type="data_tests", resume=True)

    # fetching the artifact path
    data_path = run.use_artifact(request.config.option.csv).file()

    # failing the test if we doesn't pass the csv parameter
    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df


#============================================ ref_data fixture =====================================


@pytest.fixture(scope='session')
def ref_data(request):

    """
    this function will fetch the reference clean data artifact and 
    the read it using pd.read_csv() and then return this dataframe

    Input: request
    Output: df (dataframe)
    
    """
    
    # Initializing the run 
    run = wandb.init(job_type="data_tests", resume=True)

    # Fetching the artifact path
    data_path = run.use_artifact(request.config.option.ref).file()

    if data_path is None:
        pytest.fail("You must provide the --ref option on the command line")

    # reading the artifact
    df = pd.read_csv(data_path)

    return df


#============================================ kl_threshold fixture =====================================


@pytest.fixture(scope='session')
def kl_threshold(request):

    """
    
    this function will return the kl_threshold value 

    Input: request
    Output: kl_threshold (float)

    """
    # getting the kl_threshold value from the parameters
    kl_threshold = request.config.option.kl_threshold

    if kl_threshold is None:
        pytest.fail("You must provide a threshold for the KL test")

    return float(kl_threshold)


#============================================ min_price fixture =====================================


@pytest.fixture(scope='session')
def min_price(request):

    """
    this function will return the min_price variable

    Input: request
    Output: min_price (float)
    
    """
    
    # getting the min_price parameter 
    min_price = request.config.option.min_price

    if min_price is None:
        pytest.fail("You must provide min_price")

    return float(min_price)


#============================================ max_price fixture =====================================


@pytest.fixture(scope='session')
def max_price(request):

    """
    this function will return the max_price value

    Input: request
    Output: max_price (float)

    """
    
    # getting the max_price parameters
    max_price = request.config.option.max_price

    if max_price is None:
        pytest.fail("You must provide max_price")

    return float(max_price)

