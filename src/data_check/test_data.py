"""
This module is all about cheking the data to ensure that our
data is according to our problem that we want to solve and our
data doesn't contain any surprises that we are not aware of.

Date: 06/March/2023
Developer: ashbab khan

"""

# importing packages
import pandas as pd
import numpy as np
import scipy.stats


def test_column_names(data):

    """
    Testing the column in the dataframe so that our data have the column 
    that we need.
    if there is any less or extra column compare to the expected_column or the 
    order is not right then this test will fail.

    """

    expected_colums = [
        "id",
        "name",
        "host_id",
        "host_name",
        "neighbourhood_group",
        "neighbourhood",
        "latitude",
        "longitude",
        "room_type",
        "price",
        "minimum_nights",
        "number_of_reviews",
        "last_review",
        "reviews_per_month",
        "calculated_host_listings_count",
        "availability_365",
    ]

    these_columns = data.columns.values

    # This also enforces the same order
    assert list(expected_colums) == list(these_columns)


def test_neighborhood_names(data):

    """
    Testing the value in neighborhood column if the column doesn't contain
    the known_names value or contain any extra value other then known_names
    then this test will fail.

    """

    known_names = ["Bronx", "Brooklyn", "Manhattan", "Queens", "Staten Island"]

    neigh = set(data['neighbourhood_group'].unique())

    # Unordered check
    assert set(known_names) == set(neigh)


def test_proper_boundaries(data: pd.DataFrame):
    
    """
    Testing proper longitude and latitude boundaries for properties in and around NYC

    """
    
    idx = data['longitude'].between(-74.25, -73.50) & data['latitude'].between(40.5, 41.2)

    assert np.sum(~idx) == 0


def test_similar_neigh_distrib(data: pd.DataFrame, ref_data: pd.DataFrame, kl_threshold: float):
    
    """
    Applying a threshold on the KL divergence to detect if the distribution of the new data is
    significantly different than that of the reference dataset
    
    """
    dist1 = data['neighbourhood_group'].value_counts().sort_index()
    dist2 = ref_data['neighbourhood_group'].value_counts().sort_index()

    assert scipy.stats.entropy(dist1, dist2, base=2) < kl_threshold

def test_row_count(data):

    """
    Testing the shape of the dataset so that it contain meaningful 
    amount of data.
    
    """
    assert (data.shape[0] > 15000) & (data.shape[0] < 100000) 



def test_price_range(data,min_price,max_price):
    assert data["price"].between(min_price,max_price).all()
