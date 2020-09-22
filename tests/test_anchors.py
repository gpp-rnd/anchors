#!/usr/bin/env python

"""Tests for `anchor_score` package."""

import pytest
import pandas as pd
from anchors import score
from numpy.testing import assert_almost_equal


@pytest.fixture
def lfc_df():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/PeterDeWeirdt/anchor_screen_parp_lfcs/master/parp_example_lfcs.csv')
    return df


@pytest.fixture
def reference_df():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/PeterDeWeirdt/anchor_screen_parp_lfcs/master/parp_example_mapping.csv')
    return df


@pytest.fixture
def guide_mapping_df():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/PeterDeWeirdt/anchor_screen_parp_lfcs/master/brunello_guide_map.csv')
    return df


def test_check_input(lfc_df, reference_df):
    with pytest.raises(ValueError):
        score.check_guide_inputs(lfc_df.drop('HAP1_PARP1_SSC_Dropout_Negative_NA_Primary',
                                             axis='columns'), reference_df)


def test_guide_residuals(lfc_df, reference_df):
    guide_residuals, _, _ = score.get_guide_residuals(lfc_df, reference_df)
    assert guide_residuals.shape[0]/8 == 77441
    # DNA polymerase beta
    assert guide_residuals.sort_values('residual_z')['Construct Barcode'].head(1).values[0] == 'ACAAGTACAATGCTTACAGG'
    avg_std = (guide_residuals.groupby('condition')
               .residual_z
               .std()).mean()
    assert_almost_equal(1, avg_std)


def test_gene_residuals(lfc_df, reference_df, guide_mapping_df):
    guide_residuals, _, _ = score.get_guide_residuals(lfc_df, reference_df)
    gene_residuals = score.get_gene_residuals(guide_residuals, guide_mapping_df)
    assert gene_residuals.sort_values('residual_zscore').head(1)['Gene Symbol'].values[0] == 'CENPS'
