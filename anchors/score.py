"""score module."""
import pandas as pd
import statsmodels.api as sm
import statsmodels
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import gpplot
from scipy import stats


def check_guide_inputs(lfc_df, reference_df):
    """ Check that input dataframes have the right column names

    Parameters
    ----------
    lfc_df: DataFrame
        DataFrame of log-fold changes with construct barcodes as the first column and conditions as the following
        conditions
    reference_df: DataFrame
        DataFrame where each anchor or modifier condition (column 1) is matched with an unperturbed reference
        condition (column 2)

    Raises
    ------
    Value Error
        If the LFC dataframe doesn't have all of the columns in the reference df
    """
    lfc_conditions = set(lfc_df.columns.to_list()[1:])
    perturbed_conditions = set(reference_df.iloc[:, 0])
    reference_conditions = set(reference_df.iloc[:, 1])
    all_conditions = perturbed_conditions.union(reference_conditions)
    all_diff = list(all_conditions - lfc_conditions)
    if len(all_diff) > 0:
        raise ValueError('LFC df missing columns ' + ', '.join(all_diff))


def fit_natural_cubic_spline(model_df, deg, x_col, y_col):
    """Fits a natural cubic spine using statsmodels and patsy

    Parameters
    ----------
    model_df: DataFrame
        Dataframe with columns x_col and y_col for modeling
    deg: int
        Degrees of freedom for the natural cubic spline
    x_col: str
        X column in data to model
    y_col: str
        Y column in data to model

    Returns
    -------
    RegressionResults
        statsmodels fit model
    """
    model_fit = sm.formula.ols(y_col + " ~ 1+ cr(" + x_col + ", df=" + str(deg) + ", constraints='center')",
                               data=model_df).fit()
    return model_fit


def cross_validate_model(model_df, deg, x_col, y_col, folds):
    """Caclulate the mean squared error for a given degree on held out folds

    Parameters
    ----------
    model_df: DataFrame
        Dataframe with columns x_col and y_col for modeling
    deg: int
        Degrees of freedom for the natural cubic spline
    x_col: str
        X column in data to model
    y_col: str
        Y column in data to model
    folds: int
        Number of folds to split data into

    Returns
    -------
    int
        Mean squared error averaged across folds
    """
    sorted_model_df = model_df.sort_values(x_col)  # sort, so we we're removing data along the x-axis
    kf = KFold(folds)
    mse_list = []
    for train_index, test_index in kf.split(sorted_model_df):
        train_model_df = sorted_model_df.iloc[train_index, ]
        model = fit_natural_cubic_spline(train_model_df, deg, x_col, y_col)
        test_model_df = sorted_model_df.iloc[test_index, ]
        test_predictions = model.predict(test_model_df[x_col])
        mse = mean_squared_error(test_model_df[y_col], test_predictions)
        mse_list.append(mse)
    mean_mse = np.mean(mse_list)
    return mean_mse


def find_optimal_degree(model_df, degrees, folds, x_col, y_col):
    """Use k-fold cross validation to find the optimal degrees of freedom for the natural cubic spline

    Parameters
    ----------
    model_df: DataFrame
        Dataframe with columns x_col and y_col for modeling
    degrees: iterable object of int
        Each value represents a degrees of freedom to test with the natural cubic spline
    folds: int
        Number of folds to split data into
    x_col: str
        X column in data to model
    y_col: str
        Y column in data to model

    Returns
    -------
    int
        Optimal degrees of freedom for the spline model
    """
    degree_mse = {}
    for d in degrees:
        degree_mse[d] = cross_validate_model(model_df, d, x_col, y_col, folds)
    optimal_degree = min(degree_mse, key=degree_mse.get)
    return optimal_degree


def plot_model_fit(model_df, predictions, x_col, y_col, condition_x, condition_y):
    """Scatterplot of y data vs x data with the natural cubic spline visualized and the line y=x for reference

    Parameters
    ----------
    model_df: DataFrame
        Dataframe with columns x_col and y_col for modeling
    predictions: array_like
        Array of predictions from the fit model
    x_col: str
        X column in data to model
    y_col: str
        Y column in data to model
    condition_x: str
        Name of the plot's x axis
    condition_y: str
        Name of the plot's y axis

    Returns
    -------
    matplotlib.axes.Axes
    matplotlib.figure.Figure

    """
    fig, ax = plt.subplots(figsize=(4, 4))
    gpplot.point_densityplot(data=model_df, x=x_col, y=y_col, alpha=0.3)
    ordered_x, ordered_predictions = zip(*sorted(zip(model_df[x_col], predictions)))
    ab_ends = [ax.get_xlim()[0], ax.get_xlim()[1]]
    plt.plot(ab_ends, ab_ends, label='y=x', linestyle='--', color='grey')
    plt.plot(ordered_x, ordered_predictions, color='black', label='fit line')
    plt.xlabel(condition_x)
    plt.ylabel(condition_y)
    plt.legend()
    return fig, ax


def get_condition_residuals(condition_x, condition_y, lfc_df, folds, degrees):
    """Calculate the residual from the function fit  between two conditions

    Parameters
    ----------
    condition_x: str
        Column name of reference condition
    condition_y: str
        Column name of perturbed condition
    lfc_df: DataFrame
        Log-fold change data
    folds: int
        Number of folds to split data into
    degrees: iterable object of ints
        Each value represents a degrees of freedom to test with the natural cubic spline

    Returns
    -------
    array_like
        Residuals for the given condition
    dict
        Information about the fit model
    matplotlib.figure.Figure
        Plot of spline fit
    """
    x_data = lfc_df[condition_x]
    y_data = lfc_df[condition_y]
    x_data = x_data.rename('x', axis=1)
    y_data = y_data.rename('y', axis=1)
    model_df = pd.concat([x_data, y_data], axis=1)
    optimal_degree = find_optimal_degree(model_df, degrees, folds, 'x', 'y')
    model_fit = fit_natural_cubic_spline(model_df, optimal_degree, 'x', 'y')
    model_info = {'model': 'spline', 'deg_fdm': optimal_degree, 'const': model_fit.params.xs('Intercept')}
    predictions = model_fit.predict(x_data)
    residuals = y_data - predictions
    fig, _ = plot_model_fit(model_df, predictions, 'x', 'y', condition_x, condition_y)
    return residuals, model_info, fig


def z_score_residuals(residual_df):
    """Standardize residuals by the mean and standard deveiation of each condition

    Parameters
    ----------
    residual_df: DataFrame
        Dataframe of residuals with the first column as construct barcodes and the subsequent conditions as residuals

    Returns
    -------
    DataFrame
        Melted dataframe with residuals for each condition
    """
    melted_residuals = (residual_df.melt(id_vars=residual_df.columns[0],
                                         var_name='condition', value_name='residual'))
    melted_residuals['residual_z'] = (melted_residuals.groupby('condition')
                                      .residual
                                      .transform(lambda x: (x - x.mean())/x.std()))
    return melted_residuals


def merge_zs_lfcs(reference_df, z_scored_residuals, lfc_df):
    """Merge z-score and lfc DataFrames

    Parameters
    ----------
    reference_df: DataFrame
        Mapping between reference and perturbed conditions
    z_scored_residuals: DataFrame
        Melted dataframe with residuals for each condition
    lfc_df: DataFrame
        Log-fold change data

    Returns
    -------
    DataFrame
        Merged residual z-scores and lfcs
    """
    perturbed_col = reference_df.columns[0]
    reference_col = reference_df.columns[1]
    condition_merged_zs = (z_scored_residuals.merge(reference_df, how='inner',
                                                    left_on='condition', right_on=perturbed_col))
    if perturbed_col != 'condition':
        condition_merged_zs = condition_merged_zs.drop(perturbed_col, axis=1)
    construct_col = lfc_df.columns[0]
    long_lfcs = lfc_df.melt(id_vars=construct_col,
                            var_name='condition', value_name='lfc')
    merged_zs_lfc = (condition_merged_zs
                     .merge(long_lfcs, how='inner', on=[construct_col, 'condition'])
                     .merge(long_lfcs, how='inner', left_on=[construct_col, reference_col],
                            right_on=[construct_col, 'condition'], suffixes=['', '_reference'])
                     .drop('condition_reference', axis=1))
    return merged_zs_lfc


def get_guide_residuals(lfc_df, reference_df, folds=10, degrees=range(2, 10)):
    """Calculate guide level residuals for paired conditions

    Parameters
    ----------
    lfc_df: DataFrame
        Log-fold change data
    reference_df: DataFrame
        Mapping between reference and perturbed conditions
    folds: int, optional
        Number of folds used in cross fold validation in picking the optimal degree for the spline model
    degrees: iterable object of ints
        Each value represents a degrees of freedom to test with the natural cubic spline

    Returns
    -------
    ...
    """
    check_guide_inputs(lfc_df, reference_df)
    residual_df = lfc_df.iloc[:, [0]].copy()
    all_model_info = {}
    model_fit_plots = {}
    for i, row in reference_df.iterrows():
        condition_x = row[1]
        condition_y = row[0]
        residuals, model_info, fig = get_condition_residuals(condition_x, condition_y, lfc_df, folds, degrees)
        residual_df[condition_y] = residuals
        all_model_info[condition_y] = model_info
        model_fit_plots[condition_y] = fig
        plt.close(fig)
    z_scored_residuals = z_score_residuals(residual_df)
    residuals_lfcs = merge_zs_lfcs(reference_df, z_scored_residuals, lfc_df)
    return residuals_lfcs, all_model_info, model_fit_plots


def check_gene_inputs(guide_residuals, guide_mapping):
    """ Check that input dataframes have the right column names

    Parameters
    ----------
    guide_residuals: DataFrame
        Results from get_guide_residuals
    guide_mapping: DataFrame
        Mapping between guides and genes. The
        first column should identify sgRNAs and the second column should identify gene symbols

    Raises
    ------
    Value Error
        If the LFC dataframe doesn't have all of the columns in the reference df
    """
    if guide_mapping.shape[1] < 2:
        raise ValueError('Guide mapping dataframe needs at least two columns')
    mapping_construct_col = guide_mapping.columns[0]
    residual_construct_col = guide_residuals.columns[0]
    residual_constructs = guide_residuals[residual_construct_col].unique()
    guide_mapping_constructs = guide_mapping[mapping_construct_col].unique()
    if not (set(residual_constructs) <= set(guide_mapping_constructs)):
        raise ValueError('Guide mapping is missing guides from the residual dataframe')


def merge_residual_mapping(guide_residuals, guide_mapping, residual_construct_col, mapping_construct_col):
    """Join guide residual df and mapping to genes

    Parameters
    ----------
    guide_residuals: DataFrame
        Guide-level residuals
    guide_mapping: DataFrame
        Mapping between guides and genes
    residual_construct_col: str
        Name of column with constructs in the residual DataFrame
    mapping_construct_col: str
        Name of column with constructs in the guide/gene mapping file

    Returns
    -------
    DataFrame
        Guide residuals mapped to genes
    """
    mapped_guide_residuals = guide_residuals.merge(guide_mapping, how='inner',
                                                   left_on=residual_construct_col,
                                                   right_on=mapping_construct_col)
    return mapped_guide_residuals


def aggregate_guide_residuals(mapped_guide_residuals, gene_col, construct_col):
    """Combine guide residuals at the gene level

    Parameters
    ----------
    mapped_guide_residuals: DataFrame
        Guide residuals mapped to genes
    gene_col: str
        Name of column with genes
    construct_col: str
        Name of column with constructs

    Returns
    -------
    DataFrame
        Gene summarized residuals
    """
    gene_residuals = (mapped_guide_residuals.groupby(['condition', gene_col])
                      .agg(sum_z=('residual_z', 'sum'),
                           guides=(construct_col, 'nunique'),
                           avg_lfc=('lfc', 'mean'),
                           avg_lfc_reference=('lfc_reference', 'mean'))
                      .reset_index())
    gene_residuals['residual_zscore'] = gene_residuals['sum_z'] / np.sqrt(gene_residuals['guides'])
    gene_residuals = gene_residuals.drop('sum_z', axis=1)
    gene_residuals['p_value'] = stats.norm.sf(abs(gene_residuals.residual_zscore)) * 2
    gene_residuals['fdr_bh'] = (gene_residuals.groupby('condition')
                                .p_value
                                .transform(lambda x: statsmodels.stats.multitest.multipletests(x, method='fdr_bh')[1]))
    return gene_residuals


def get_gene_residuals(guide_residuals, guide_mapping):
    """Combine guide residuals at the gene level

    Parameters
    ----------
    guide_residuals: DataFrame
        Results from get_guide_residuals
    guide_mapping: DataFrame
        Mapping between guides and genes. The
        first column should identify sgRNAs and the second column should identify gene symbols

    Returns
    -------

    """
    check_gene_inputs(guide_residuals, guide_mapping)
    mapping_construct_col = guide_mapping.columns[0]
    mapping_gene_col = guide_mapping.columns[1]
    residual_construct_col = guide_residuals.columns[0]
    mapped_guide_residuals = merge_residual_mapping(guide_residuals, guide_mapping, residual_construct_col,
                                                    mapping_construct_col)
    gene_residuals = aggregate_guide_residuals(mapped_guide_residuals, mapping_gene_col, residual_construct_col)
    return gene_residuals
