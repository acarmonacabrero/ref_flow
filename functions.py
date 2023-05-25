import argparse
import pandas as pd


def parser_call():
    """
    Takes in the arguments to run the analysis on the HPC
    :return: year of the study, lag, model
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', '-year', help='study year of the analysis')
    parser.add_argument('--lag', '-lag', help='lag')
    parser.add_argument('--MLmodel', '-MLmodel', help='ML model, either RF or XGB')
    study_year = int(parser.parse_args().year)
    lag = int(parser.parse_args().lag)
    MLmodel = str(parser.parse_args().MLmodel)

    # FOR QAP
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--run_id", "-id", help="run id. Used to identify output names")
    # parser.add_argument('--sam_method', '-sam', help='LHS or RS')
    # parser.add_argument('--n_gsa_runs', 'n_gsa', help='definition of gsa size. Different for LHS or RS')
    # Read arguments from the command line
    # args = parser.parse_args()
    # run_id = int(args.run_id)
    # sam_method = args.sam_method
    # n_gsa_runs = args.n_gsa_runs
    return study_year, lag, MLmodel


def feature_filter(data, country_year, year, lag=0, prior_ref=False):
    """
    Selects the predictors and target based on the study year, lag and existence of the country in the data set
    :param data: DataFrame with predictors and targets
    :param country_year: DataFrame that determines whether a country exists or not
    :param year: study year
    :param lag: lag=-1 for use of previous year data
    :param prior_ref: include prior refugee flow in the data as predictor?
    :return: X and y for the study year that includes countries that exist in that period. X includes dummies for the
    ongoing_conflict predictor
    """
    data_c = data.copy()
    for out_of_period_iso3 in country_year['ISO3']:
        if (country_year[country_year['ISO3'] == out_of_period_iso3]['year'] > (year - lag)).values:
            data_c.drop(data_c[data_c['origin_iso3'] == out_of_period_iso3].index, inplace=True)
            data_c.drop(data_c[data_c['destination_iso3'] == out_of_period_iso3].index, inplace=True)
    data_c.reset_index(inplace=True, drop=True)
    selected_columns = data_c.filter(regex=(str(year - lag))).columns.to_list()
    # try:
    #     selected_columns.remove('immigrant_population_' + str(year - lag))
    # except:
    #     pass
    # immigrant_pop_year = min(int(floor((year - lag)/10) * 10), 2010)
    # selected_columns.append('immigrant_population_' + str(immigrant_pop_year))
    selected_columns.append('contiguity_any')
    if not prior_ref:
        selected_columns.remove('ref_flow_' + str(year - lag))
    [selected_columns.append(item) for item in ['state_destination_name', 'state_origin_name', 'iid']]
    x = data_c[sorted(selected_columns)]
    x['ongoing_conflict_gradient_{}'.format(year - lag)] = pd.Categorical(
        x['ongoing_conflict_gradient_{}'.format(year - lag)])
    dum_x = pd.get_dummies(x, columns=['ongoing_conflict_gradient_{}'.format(year - lag)], prefix=['ongoing_conflict'],
                           dtype='int64')
    dum_names = {'ongoing_conflict_From 0 to 0': 'ongoing_conflict_From 0 to 0_{}'.format(year - lag),
                 'ongoing_conflict_From 0 to 1': 'ongoing_conflict_From 0 to 1_{}'.format(year - lag),
                 'ongoing_conflict_From 1 to 0': 'ongoing_conflict_From 1 to 0_{}'.format(year - lag),
                 'ongoing_conflict_From 1 to 1': 'ongoing_conflict_From 1 to 1_{}'.format(year - lag)}
    dum_x.rename(dum_names, axis=1, inplace=True)
    y = data_c['ref_flow_' + str(year)]
    return dum_x, y

