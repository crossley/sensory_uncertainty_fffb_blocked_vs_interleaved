from imports import *
from util_funcs import *

if __name__ == "__main__":
    d = load_all_data()

    grp = [15, 19, 20]
    grp += [16, 17, 18]

    dd = d.loc[np.isin(d["group"], grp)]
    dd = dd.loc[np.isin(dd["phase"], ["adaptation", "washout"])]
    dd = dd.reset_index(drop=True)

    # NOTE: report bic and predictions from models
    # grp = [15, 19, 20]
    # grp_lab = ['Experiment 3', 'Experiment 2', 'Experiment 1']
    # inspect_model_fits(grp, grp_lab)

    # NOTE: make model mechanics figures
    # models = define_models()
    # x = prepare_fit_summary(models, dd)
    # for g, grp in enumerate(dd['group'].unique()):
    #     for i, modname in enumerate(models['name'].unique()):
    #         xx = x.loc[(x['group'] == grp) & (x['model'] == modname)]
    #         fig_summary_1(xx)

    # # NOTE: report model comparison stats
    # inspect_model_stats()

    # NOTE: fit state-space model
    # models = define_models()
    # fit_models(models, dd)
    # report_fit_summary(models, dd)

    # NOTE:
    # G20 is MP only,
    # G19 is congruent MP_EP
    # G15 is 2x2

    # NOTE: regression analysis on adaptation phase
    inspect_regression(20)
    inspect_regression(19)
    inspect_regression(15)

    inspect_regression(18)
    inspect_regression(17)
    inspect_regression(16)

    # NOTE: adaptation + washout analysis
    inspect_washout(20)
    inspect_washout(19)
    inspect_washout(15)

    inspect_washout(18)
    inspect_washout(17)
    inspect_washout(16)
