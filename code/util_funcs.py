from imports import *


def define_models():
    bnds = pd.DataFrame(
        {
            "credit": "smart",
            "n_params": 18,
            "bid": ("lb", "ub"),
            "alpha_ff": (0, 1),
            "beta_ff": (0, 1),
            "bias_ff": (0, 0),
            "alpha_ff2": (0, 1),
            "beta_ff2": (0, 1),
            "bias_ff2": (-10, 10),
            "alpha_fb": (0, 1),
            "beta_fb": (-10, 10),
            "xfb_init": (-2, 2),
            "gamma_fbint_1": (0, 1),
            "gamma_fbint_2": (0, 1),
            "gamma_fbint_3": (0, 1),
            "gamma_fbint_4": (0, 1),
            "gamma_ff_1": (0, 1),
            "gamma_ff_2": (0, 1),
            "gamma_ff_3": (0, 1),
            "gamma_ff_4": (0, 1),
            "temporal_discount": (0, 1),
        }
    )

    # NOTE: error-scaling models
    m0 = bnds.copy()
    m0["name"] = "error-scale-two-state"

    m00 = bnds.copy()
    m00["name"] = "error-scale-one-state"
    m00["alpha_ff"] = (0, 0)
    m00["beta_ff"] = (0, 0)
    m00["bias_ff"] = (0, 0)
    m00["n_params"] = m00["n_params"] - 3

    m000 = m0.copy()
    m000["name"] = "error-scale-two-state-non-neg"
    m000["bias_ff2"] = (0, 10)

    # NOTE: retention-scaling models
    m1 = bnds.copy()
    m1["name"] = "retention-scale-two-state"

    m11 = bnds.copy()
    m11["name"] = "retention-scale-one-state"
    m11["alpha_ff"] = (0, 0)
    m11["beta_ff"] = (0, 0)
    m11["bias_ff"] = (0, 0)
    m11["n_params"] = m11["n_params"] - 3

    m111 = m1.copy()
    m111["name"] = "retention-scale-two-state-non-neg"
    m111["bias_ff2"] = (0, 10)

    # NOTE: bias-scaling models
    m2 = bnds.copy()
    m2["name"] = "bias-scale-two-state"
    m2["gamma_ff_1"] = (-1, 1)
    m2["gamma_ff_2"] = (-1, 1)
    m2["gamma_ff_3"] = (-1, 1)
    m2["gamma_ff_4"] = (-1, 1)

    m22 = bnds.copy()
    m22["name"] = "bias-scale-one-state"
    m22["gamma_ff_1"] = (-1, 1)
    m22["gamma_ff_2"] = (-1, 1)
    m22["gamma_ff_3"] = (-1, 1)
    m22["gamma_ff_4"] = (-1, 1)
    m22["alpha_ff"] = (0, 0)
    m22["beta_ff"] = (0, 0)
    m22["bias_ff"] = (0, 0)
    m22["n_params"] = m22["n_params"] - 3

    m222 = m2.copy()
    m222["name"] = "bias-scale-two-state-non-neg"
    m222["bias_ff2"] = (0, 10)
    m222["gamma_ff_4"] = (0, 1)

    # NOTE: x-aim-scaling models
    m3 = bnds.copy()
    m3["name"] = "x-aim-scale-two-state"
    m3["gamma_ff_1"] = (-20, 20)
    m3["gamma_ff_2"] = (-20, 20)
    m3["gamma_ff_3"] = (-20, 20)
    m3["gamma_ff_4"] = (-20, 20)
    m3["alpha_ff2"] = (1, 1)
    m3["beta_ff2"] = (0, 0)
    m3["bias_ff2"] = (0, 0)
    m3["n_params"] = m3["n_params"] - 3

    m33 = bnds.copy()
    m33["name"] = "x-aim-scale-one-state"
    m33["gamma_ff_1"] = (-20, 20)
    m33["gamma_ff_2"] = (-20, 20)
    m33["gamma_ff_3"] = (-20, 20)
    m33["gamma_ff_4"] = (-20, 20)
    m33["alpha_ff2"] = (1, 1)
    m33["beta_ff2"] = (0, 0)
    m33["bias_ff2"] = (0, 0)
    m33["alpha_ff"] = (0, 0)
    m33["beta_ff"] = (0, 0)
    m33["bias_ff"] = (0, 0)
    m33["n_params"] = m33["n_params"] - 6

    m333 = m3.copy()
    m333["name"] = "x-aim-scale-two-state-non-neg"
    m333["gamma_ff_1"] = (0, 20)
    m333["gamma_ff_2"] = (0, 20)
    m333["gamma_ff_3"] = (0, 20)
    m333["gamma_ff_4"] = (0, 20)

    # NOTE: y-aim-scaling models
    m4 = bnds.copy()
    m4["name"] = "y-aim-scale-two-state"
    m4["gamma_ff_1"] = (-20, 20)
    m4["gamma_ff_2"] = (-20, 20)
    m4["gamma_ff_3"] = (-20, 20)
    m4["gamma_ff_4"] = (-20, 20)
    m4["alpha_ff2"] = (1, 1)
    m4["beta_ff2"] = (0, 0)
    m4["bias_ff2"] = (0, 0)
    m4["n_params"] = m4["n_params"] - 3

    m44 = bnds.copy()
    m44["name"] = "y-aim-scale-one-state"
    m44["gamma_ff_1"] = (-20, 20)
    m44["gamma_ff_2"] = (-20, 20)
    m44["gamma_ff_3"] = (-20, 20)
    m44["gamma_ff_4"] = (-20, 20)
    m44["alpha_ff2"] = (1, 1)
    m44["beta_ff2"] = (0, 0)
    m44["bias_ff2"] = (0, 0)
    m44["alpha_ff"] = (0, 0)
    m44["beta_ff"] = (0, 0)
    m44["bias_ff"] = (0, 0)
    m44["n_params"] = m44["n_params"] - 6

    m444 = m4.copy()
    m444["name"] = "y-aim-scale-two-state-non-neg"
    m444["gamma_ff_1"] = (0, 20)
    m444["gamma_ff_2"] = (0, 20)
    m444["gamma_ff_3"] = (0, 20)
    m444["gamma_ff_4"] = (0, 20)

    b1 = pd.concat((m0, m1, m2, m3, m4))
    b2 = pd.concat((m00, m11, m22, m33, m44))
    b3 = pd.concat((m000, m111, m222, m333, m444))

    b = pd.concat((b1, b2, b3))

    return b


def fit_models(models, dd):
    b = models

    for i, modname in enumerate(b["name"].unique()):
        credit = b.loc[(b["name"] == modname), "credit"][0]

        lb = (
            b.loc[(b["name"] == modname) & (b["bid"] == "lb")]
            .drop(["name", "bid", "n_params", "credit"], axis=1)
            .to_numpy()[0]
        )

        ub = (
            b.loc[(b["name"] == modname) & (b["bid"] == "ub")]
            .drop(["name", "bid", "n_params", "credit"], axis=1)
            .to_numpy()[0]
        )

        bb = tuple(zip(lb, ub))

        # alpha_ff, beta_ff, bias_ff, alpha_ff2, beta_ff2, bias_ff2, alpha_fb,
        # beta_fb, xfb_init, gamma_fbint_1, gamma_fbint_2, gamma_fbint_3, gamma_fbint_4,
        # gamma_ff_1, gamma_ff_2, gamma_ff_3, gamma_ff_4, temporal_discount
        constraints = LinearConstraint(
            A=[
                [1, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
            lb=[-1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ub=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        )

        # to improve your chances of finding a global minimum use higher
        # popsize (default 15), with higher mutation (default 0.5) and
        # (dithering), but lower recombination (default 0.7). this has the
        # effect of widening the search radius, but slowing convergence.
        fit_args = {
            "obj_func": obj_func,
            "bounds": bb,
            "constraints": constraints,
            "disp": False,
            "maxiter": 3000,
            "popsize": 22,
            "mutation": 0.8,
            "recombination": 0.4,
            "tol": 1e-3,
            "polish": True,
            "updating": "deferred",
            "workers": -1,
        }

        froot = "../fits/fit_" + modname
        fit_individual(modname, dd, fit_args, froot)
        # fit_boot(credit, dd, fit_args, froot)


# TODO: big to fix here. Multiple sig_mpep_prev per trial. Not correct.
def inspect_grp_19():
    d = pd.read_csv("../data/G19_20.csv")

    d.columns = d.columns.str.lower()
    d.phase = [x.lower() for x in d.phase.to_numpy()]

    d = d[d["group"] == 19]

    phase_order = ["baseline", "adaptation", "washout", "readaptation", "testing"]
    d["phase"] = pd.Categorical(d["phase"], categories=phase_order, ordered=True)
    d = d.sort_values(["subject", "phase", "trial"]).reset_index()

    # d["trial"] = d["trial_abs"]
    # d["trial"] = d.groupby(["group", "subject", "phase"]).cumcount()

    d = prep_data_regression(d)

    dd = d.copy()
    dd = dd[dd["phase"] != "baseline"]
    dd["phase"] = dd["phase"].cat.remove_unused_categories()
    dd["sig_mpep"] = dd["sig_mpep"].cat.remove_unused_categories()
    dd["sig_mpep_prev"] = dd["sig_mpep_prev"].cat.remove_unused_categories()
    dd["rot"] *= -1

    ddd = (
        dd.groupby(["phase", "trial", "sig_mpep_prev"], observed=False)[
            [
                "ha_init",
                "ha_end",
                "delta_ha_init",
                "fb_int",
                "error_mp",
                "error_ep",
                "error_mp_prev",
                "error_ep_prev",
                "rot",
            ]
        ]
        .mean()
        .reset_index()
    )

    dd["trial"] = dd.groupby(["group", "subject"]).cumcount() + 1
    ddd["trial"] = ddd.groupby(["group"]).cumcount() + 1

    dd["group"] = dd["group"].astype("category")
    ddd["group"] = ddd["group"].astype("category")

    fig, ax = plt.subplots(3, 2, figsize=(8, 10))

    sns.scatterplot(
        data=ddd, x="trial", y="rot", ax=ax[0, 0], alpha=0.1, marker="+", color="black"
    )
    sns.scatterplot(
        data=ddd, x="trial", y="rot", ax=ax[0, 1], alpha=0.1, marker="+", color="black"
    )

    sns.lineplot(
        data=dd,
        x="trial",
        y="ha_init",
        hue="sig_mpep_prev",
        style="phase",
        markers=True,
        ax=ax[0, 0],
    )
    sns.lineplot(
        data=dd,
        x="trial",
        y="ha_end",
        hue="sig_mpep_prev",
        style="phase",
        markers=True,
        ax=ax[0, 1],
    )

    for smpep in np.sort(ddd.sig_mpep_prev.unique()):
        sns.regplot(
            data=ddd.loc[
                (ddd["phase"] == "adaptation") & (ddd["sig_mpep_prev"] == smpep)
            ],
            x="error_mp",
            y="delta_ha_init",
            label=str(smpep),
            scatter_kws={"alpha": 0.25},
            robust=False,
            ax=ax[1, 0],
        )
    #    for smpep in np.sort(ddd.sig_mpep.unique()):
    #        sns.regplot(
    #            data=ddd.loc[(ddd["phase"] == "adaptation") & (ddd["sig_mpep"] == smpep)],
    #            x="error_mp",
    #            y="fb_int",
    #            label=str(smpep),
    #            scatter_kws={"alpha": 0.25},
    #            robust=False,
    #            ax=ax[2, 0],
    #        )

    for smpep in np.sort(ddd.sig_mpep_prev.unique()):
        sns.regplot(
            data=ddd.loc[
                (ddd["phase"] == "adaptation") & (ddd["sig_mpep_prev"] == smpep)
            ],
            x="error_ep",
            y="delta_ha_init",
            label=str(smpep),
            scatter_kws={"alpha": 0.25},
            robust=False,
            ax=ax[1, 1],
        )
    #    for smpep in np.sort(ddd.sig_mpep.unique()):
    #        sns.regplot(
    #            data=ddd.loc[(ddd["phase"] == "adaptation") & (ddd["sig_mpep"] == smpep)],
    #            x="error_ep",
    #            y="fb_int",
    #            label=str(smpep),
    #            scatter_kws={"alpha": 0.25},
    #            robust=False,
    #            ax=ax[2, 1],
    #        )

    for axx in [ax[0, 0], ax[0, 1]]:
        sns.move_legend(
            axx,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=3,
            title=None,
            frameon=False,
        )
    ax[0, 0].set_ylabel("Initial hand angle (deg)")
    ax[0, 1].set_ylabel("Endpoint hand angle (deg)")
    ax[1, 0].set_ylabel("Change in initial hand angle (deg)")
    ax[1, 1].set_ylabel("Change in initial hand angle (deg)")
    ax[2, 0].set_ylabel("Feedback integratino (deg)")
    ax[2, 1].set_ylabel("Feedback integratino (deg)")
    plt.tight_layout()
    plt.savefig("../figures/tmp_19_2.pdf")
    plt.close()
    # plt.show()

    # # NOTE: regression
    # mod_formula = "ha_init ~ "
    # mod_formula += "C(sig_mpep_prev, Diff) * error_mp_prev + "
    # mod_formula += "C(grouppep_prev, Diff) * error_ep_prev + "
    # mod_formula += "np.log(trial) + "
    # mod_formula += "1"

    # mod = smf.ols(mod_formula, data=ddd[ddd["phase"] == "adaptation"])
    # res_sm = mod.fit()
    # print(res_sm.summary())


def inspect_block_exp():
    d = pd.read_csv("../data/exp6_2020.csv")

    d.columns = d.columns.str.lower()
    d.phase = [x.lower() for x in d.phase.to_numpy()]

    d = d.sort_values(["group", "subject", "phase", "trial_abs"]).reset_index()

    d["trial"] = d["trial_abs"]
    d["trial"] = d.groupby(["group", "subject", "phase"]).cumcount()
    phase_order = ["baseline", "adaptation", "washout", "readaptation", "testing"]
    d["phase"] = pd.Categorical(d["phase"], categories=phase_order, ordered=True)
    d = d.sort_values(["group", "subject", "phase", "trial"]).reset_index()

    d = prep_data_regression(d)

    dd = d.copy()
    dd = dd[dd["phase"] != "baseline"]
    dd["phase"] = dd["phase"].cat.remove_unused_categories()
    dd["sig_mpep"] = dd["sig_mpep"].cat.remove_unused_categories()
    dd["sig_mpep_prev"] = dd["sig_mpep_prev"].cat.remove_unused_categories()
    dd["rot"] *= -1

    ddd = (
        dd.groupby(["group", "phase", "trial", "sig_mpep_prev"], observed=True)[
            [
                "ha_init",
                "ha_end",
                "delta_ha_init",
                "fb_int",
                "error_mp",
                "error_ep",
                "error_mp_prev",
                "error_ep_prev",
                "rot",
            ]
        ]
        .mean()
        .reset_index()
    )

    dd["trial"] = dd.groupby(["group", "subject"]).cumcount() + 1
    ddd["trial"] = ddd.groupby(["group"]).cumcount() + 1

    dd["group"] = dd["group"].astype("category")
    ddd["group"] = ddd["group"].astype("category")

    fig, ax = plt.subplots(3, 2, figsize=(8, 10))

    sns.scatterplot(
        data=ddd, x="trial", y="rot", ax=ax[0, 0], alpha=0.1, marker="+", color="black"
    )
    sns.scatterplot(
        data=ddd, x="trial", y="rot", ax=ax[0, 1], alpha=0.1, marker="+", color="black"
    )

    sns.lineplot(
        data=dd,
        x="trial",
        y="ha_init",
        hue="group",
        ax=ax[0, 0],
    )
    sns.lineplot(
        data=dd,
        x="trial",
        y="ha_end",
        hue="group",
        ax=ax[0, 1],
    )

    for grp in np.sort(ddd.group.unique()):
        sns.regplot(
            data=ddd.loc[(ddd["phase"] == "adaptation") & (ddd["group"] == grp)],
            x="error_mp",
            y="delta_ha_init",
            label=str(smpep),
            scatter_kws={"alpha": 0.25},
            robust=False,
            ax=ax[1, 0],
        )
    for grp in np.sort(ddd.group.unique()):
        sns.regplot(
            data=ddd.loc[(ddd["phase"] == "adaptation") & (ddd["group"] == grp)],
            x="error_mp",
            y="fb_int",
            label=str(smpep),
            scatter_kws={"alpha": 0.25},
            robust=False,
            ax=ax[2, 0],
        )

    for grp in np.sort(ddd.group.unique()):
        sns.regplot(
            data=ddd.loc[(ddd["phase"] == "adaptation") & (ddd["group"] == grp)],
            x="error_ep",
            y="delta_ha_init",
            label=str(smpep),
            scatter_kws={"alpha": 0.25},
            robust=False,
            ax=ax[1, 1],
        )
    for grp in np.sort(ddd.group.unique()):
        sns.regplot(
            data=ddd.loc[(ddd["phase"] == "adaptation") & (ddd["group"] == grp)],
            x="error_ep",
            y="fb_int",
            label=str(smpep),
            scatter_kws={"alpha": 0.25},
            robust=False,
            ax=ax[2, 1],
        )

    for axx in [ax[0, 0], ax[0, 1]]:
        sns.move_legend(
            axx,
            "lower center",
            bbox_to_anchor=(0.5, 1),
            ncol=3,
            title=None,
            frameon=False,
        )
    ax[0, 0].set_ylabel("Initial hand angle (deg)")
    ax[0, 1].set_ylabel("Endpoint hand angle (deg)")
    ax[1, 0].set_ylabel("Change in initial hand angle (deg)")
    ax[1, 1].set_ylabel("Change in initial hand angle (deg)")
    ax[2, 0].set_ylabel("Feedback integratino (deg)")
    ax[2, 1].set_ylabel("Feedback integratino (deg)")
    plt.tight_layout()
    plt.savefig("../figures/tmp_blocked.pdf")
    plt.close()
    # plt.show()

    # NOTE: regression
    mod_formula = "ha_init ~ "
    mod_formula += "C(group, Diff) * error_mp_prev + "
    mod_formula += "C(group, Diff) * error_ep_prev + "
    mod_formula += "np.log(trial) + "
    mod_formula += "1"

    mod = smf.ols(mod_formula, data=ddd[ddd["phase"] == "adaptation"])
    res_sm = mod.fit()
    print(res_sm.summary())


def load_all_data_grand():
    d1 = pd.read_csv("../data/exp1_2020.csv")
    d2 = pd.read_csv("../data/exp2_2020.csv")
    d3 = pd.read_csv("../data/exp3_2020.csv")
    d4 = pd.read_csv("../data/exp4_2020.csv")
    d5 = pd.read_csv("../data/exp5_2020.csv")
    d6 = pd.read_csv("../data/exp6_2020.csv")
    d7 = pd.read_csv("../data/exp7_2020.csv")
    d8 = pd.read_csv("../data/exp8_2021.csv")
    d345 = pd.read_csv("../data/exp345_2021.csv")
    d15 = pd.read_csv("../data/G15.csv")
    d16 = pd.read_csv("../data/G16.csv")
    d1718 = pd.read_csv("../data/G17_18.csv")
    d1920 = pd.read_csv("../data/G19_20.csv")

    d15["HA_INIT"] = d15["HA_INT"]
    d15["HA_MID"] = d15["HA_INT"]

    d = pd.concat(
        (d1, d2, d3, d4, d5, d6, d7, d8, d345, d15, d16, d1718, d1920), sort=False
    )

    d.columns = d.columns.str.lower()
    d.phase = [x.lower() for x in d.phase.to_numpy()]

    d["trial"] = d["trial_abs"]
    d["trial"] = d.groupby(["group", "subject", "phase"]).cumcount()

    phase_order = ["baseline", "adaptation", "washout", "readaptation", "testing"]
    d["phase"] = pd.Categorical(d["phase"], categories=phase_order, ordered=True)

    d = d.sort_values(["group", "subject", "phase", "trial"]).reset_index()

    d["trial"] = d.groupby(["group", "subject"]).cumcount()

    d = prep_data_regression(d)

    return d


def load_all_data():
    d15 = pd.read_csv("../data/G15.csv")
    d16 = pd.read_csv("../data/G16.csv")
    d1718 = pd.read_csv("../data/G17_18.csv")
    d1920 = pd.read_csv("../data/G19_20.csv")

    d15["HA_INIT"] = d15["HA_INT"]
    d15["HA_MID"] = d15["HA_INT"]

    d = pd.concat((d15, d16, d1718, d1920), sort=False)

    d.columns = d.columns.str.lower()
    d.phase = [x.lower() for x in d.phase.to_numpy()]

    d = d.sort_values(["group", "subject", "phase", "trial"])
    d["trial"] = d.groupby(["group", "subject"]).cumcount()
    d["trial"] = d["trial"] + 1

    return d


def prepare_fit_summary(models, d):
    drec = {
        "group": [],
        "subject": [],
        "model": [],
        "params": [],
        "x_pred_ep": [],
        "x_pred_mp": [],
        "x_obs_ep": [],
        "x_obs_mp": [],
        "sig_mp": [],
        "sig_ep": [],
        "rot": [],
        "y": [],
        "yff": [],
        "xff": [],
        "xff2": [],
        "r_squared_ep": [],
        "r_squared_mp": [],
        "r_squared": [],
        "bic": [],
    }

    for g, grp in enumerate(d["group"].unique()):
        subs = d[d["group"] == grp]["subject"].unique()

        for s, sub in enumerate(subs):
            for i, modname in enumerate(models["name"].unique()):
                credit = models.loc[(models["name"] == modname), "credit"][0]

                dd = d[(d["group"] == grp) & (d["subject"] == sub)]
                x_obs_mp = dd["ha_init"].to_numpy()
                x_obs_ep = dd["ha_end"].to_numpy()
                rot = dd["rot"].to_numpy()
                sig_mp = dd["sig_mp"].to_numpy()
                sig_ep = dd["sig_ep"].to_numpy()
                group = dd["group"].to_numpy()
                args = (rot, sig_mp, sig_ep, group, modname)

                fname = (
                    "../fits/fit_"
                    + modname
                    + "_group_"
                    + str(grp)
                    + "_sub_"
                    + str(sub)
                    + ".txt"
                )
                p = np.loadtxt(fname, delimiter=",")
                if modname == "kalman":
                    (y, yff, yfb, xff, xfb, xff2) = simulate_kalman(p[:-1], args)
                else:
                    (y, yff, yfb, xff, xfb, xff2) = simulate(p[:-1], args)

                x_obs_mp = x_obs_mp[:-1]
                x_obs_ep = x_obs_ep[:-1]
                yff = yff[:-1]
                xff = xff[:-1]
                xff2 = xff2[:-1]
                y = y[:-1]
                sig_mp = sig_mp[:-1]
                sig_ep = sig_ep[:-1]

                ss_tot_mp = np.nansum((x_obs_mp - np.nanmean(x_obs_mp)) ** 2)
                ss_reg_mp = np.nansum((yff - np.nanmean(x_obs_mp)) ** 2)
                ss_res_mp = np.nansum((x_obs_mp - yff) ** 2)
                ss_tot_ep = np.nansum((x_obs_ep - np.nanmean(x_obs_ep)) ** 2)
                ss_reg_ep = np.nansum((y - np.nanmean(x_obs_ep)) ** 2)
                ss_res_ep = np.nansum((x_obs_ep - y) ** 2)

                r_squared_mp = 1 - ss_res_mp / ss_tot_mp
                r_squared_ep = 1 - ss_res_ep / ss_tot_ep
                r_squared = 1 - (ss_res_ep + ss_res_mp) / (ss_tot_ep + ss_tot_mp)

                n = dd.shape[0]
                k = models.loc[models["name"] == modname, "n_params"].unique()[0]
                bic = compute_bic(r_squared, n, k)

                drec["group"].append(grp)
                drec["subject"].append(sub)
                drec["model"].append(modname)
                drec["params"].append(p)
                drec["x_pred_ep"].append(y)
                drec["x_pred_mp"].append(yff)
                drec["x_obs_ep"].append(x_obs_ep)
                drec["x_obs_mp"].append(x_obs_mp)
                drec["sig_mp"].append(sig_mp)
                drec["sig_ep"].append(sig_ep)
                drec["rot"].append(rot)
                drec["y"].append(y)
                drec["yff"].append(yff)
                drec["xff"].append(xff)
                drec["xff2"].append(xff2)
                drec["r_squared_ep"].append(r_squared_ep)
                drec["r_squared_mp"].append(r_squared_mp)
                drec["r_squared"].append(r_squared)
                drec["bic"].append(bic)

    drec = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in drec.items()]))

    return drec


def comp_traj(x):
    x_obs_mp = x["x_obs_mp"].mean()
    x_obs_ep = x["x_obs_ep"].mean()
    y = x["y"].mean()
    yff = x["yff"].mean()
    xff = x["xff"].mean()
    xff2 = x["xff2"].mean()
    p = np.vstack(x["params"].to_numpy())
    grp = x["group"].unique()[0]
    model = x["model"].unique()[0]
    r_squared_mp = np.round(x["r_squared_mp"].mean(), 2)
    r_squared_ep = np.round(x["r_squared_ep"].mean(), 2)
    subject = x["subject"].to_numpy()[0]
    sig_mp = x["sig_mp"].mean()
    sig_ep = x["sig_ep"].mean()

    ss_tot_mp = np.nansum((x_obs_mp - np.nanmean(x_obs_mp)) ** 2)
    ss_reg_mp = np.nansum((yff - np.nanmean(x_obs_mp)) ** 2)
    ss_res_mp = np.nansum((x_obs_mp - yff) ** 2)
    ss_tot_ep = np.nansum((x_obs_ep - np.nanmean(x_obs_ep)) ** 2)
    ss_reg_ep = np.nansum((y - np.nanmean(x_obs_ep)) ** 2)
    ss_res_ep = np.nansum((x_obs_ep - y) ** 2)

    r_squared_mp_mean = 1 - ss_res_mp / ss_tot_mp
    r_squared_ep_mean = 1 - ss_res_ep / ss_tot_ep
    r_squared_mean = 1 - (ss_res_ep + ss_res_mp) / (ss_tot_ep + ss_tot_mp)

    r_squared_mp_mean = np.round(r_squared_mp_mean, 2)
    r_squared_ep_mean = np.round(r_squared_ep_mean, 2)
    r_squared_mean = np.round(r_squared_mean, 2)

    return {
        "x_obs_mp": x_obs_mp,
        "x_obs_ep": x_obs_ep,
        "y": y,
        "yff": yff,
        "xff": xff,
        "xff2": xff2,
        "p": p,
        "grp": grp,
        "model": model,
        "r_squared_mp": r_squared_mp,
        "r_squared_ep": r_squared_ep,
        "subject": subject,
        "sig_mp": sig_mp,
        "sig_ep": sig_ep,
        "r_squared_mp_mean": r_squared_mp_mean,
        "r_squared_ep_mean": r_squared_ep_mean,
        "r_squared_mean": r_squared_mean,
    }


def fig_summary_1(x):
    x = comp_traj(x)

    x_obs_mp = x["x_obs_mp"]
    x_obs_ep = x["x_obs_ep"]
    y = x["y"]
    yff = x["yff"]
    xff = x["xff"]
    xff2 = x["xff2"]
    p = x["p"]
    grp = x["grp"]
    model = x["model"]
    r_squared_mp = x["r_squared_mp"]
    r_squared_ep = x["r_squared_ep"]
    subject = x["subject"]
    sig_mp = x["sig_mp"]
    sig_ep = x["sig_ep"]
    r_squared_mp_mean = x["r_squared_mp_mean"]
    r_squared_ep_mean = x["r_squared_ep_mean"]
    r_squared_mean = x["r_squared_mean"]

    if model == "x-aim-scale-one-state":
        model = "state-aim-scale-one-state"
    elif model == "x-aim-scale-two-state":
        model = "state-aim-scale-two-state"
    elif model == "x-aim-scale-two-state-non-neg":
        model = "state-aim-scale-two-state-non-neg"
    elif model == "y-aim-scale-one-state":
        model = "output-aim-scale-one-state"
    elif model == "y-aim-scale-two-state":
        model = "output-aim-scale-two-state"
    elif model == "y-aim-scale-two-state-non-neg":
        model = "output-aim-scale-two-state-non-neg"

    p_names = [
        "alpha_ff_1",
        "beta_ff_1",
        "bias_ff_1",
        "alpha_ff_2",
        "beta_ff_2",
        "bias_ff_2",
        "alpha_fb",
        "beta_fb",
        "fb_init",
        "gamma_fb_1",
        "gamma_fb_2",
        "gamma_fb_3",
        "gamma_fb_4",
        "gamma_ff_1",
        "gamma_ff_2",
        "gamma_ff_3",
        "gamma_ff_4",
        "temporal_discount",
    ]
    dfp = pd.DataFrame(data=p[:, :-1])
    dfp.columns = p_names
    dfp = dfp[
        [
            "alpha_ff_1",
            "alpha_ff_2",
            "beta_ff_1",
            "beta_ff_2",
            "bias_ff_1",
            "bias_ff_2",
            "alpha_fb",
            "beta_fb",
            "fb_init",
            "gamma_fb_1",
            "gamma_fb_2",
            "gamma_fb_3",
            "gamma_fb_4",
            "gamma_ff_1",
            "gamma_ff_2",
            "gamma_ff_3",
            "gamma_ff_4",
            "temporal_discount",
        ]
    ]

    # NOTE: Rescale parameters to (0, 1)
    # dfp['bias_ff_1'] = dfp['bias_ff_1'] / 10
    # dfp['bias_ff_2'] = dfp['bias_ff_2'] / 10
    # dfp['gamma_fb_4'] = dfp['gamma_fb_4'] / 1

    dfp = dfp.drop(columns=["alpha_fb", "beta_fb", "fb_init"])
    dfp = dfp.melt()
    dfp["subject"] = dfp.groupby(["variable"]).transform(
        lambda x: np.arange(0, x.shape[0], 1)
    )
    dfp["var_color"] = "C0"

    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 8)

    ax1 = fig.add_subplot(gs[0, :4])
    ax2 = fig.add_subplot(gs[0, 4:])
    x = np.arange(0, x_obs_mp.shape[0], 1)
    ax1.plot(x_obs_mp, label="Human")
    ax1.plot(yff, label="Model Full Output")
    ax1.plot(xff, label="Model State slow")
    ax1.plot(xff2, label="Model State fast")
    ax2.plot(x_obs_ep)
    ax2.plot(y)
    ax1.set_ylabel("Initial movement vector")
    ax2.set_ylabel("Endpoint hand angle")

    if grp == 15 or grp == 16:
        ax3 = fig.add_subplot(gs[1, :4])
        d = dfp.loc[np.isin(dfp["variable"], ["gamma_ff_1", "gamma_ff_3"])]
    else:
        ax3 = fig.add_subplot(gs[1, :4])
        d = dfp.loc[
            np.isin(dfp["variable"], ["gamma_ff_1", "gamma_ff_2", "gamma_ff_3"])
        ]

    pg.plot_paired(
        data=d,
        dv="value",
        within="variable",
        subject="subject",
        boxplot_in_front=True,
        ax=ax3,
    )

    if grp == 15 or grp == 16:
        ax4 = fig.add_subplot(gs[1, 4:])
        d = dfp.loc[np.isin(dfp["variable"], ["gamma_fb_1", "gamma_fb_3"])]
    else:
        ax4 = fig.add_subplot(gs[1, 4:])
        d = dfp.loc[
            np.isin(dfp["variable"], ["gamma_fb_1", "gamma_fb_2", "gamma_fb_3"])
        ]

    pg.plot_paired(
        data=d,
        dv="value",
        within="variable",
        subject="subject",
        boxplot_in_front=True,
        ax=ax4,
    )

    ax5 = fig.add_subplot(gs[2, :2])
    d = dfp.loc[np.isin(dfp["variable"], ["alpha_ff_1", "alpha_ff_2"])]
    pg.plot_paired(
        data=d,
        dv="value",
        within="variable",
        subject="subject",
        boxplot_in_front=True,
        ax=ax5,
    )
    ax6 = fig.add_subplot(gs[2, 2:4])
    d = dfp.loc[np.isin(dfp["variable"], ["beta_ff_1", "beta_ff_2"])]
    pg.plot_paired(
        data=d,
        dv="value",
        within="variable",
        subject="subject",
        boxplot_in_front=True,
        ax=ax6,
    )

    ax7 = fig.add_subplot(gs[2, 7])
    d = dfp.loc[np.isin(dfp["variable"], ["bias_ff_2"])]
    pg.plot_paired(
        data=d,
        dv="value",
        within="variable",
        subject="subject",
        boxplot_in_front=True,
        pointplot_kwargs={"scale": 0.0},
        colors=["black", "black", "black"],
        ax=ax7,
    )

    ax8 = fig.add_subplot(gs[2, 6])
    d = dfp.loc[np.isin(dfp["variable"], ["temporal_discount"])]
    pg.plot_paired(
        data=d,
        dv="value",
        within="variable",
        subject="subject",
        boxplot_in_front=True,
        ax=ax8,
    )

    ax33 = fig.add_subplot(gs[2, 4])
    d33 = dfp.loc[np.isin(dfp["variable"], ["gamma_ff_4"])]
    pg.plot_paired(
        data=d33,
        dv="value",
        within="variable",
        subject="subject",
        boxplot_in_front=True,
        ax=ax33,
    )

    ax44 = fig.add_subplot(gs[2, 5])
    d44 = dfp.loc[np.isin(dfp["variable"], ["gamma_fb_4"])]
    pg.plot_paired(
        data=d44,
        dv="value",
        within="variable",
        subject="subject",
        boxplot_in_front=True,
        ax=ax44,
    )

    ax1.legend()
    # plt.xticks(rotation=45)
    grp_title = "undefined"
    if grp == 20:
        grp_title = "Experiment 1"
    elif grp == 19:
        grp_title = "Experiment 2"
    elif grp == 15:
        grp_title = "Experiment 3"
    ax1.set_title(
        str(grp_title) + " " + str(model) + "\n" r"$R^2 = $" + str(r_squared_mp)
    )
    ax2.set_title(
        str(grp_title) + " " + str(model) + "\n" r"$R^2 = $" + str(r_squared_ep)
    )
    labels = list(string.ascii_lowercase)
    ax = [ax1, ax2, ax3, ax4, ax5, ax6, ax33, ax44, ax8, ax7]
    for i, x in enumerate(ax):
        x.annotate(text=labels[i], xy=(-0.0, 1.1), xycoords="axes fraction", size=14)
    plt.tight_layout()
    plt.savefig(
        "../figures/fit_summary_grp_" + str(grp) + "_mod_" + str(model) + ".tif"
    )
    plt.savefig(
        "../figures/fit_summary_grp_" + str(grp) + "_mod_" + str(model) + ".pdf"
    )
    plt.close()


def inspect_model_fits(g, g_lab):
    models = define_models()

    d = load_all_data()
    d = d.sort_values(["group", "subject", "trial"])
    d = d.loc[np.isin(d["group"], g)]
    d = d.loc[np.isin(d["phase"], ["adaptation", "washout"])]
    d = prepare_fit_summary(models, d)

    n_models = d["model"].unique().shape[0]

    model_names_old = [
        "error-scale-one-state",
        "error-scale-two-state",
        "error-scale-two-state-non-neg",
        "retention-scale-one-state",
        "retention-scale-two-state",
        "retention-scale-two-state-non-neg",
        "bias-scale-one-state",
        "bias-scale-two-state",
        "bias-scale-two-state-non-neg",
        "x-aim-scale-one-state",
        "x-aim-scale-two-state",
        "x-aim-scale-two-state-non-neg",
        "y-aim-scale-one-state",
        "y-aim-scale-two-state",
        "y-aim-scale-two-state-non-neg",
    ]

    model_names_new = [
        "error-scale-one-state",
        "error-scale-two-state",
        "error-scale-two-state-non-neg",
        "retention-scale-one-state",
        "retention-scale-two-state",
        "retention-scale-two-state-non-neg",
        "bias-scale-one-state",
        "bias-scale-two-state",
        "bias-scale-two-state-non-neg",
        "state-aim-scale-one-state",
        "state-aim-scale-two-state",
        "state-aim-scale-two-state-non-neg",
        "output-aim-scale-one-state",
        "output-aim-scale-two-state",
        "output-aim-scale-two-state-non-neg",
    ]

    d["model"] = d["model"].astype("category")
    d["model"] = d["model"].cat.rename_categories(
        dict(zip(model_names_old, model_names_new))
    )

    palette = []
    palette.extend(sns.light_palette("#1f77b4", 4)[1:])
    palette.extend(sns.light_palette("#ff7f0e", 4)[1:])
    palette.extend(sns.light_palette("#2ca02c", 4)[1:])
    palette.extend(sns.light_palette("#d62728", 4)[1:])
    palette.extend(sns.light_palette("#9467bd", 4)[1:])

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 4))
    sns.barplot(
        data=d,
        x="group",
        y="bic",
        hue="model",
        hue_order=model_names_new,
        palette=palette,
        ax=ax[0, 0],
    )
    sns.move_legend(ax[0, 0], loc=(1.05, 0.0), ncol=1)
    ax[0, 0].set_xlabel("")
    ax[0, 0].set_ylabel("BIC")
    ax[0, 0].xaxis.tick_top()
    ax[0, 0].set_xticklabels(g_lab)
    ax[0, 0].invert_xaxis()
    plt.tight_layout()
    plt.savefig("../figures/fig_mod_bic.tif")
    plt.savefig("../figures/fig_mod_bic.pdf")
    plt.close()

    def create_fit_frame(x):
        x_obs_mp = x["x_obs_mp"].to_numpy()[0]
        x_obs_ep = x["x_obs_ep"].to_numpy()[0]
        x_pred_mp = x["x_pred_mp"].to_numpy()[0]
        x_pred_ep = x["x_pred_ep"].to_numpy()[0]
        trial = np.arange(0, x_obs_mp.shape[0], 1)
        xx = pd.DataFrame(
            {
                "trial": trial,
                "x_obs_mp": x_obs_mp,
                "x_obs_ep": x_obs_ep,
                "x_pred_mp": x_pred_mp,
                "x_pred_ep": x_pred_ep,
            }
        )
        return xx

    dfit = (
        d.groupby(["group", "model", "subject"]).apply(create_fit_frame).reset_index()
    )

    model_order = [
        "error-scale-two-state",
        "retention-scale-two-state",
        "bias-scale-two-state",
        "state-aim-scale-two-state",
        "output-aim-scale-two-state",
    ]

    model_title = [
        "error-scale",
        "retention-scale",
        "bias-scale",
        "state-aim",
        "output-aim",
    ]

    for g in dfit["group"].unique():
        fig, ax = plt.subplots(5, 2, squeeze=False, figsize=(8, 11))
        for i, m in enumerate(model_order):
            dfitgm = dfit.loc[(dfit["group"] == g) & (dfit["model"] == m)]
            r2_mp_mean = d.loc[
                (d["group"] == g) & (d["model"] == m), "r_squared_mp"
            ].mean()
            r2_ep_mean = d.loc[
                (d["group"] == g) & (d["model"] == m), "r_squared_ep"
            ].mean()
            r2_mp_sd = d.loc[
                (d["group"] == g) & (d["model"] == m), "r_squared_mp"
            ].std()
            r2_ep_sd = d.loc[
                (d["group"] == g) & (d["model"] == m), "r_squared_ep"
            ].std()
            sns.lineplot(
                data=dfitgm,
                x="trial",
                y="x_obs_mp",
                label="obs",
                n_boot=10,
                ax=ax[i, 0],
            )
            sns.lineplot(
                data=dfitgm,
                x="trial",
                y="x_pred_mp",
                label="pred",
                n_boot=10,
                ax=ax[i, 0],
            )
            sns.lineplot(
                data=dfitgm,
                x="trial",
                y="x_obs_ep",
                label="obs",
                n_boot=10,
                ax=ax[i, 1],
            )
            sns.lineplot(
                data=dfitgm,
                x="trial",
                y="x_pred_ep",
                label="pred",
                n_boot=10,
                ax=ax[i, 1],
            )
            [x.set_title(model_title[i]) for x in ax[i, :]]
            ax[i, 0].annotate(
                text=str(
                    r"$R^2 = $"
                    + str(np.round(r2_mp_mean, 2))
                    + r"$\pm$"
                    + str(np.round(r2_mp_sd, 2))
                    + " s.d."
                ),
                xy=(0.2, 0.1),
                xycoords="axes fraction",
                size=10,
            )
            ax[i, 1].annotate(
                text=r"$R^2 = $"
                + str(np.round(r2_ep_mean, 2))
                + r"$\pm$"
                + str(np.round(r2_ep_sd, 2))
                + " s.d.",
                xy=(0.2, 0.1),
                xycoords="axes fraction",
                size=10,
            )
        [sns.move_legend(x, loc="upper right", ncol=1) for x in ax.flatten()]
        [x.set_xlabel("Trial") for x in ax.flatten()]
        [x.set_ylabel("Initial movement vector") for x in ax[:, 0]]
        [x.set_ylabel("Endpoint hand angle") for x in ax[:, 1]]
        labels = list(string.ascii_lowercase)
        for i, x in enumerate(ax.flatten()):
            x.annotate(
                text=labels[i], xy=(-0.0, 1.1), xycoords="axes fraction", size=14
            )
        plt.tight_layout()
        plt.savefig("../figures/fig_mod_pred_grp_" + str(g) + ".tif")
        plt.savefig("../figures/fig_mod_pred_grp_" + str(g) + ".pdf")
        plt.close()


def report_fit_summary(models, d):
    d = prepare_fit_summary(models, d)
    d[["group", "model", "subject", "bic"]].to_csv(
        "../fits/fit_summary.csv", index=False
    )

    d["bic_min"] = d.groupby(["group", "subject"])["bic"].transform(
        lambda x: np.min(x.to_numpy())
    )

    fit_summary = d.loc[d["bic"] == d["bic_min"]].groupby(["group", "model"])

    print()
    print(fit_summary["bic", "r_squared", "r_squared_mp", "r_squared_ep"].mean())
    print()
    print(fit_summary["model"].count())


def fit_individual(modname, d, fit_args, froot):
    obj_func = fit_args["obj_func"]
    bounds = fit_args["bounds"]
    constraints = fit_args["constraints"]
    maxiter = fit_args["maxiter"]
    disp = fit_args["disp"]
    tol = fit_args["tol"]
    polish = fit_args["polish"]
    updating = fit_args["updating"]
    workers = fit_args["workers"]
    popsize = fit_args["popsize"]
    mutation = fit_args["mutation"]
    recombination = fit_args["recombination"]

    for grp in d["group"].unique():
        for sub in d[d["group"] == grp]["subject"].unique():
            dd = d[(d["subject"] == sub) & (d["group"] == grp)][
                ["rot", "ha_init", "ha_end", "trial_abs", "group", "sig_mp", "sig_ep"]
            ]

            rot = dd.rot.to_numpy()
            sig_mp = dd.sig_mp.to_numpy()
            sig_ep = dd.sig_ep.to_numpy()
            group = dd.group.to_numpy()
            x_obs_mp = dd["ha_init"].to_numpy()
            x_obs_ep = dd["ha_end"].to_numpy()

            args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep, group, modname)

            results = differential_evolution(
                func=obj_func,
                bounds=bounds,
                constraints=constraints,
                args=args,
                disp=disp,
                maxiter=maxiter,
                popsize=popsize,
                mutation=mutation,
                recombination=recombination,
                tol=tol,
                polish=polish,
                updating=updating,
                workers=workers,
            )

            fout = froot + "_group_" + str(grp) + "_sub_" + str(sub) + ".txt"
            with open(fout, "w") as f:
                tmp = np.concatenate((results["x"], [results["fun"]]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, "%0.4f", delimiter=",", newline="\n")


def obj_func(params, *args):
    obs = args

    rot = obs[0]
    sig_mp = obs[1]
    sig_ep = obs[2]
    x_obs_mp = obs[3]
    x_obs_ep = obs[4]
    group = obs[5]
    modname = obs[6]

    args = (rot, sig_mp, sig_ep, group, modname)

    x_pred = simulate(params, args)
    x_pred_mp = x_pred[1]
    x_pred_ep = x_pred[0]

    sse_mp = np.sum((x_obs_mp - x_pred_mp) ** 2)
    sse_ep = np.sum((x_obs_ep - x_pred_ep) ** 2)

    sse = sse_mp + sse_ep

    return sse


def obj_func_context(params, *args):
    obs = args

    rot = obs[0]
    sig_mp = obs[1]
    sig_ep = obs[2]
    x_obs_mp = obs[3]
    x_obs_ep = obs[4]
    group = obs[5]
    modname = obs[6]

    args = (rot, sig_mp, sig_ep, group, modname)

    x_pred = simulate_context(params, args)
    x_pred_mp = x_pred[1]
    x_pred_ep = x_pred[0]

    sse_mp = np.sum((x_obs_mp - x_pred_mp) ** 2)
    sse_ep = np.sum((x_obs_ep - x_pred_ep) ** 2)

    sse = sse_mp + sse_ep

    return sse


def simulate_context(params, args):
    alpha_ff = params[0]
    beta_ff = params[1]
    bias_ff = params[2]

    alpha_ff2 = params[3]
    beta_ff2 = params[4]
    bias_ff2 = params[5]

    alpha_fb = params[6]
    beta_fb = params[7]
    xfb_init = params[8]

    gamma_fbint_1 = params[9]
    gamma_fbint_2 = params[10]
    gamma_fbint_3 = params[11]
    gamma_fbint_4 = params[12]
    gamma_fbint = np.array([gamma_fbint_1, gamma_fbint_2, gamma_fbint_3, gamma_fbint_4])

    gamma_ff_1 = params[13]
    gamma_ff_2 = params[14]
    gamma_ff_3 = params[15]
    gamma_ff_4 = params[16]
    gamma_ff = np.array([gamma_ff_1, gamma_ff_2, gamma_ff_3, gamma_ff_4])

    r = args[0]
    sig_mp = args[1]
    sig_ep = args[2]
    group = args[3]
    modname = args[4]

    n_trials = r.shape[0]

    delta_ep = np.zeros(n_trials)
    delta_mp = np.zeros(n_trials)
    xff = np.zeros((4, n_trials))
    xff2 = np.zeros(n_trials)
    xfb = np.zeros(n_trials)
    yff = np.zeros(n_trials)
    yfb = np.zeros(n_trials)
    y = np.zeros(n_trials)

    xfb[0] = xfb_init

    # some condition files call a 4 a 0
    sig_mp[sig_mp == 0] = 4
    sig_ep[sig_ep == 0] = 4

    current_context = 0
    for i in range(n_trials - 1):
        ff_adapt_mp = 0.0
        ff_adapt_ep = 0.0
        ff_adapt_mp2 = 0.0
        ff_adapt_ep2 = 0.0

        # TODO: current_context is set by the uncertainty at EP on the previous trial.
        # If fully uncertain, then the current context is maintained
        # TODO: This should work for matched or EP-only conditions but not for
        # MP-only conditions
        # TODO: We will eventually need to sort out what to do with the
        # mismatched MP / EP conditions etc.
        # TODO: COIN model or generative whatever may be the most natural
        # solution here.
        # TODO: A simpler method may be to include generalisation between
        # contexts at the update phase and mixing at the output stage.

        if sig_ep[i - 1] != 4:
            current_context = sig_ep[i - 1] - 1

        # NOTE: context-specific slow and global fast
        yff[i] = xff[current_context, i] + xff2[i]

        yfb[i] = 0.0
        y[i] = yff[i] + yfb[i]

        if sig_mp[i] != 4:
            delta_mp[i] = 0.0 - (y[i] + r[i])
            yfb[i] = xfb[i] * delta_mp[i] * gamma_fbint[sig_mp[i] - 1]
            ff_adapt_mp = alpha_ff * delta_mp[i]
            ff_adapt_mp2 = alpha_ff2 * delta_mp[i]
        else:
            delta_mp[i] = 0.0
            yfb[i] = gamma_fbint[sig_mp[i] - 1]

        # midpoint to endpoint
        y[i] = yff[i] + yfb[i]

        if sig_ep[i] != 4:
            delta_ep[i] = 0.0 - (y[i] + r[i])
            ff_adapt_ep = alpha_ff * (delta_ep[i] - yfb[i])
            ff_adapt_ep2 = alpha_ff2 * (delta_ep[i] - yfb[i])
        else:
            delta_ep[i] = 0.0

        # update fb state
        xfb[i + 1] = beta_fb * xfb[i] + alpha_fb * delta_ep[i]

        # NOTE: context-specific slow
        xff[:, i + 1] = beta_ff * xff[:, i]
        xff[current_context, i + 1] += gamma_ff[sig_mp[i] - 1] * ff_adapt_mp
        xff[current_context, i + 1] += gamma_ff[sig_ep[i] - 1] * ff_adapt_ep
        xff[current_context, i + 1] += bias_ff

        # NOTE: context-specific fast
        # xff2[:, i + 1] = beta_ff2 * xff2[:, i]
        # xff2[current_context, i + 1] += gamma_ff[sig_mp[i] - 1] * ff_adapt_mp2
        # xff2[current_context, i + 1] += gamma_ff[sig_ep[i] - 1] * ff_adapt_ep2
        # xff2[current_context, i + 1] += bias_ff2

        # NOTE: global fast
        xff2[i + 1] = beta_ff2 * xff2[i]
        xff2[i + 1] += ff_adapt_mp2
        xff2[i + 1] += ff_adapt_ep2
        xff2[i + 1] += bias_ff2

        # clip the feedback gain to prevent instability
        xfb = np.clip(xfb, -2, 2)

    return (y, yff, yfb, xff, xfb)


def simulate(params, args):
    alpha_ff = params[0]
    beta_ff = params[1]
    bias_ff = params[2]
    alpha_ff2 = params[3]
    beta_ff2 = params[4]
    bias_ff2 = params[5]
    alpha_fb = params[6]
    beta_fb = params[7]
    xfb_init = params[8]

    gamma_fbint_1 = params[9]
    gamma_fbint_2 = params[10]
    gamma_fbint_3 = params[11]
    gamma_fbint_4 = params[12]
    gamma_fbint = np.array([gamma_fbint_1, gamma_fbint_2, gamma_fbint_3, gamma_fbint_4])

    gamma_ff_1 = params[13]
    gamma_ff_2 = params[14]
    gamma_ff_3 = params[15]
    gamma_ff_4 = params[16]
    gamma_ff = np.array([gamma_ff_1, gamma_ff_2, gamma_ff_3, gamma_ff_4])

    td = params[17]

    r = args[0]
    sig_mp = args[1]
    sig_ep = args[2]
    group = args[3]
    modname = args[4]

    n_trials = r.shape[0]

    delta_ep = np.zeros(n_trials)
    delta_mp = np.zeros(n_trials)
    xff = np.zeros(n_trials)
    xfb = np.zeros(n_trials)
    yff = np.zeros(n_trials)
    yfb = np.zeros(n_trials)
    y = np.zeros(n_trials)

    xff2 = np.zeros(n_trials)

    xfb[0] = xfb_init

    for i in range(n_trials - 1):
        ff_adapt_mp = 0.0
        ff_adapt_ep = 0.0
        ff_adapt_mp2 = 0.0
        ff_adapt_ep2 = 0.0

        # start to midpoint
        yff[i] = xff[i] + xff2[i]

        if "y-aim-scale" in modname:
            if i > 0:
                yff[i] += td * gamma_ff[sig_mp[i - 1] - 1]
                yff[i] += (1 - td) * gamma_ff[sig_ep[i - 1] - 1]

        yfb[i] = 0.0
        y[i] = yff[i] + yfb[i]

        if sig_mp[i] != 4:
            delta_mp[i] = 0.0 - (y[i] + r[i])
            yfb[i] = xfb[i] * delta_mp[i] * gamma_fbint[sig_mp[i] - 1]
            ff_adapt_mp = alpha_ff * delta_mp[i]
            ff_adapt_mp2 = alpha_ff2 * delta_mp[i]
        else:
            delta_mp[i] = 0.0
            yfb[i] = gamma_fbint[sig_mp[i] - 1]

        # midpoint to endpoint
        y[i] = yff[i] + yfb[i]

        if sig_ep[i] != 4:
            delta_ep[i] = 0.0 - (y[i] + r[i])
            ff_adapt_ep = alpha_ff * (delta_ep[i] - yfb[i])
            ff_adapt_ep2 = alpha_ff2 * (delta_ep[i] - yfb[i])
        else:
            delta_ep[i] = 0.0

        # update fb state
        xfb[i + 1] = beta_fb * xfb[i] + alpha_fb * delta_ep[i]

        # some condition files call a 4 a 0
        if sig_ep[i] == 0:
            sig_ep[i] = 4

        # update ff state
        xff[i + 1] = beta_ff * xff[i] + ff_adapt_mp + ff_adapt_ep + bias_ff

        if "error-scale" in modname:
            xff2[i + 1] = beta_ff2 * xff2[i]
            xff2[i + 1] += td * gamma_ff[sig_mp[i] - 1] * ff_adapt_mp2
            xff2[i + 1] += (1 - td) * gamma_ff[sig_ep[i] - 1] * ff_adapt_ep2
            xff2[i + 1] += bias_ff2

        elif "retention-scale" in modname:
            xff2[i + 1] = td * gamma_ff[sig_mp[i] - 1] * beta_ff2 * xff2[i]
            xff2[i + 1] = (1 - td) * gamma_ff[sig_ep[i] - 1] * beta_ff2 * xff2[i]
            xff2[i + 1] += ff_adapt_mp2 + ff_adapt_ep2
            xff2[i + 1] += bias_ff2

        elif "bias-scale" in modname:
            xff2[i + 1] = beta_ff2 * xff2[i]
            xff2[i + 1] += ff_adapt_mp2 + ff_adapt_ep2
            xff2[i + 1] += td * gamma_ff[sig_mp[i] - 1] * bias_ff2
            xff2[i + 1] += (1 - td) * gamma_ff[sig_ep[i] - 1] * bias_ff2

        elif "x-aim-scale" in modname:
            xff2[i + 1] = td * gamma_ff[sig_mp[i] - 1]
            xff2[i + 1] += (1 - td) * gamma_ff[sig_ep[i] - 1]

        # clip the feedback gain to prevent instability
        xfb = np.clip(xfb, -2, 2)

    return (y, yff, yfb, xff, xfb, xff2)


def compute_aic(rsq, n, k):
    # aic = 2 * k + n * np.log(sse / n)
    aic = n * np.log(1 - rsq) + k * 2
    return aic


def compute_bic(rsq, n, k):
    # bic = np.log(n) * k + n * np.log(sse / n)
    bic = n * np.log(1 - rsq) + k * np.log(n)
    return bic


def prep_data_regression(d):
    # some conditions use sig_ep == 4 and others sig_ep == 0
    d.loc[d["sig_ep"] == 0, "sig_ep"] = 4

    d["delta_ha_init"] = np.diff(d["ha_init"].to_numpy(), prepend=0)
    d["fb_int"] = d["ha_end"] - d["ha_init"]

    d["error_mp"] = d["ha_init"].to_numpy() + d["rot"].to_numpy()
    d["error_ep"] = d["ha_end"].to_numpy() + d["rot"].to_numpy()

    d["ha_init_prev"] = np.roll(d["ha_init"], 1)
    d["error_mp_prev"] = np.roll(d["error_mp"], 1)
    d["error_ep_prev"] = np.roll(d["error_ep"], 1)
    d["sig_mp_prev"] = np.roll(d["sig_mp"], 1)
    d["sig_ep_prev"] = np.roll(d["sig_ep"], 1)
    d["sig_mp_prev_2"] = np.roll(d["sig_mp"], 2)
    d["sig_ep_prev_2"] = np.roll(d["sig_ep"], 2)

    d["sig_mp"] = d["sig_mp"].astype("category")
    d["sig_ep"] = d["sig_ep"].astype("category")
    d["sig_mp_prev"] = d["sig_mp_prev"].astype("category")
    d["sig_ep_prev"] = d["sig_ep_prev"].astype("category")
    d["sig_mp_prev_2"] = d["sig_mp_prev_2"].astype("category")
    d["sig_ep_prev_2"] = d["sig_ep_prev_2"].astype("category")

    d["sig_mpep"] = list(zip(d["sig_mp"], d["sig_ep"]))
    d["sig_mpep"] = d["sig_mpep"].astype("category")
    d["sig_mpep_prev"] = list(zip(d["sig_mp_prev"], d["sig_ep_prev"]))
    d["sig_mpep_prev"] = d["sig_mpep_prev"].astype("category")
    d["sig_mpep_prev_2"] = list(zip(d["sig_mp_prev_2"], d["sig_ep_prev_2"]))
    d["sig_mpep_prev_2"] = d["sig_mpep_prev_2"].astype("category")

    # TODO: these are for small perturbation conditions
    # add columns for hit / miss analysis
    d["hit_mp"] = "miss"
    d.loc[(d["sig_mp"] == 1) & (np.abs(d["error_mp"]) < 2.86), "hit_mp"] = "hit"
    d.loc[(d["sig_mp"] == 2) & (np.abs(d["error_mp"]) < 17.45), "hit_mp"] = "hit"
    d.loc[(d["sig_mp"] == 3) & (np.abs(d["error_mp"]) < 36.86), "hit_mp"] = "hit"
    d.loc[(d["sig_mp"] == 4), "hit_mp"] = "none"

    d["hit_ep"] = "miss"
    d.loc[(d["sig_ep"] == 1) & (np.abs(d["error_ep"]) < 2.86), "hit_ep"] = "hit"
    d.loc[(d["sig_ep"] == 2) & (np.abs(d["error_ep"]) < 10.05), "hit_ep"] = "hit"
    d.loc[(d["sig_ep"] == 3) & (np.abs(d["error_ep"]) < 18.88), "hit_ep"] = "hit"
    d.loc[(d["sig_ep"] == 4), "hit_ep"] = "none"

    d["hit_mp_prev"] = np.roll(d["hit_mp"], 1)
    d["hit_ep_prev"] = np.roll(d["hit_ep"], 1)

    d = d.iloc[:-1]

    # NOTE: remove outlier (there is only one data point removed)
    d = d.loc[d["error_ep_prev"] > -15]

    return d


def inspect_hit_miss(d):
    print("Hit / Miss")
    dd = (
        d.groupby(["sig_mp"])[["hit_mp"]]
        .value_counts()
        .to_frame(name="N")
        .reset_index()
    )
    print(dd)
    dd = (
        d.groupby(["sig_ep"])[["hit_ep"]]
        .value_counts()
        .to_frame(name="N")
        .reset_index()
    )
    print(dd)


def inspect_inf(d):
    # NOTE: also do a washout analysis (already done
    # somewhere) to look for abrupt jumps in the individual
    # ahnd angles. I don't think we see any so it doens't
    # look a lot like a strategy.

    # NOTE: it could also be useful to look at the
    # savings-like paradigms from chris's corpus of data.
    print("inf types")

    dd = d[d["sig_mpep_prev"] == (4, 4)]
    dd = d[d["sig_mpep_prev"] == (4, 4)][d["sig_mpep_prev_2"] == (4, 4)]

    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False, figsize=(4, 4))
    sns.scatterplot(
        data=dd,
        x="trial",
        y="ha_init",
        hue="sig_mpep_prev_2",
        s=100,
        legend=True,
        ax=ax[0, 0],
    )
    sns.scatterplot(
        data=dd,
        x="trial",
        y="ha_init",
        hue="sig_mpep_prev",
        s=40,
        legend=False,
        ax=ax[0, 0],
    )
    ax[0, 0].set_xticks(dd["trial"].unique())
    ax[0, 0].set_xticklabels(dd["trial"].unique(), rotation=90)
    plt.show()


def fig_reg_ff(d, g, res_1, res_2):
    if d.sig_ep.unique().shape[0] == 1:
        fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 4))
    else:
        fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False, figsize=(12, 4))

    sns.violinplot(data=d, x="sig_mpep_prev", y="delta_ha_init", ax=ax[0, 0])
    if g not in (15, 16):
        ax[0, 0].set_xticklabels(
            labels=[
                r"$\sigma_{L}$",
                r"$\sigma_{M}$",
                r"$\sigma_{H}$",
                r"$\sigma_{\infty}$",
            ],
            rotation=0,
        )
    else:
        ax[0, 0].set_xticklabels(
            labels=[
                r"$\sigma_{LL}$",
                r"$\sigma_{LH}$",
                r"$\sigma_{HL}$",
                r"$\sigma_{HH}$",
            ],
            rotation=0,
        )

    for smp in np.sort(d.sig_mpep_prev.unique()):
        sns.regplot(
            data=d.loc[d["sig_mpep_prev"] == smp],
            x="error_mp_prev",
            y="delta_ha_init",
            label=str(smp),
            scatter_kws={"alpha": 0.25},
            robust=False,
            ax=ax[0, 1],
        )

    if d.sig_ep.unique().shape[0] != 1:
        for sep in np.sort(d.sig_mpep_prev.unique()):
            sns.regplot(
                data=d.loc[d["sig_mpep_prev"] == sep],
                x="error_ep_prev",
                y="delta_ha_init",
                label=str(sep),
                scatter_kws={"alpha": 0.25},
                robust=False,
                ax=ax[0, 2],
            )

    ax[0, 0].set_ylabel("Delta Initial movement vector\n(degrees)")
    ax[0, 1].set_ylabel("")
    if d.sig_ep.unique().shape[0] != 1:
        ax[0, 2].set_ylabel("")

    ax[0, 0].set_xlabel("")
    ax[0, 1].set_xlabel("Error MP")
    if d.sig_ep.unique().shape[0] != 1:
        ax[0, 2].set_xlabel("Error EP")

    labels = list(string.ascii_lowercase)
    for i, x in enumerate(ax.flatten()):
        x.annotate(text=labels[i], xy=(-0.0, 1.1), xycoords="axes fraction", size=14)

    plt.tight_layout()
    plt.savefig("../figures/fig_reg_ff_grp_" + str(g) + ".tif")
    plt.savefig("../figures/fig_reg_ff_grp_" + str(g) + ".pdf")
    plt.close()


def fig_reg_ff_mod(d, g, res_1, res_2):
    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 4))

    obs = d["ha_init"].to_numpy()
    pred = res_1.model.predict(res_1.params, res_1.model.exog)
    ax[0, 0].plot(obs, label="Observed")
    ax[0, 0].plot(pred, label="Predicted")
    ax[0, 0].legend()

    s = 5
    y = np.arange(0, s * res_1.params.shape[0], s)
    ax[0, 1].axvline(x=0, color="gray")
    ax[0, 1].errorbar(
        x=res_1.params,
        y=y - 0.5,
        xerr=np.diff(res_1.conf_int().to_numpy()).flatten() / 2,
        fmt=".",
        color="C0",
        label="Model 1 Beta Coefficients",
    )
    ax[0, 1].errorbar(
        x=res_2.params,
        y=y[:-1] + 0.5,
        xerr=np.diff(res_2.conf_int().to_numpy()).flatten() / 2,
        fmt=".",
        color="C1",
        label="Model 2 Beta Coefficients",
    )
    ax[0, 1].set_yticks(y)
    if g in (20, 18):
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"($\sigma_{\infty} - \sigma_{H}$):$\delta_{MP}$",
                    r"($\sigma_{H} - \sigma_{M}$):$\delta_{MP}$",
                    r"($\sigma_{M} - \sigma_{L}$):$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{\infty} - \sigma_{H}$",
                    r"$\sigma_{H} - \sigma_{M}$",
                    r"$\sigma_{M} - \sigma_{L}$",
                    r"Intercept",
                ]
            )
        )

    elif g in (15, 16):
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"($\sigma_{HH} - \sigma_{HL}$):$\delta_{EP}$",
                    r"($\sigma_{HL} - \sigma_{LH}$):$\delta_{EP}$",
                    r"($\sigma_{LH} - \sigma_{LL}$):$\delta_{EP}$",
                    r"$\delta_{EP}$",
                    r"($\sigma_{HH} - \sigma_{HL}$):$\delta_{MP}$",
                    r"($\sigma_{HL} - \sigma_{LH}$):$\delta_{MP}$",
                    r"($\sigma_{LH} - \sigma_{LL}$):$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{HH} - \sigma_{HL}$",
                    r"$\sigma_{HL} - \sigma_{LH}$",
                    r"$\sigma_{LH} - \sigma_{LL}$",
                    r"Intercept",
                ]
            )
        )

    else:
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"($\sigma_{\infty} - \sigma_{H}$):$\delta_{EP}$",
                    r"($\sigma_{H} - \sigma_{M}$):$\delta_{EP}$",
                    r"($\sigma_{M} - \sigma_{L}$):$\delta_{EP}$",
                    r"$\delta_{EP}$",
                    r"($\sigma_{\infty} - \sigma_{H}$):$\delta_{MP}$",
                    r"($\sigma_{H} - \sigma_{M}$):$\delta_{MP}$",
                    r"($\sigma_{M} - \sigma_{L}$):$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{\infty} - \sigma_{H}$",
                    r"$\sigma_{H} - \sigma_{M}$",
                    r"$\sigma_{M} - \sigma_{L}$",
                    r"Intercept",
                ]
            )
        )
    # beta_labels = res_1.params.index
    ax[0, 1].set_yticklabels(beta_labels)
    ax[0, 1].legend(loc=(0, -0.4), ncols=1)

    labels = list(string.ascii_lowercase)
    for i, x in enumerate(ax.flatten()):
        x.annotate(text=labels[i], xy=(-0.0, 1.1), xycoords="axes fraction", size=14)

    plt.tight_layout()
    plt.savefig("../figures/fig_reg_ff_mod_grp_" + str(g) + ".tif")
    plt.savefig("../figures/fig_reg_ff_mod_grp_" + str(g) + ".pdf")
    plt.close()


def fig_reg_fb_mod(d, g, res_1, res_2):
    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 4))

    obs = d["ha_end"].to_numpy()
    pred = res_1.model.predict(res_1.params, res_1.model.exog)
    ax[0, 0].plot(obs, label="Observed")
    ax[0, 0].plot(pred, label="Predicted")
    ax[0, 0].legend()

    s = 5
    y = np.arange(0, s * res_1.params.shape[0], s)
    ax[0, 1].axvline(x=0, color="gray")
    ax[0, 1].errorbar(
        x=res_1.params,
        y=y - 0.5,
        xerr=np.diff(res_1.conf_int().to_numpy()).flatten(),
        fmt=".",
        label="Model 1 Beta Coefficients",
    )
    ax[0, 1].errorbar(
        x=res_2.params,
        y=y[:-1] + 0.5,
        xerr=np.diff(res_2.conf_int().to_numpy()).flatten(),
        fmt=".",
        label="Model 2 Beta Coefficients",
    )
    ax[0, 1].set_yticks(y)
    if g in (15, 16):
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"$\sigma_{MP}$:$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{MP}$",
                    r"$\beta_{0}$",
                ]
            )
        )
    else:
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"($\sigma_{\infty} - \sigma_{H}$):$\delta_{MP}$",
                    r"($\sigma_{H} - \sigma_{M}$):$\delta_{MP}$",
                    r"($\sigma_{M} - \sigma_{L}$):$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{\infty} - \sigma_{H}$",
                    r"$\sigma_{H} - \sigma_{M}$",
                    r"$\sigma_{M} - \sigma_{L}$",
                    r"$\beta_{0}$",
                ]
            )
        )

    # beta_labels = res_1.params.index
    ax[0, 1].set_yticklabels(beta_labels)
    ax[0, 1].legend(loc=(0, -0.4), ncols=1)

    labels = list(string.ascii_lowercase)
    for i, x in enumerate(ax.flatten()):
        x.annotate(text=labels[i], xy=(-0.0, 1.1), xycoords="axes fraction", size=14)

    plt.tight_layout()
    plt.savefig("../figures/fig_reg_fb_mod_grp_" + str(g) + ".tif")
    plt.savefig("../figures/fig_reg_fb_mod_grp_" + str(g) + ".pdf")
    plt.close()


def fig_reg_fb(d, g, res_1, res_2):
    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False, figsize=(8, 4))

    sns.violinplot(data=d, x="sig_mp", y="fb_int", ax=ax[0, 0])
    if g not in (15, 16):
        ax[0, 0].set_xticklabels(
            labels=[
                r"$\sigma_{L}$",
                r"$\sigma_{M}$",
                r"$\sigma_{H}$",
                r"$\sigma_{\infty}$",
            ],
            rotation=0,
        )
    else:
        ax[0, 0].set_xticklabels(
            labels=[r"$\sigma_{L \cdot}$", r"$\sigma_{H \cdot}$"], rotation=0
        )

    for smp in np.sort(d.sig_mp.unique()):
        sns.regplot(
            data=d.loc[d["sig_mp"] == smp],
            x="error_mp",
            y="fb_int",
            label=str(smp),
            scatter_kws={"alpha": 0.25},
            robust=False,
            ax=ax[0, 1],
        )

    ax[0, 0].set_ylabel("Midpoint feedback integration\n(degrees)")
    ax[0, 1].set_ylabel("")

    [x.set_xlabel("Trial") for x in ax[:, 0]]

    ax[0, 0].set_xlabel("")
    ax[0, 1].set_xlabel("")
    ax[0, 1].set_xlabel("Error MP")

    labels = list(string.ascii_lowercase)
    for i, x in enumerate(ax.flatten()):
        x.annotate(text=labels[i], xy=(-0.0, 1.1), xycoords="axes fraction", size=14)

    plt.tight_layout()
    plt.savefig("../figures/fig_reg_fb_grp_" + str(g) + ".tif")
    plt.savefig("../figures/fig_reg_fb_grp_" + str(g) + ".pdf")
    plt.close()


def fit_reg_ff_1(d):
    if d.sig_ep.unique().shape[0] == 1:
        mod_formula = "ha_init ~ "
        mod_formula += "C(sig_mpep_prev, Diff) * error_mp_prev + "
        mod_formula += "np.log(trial) + "
        mod_formula += "1"

    else:
        mod_formula = "ha_init ~ "
        mod_formula += "C(sig_mpep_prev, Diff) * error_mp_prev + "
        mod_formula += "C(sig_mpep_prev, Diff) * error_ep_prev + "
        mod_formula += "np.log(trial) + "
        mod_formula += "1"

    # NOTE: statsmodels
    mod = smf.ols(mod_formula, data=d)
    res_sm = mod.fit()

    # NOTE: pingouin
    y, X = patsy.dmatrices(mod_formula, d, return_type="dataframe")
    res_pg = pg.linear_regression(X, np.squeeze(y.to_numpy()), relimp=True)

    return res_sm, res_pg


def fit_reg_ff_2(d):
    if d.sig_ep.unique().shape[0] == 1:
        mod_formula = "delta_ha_init ~ "
        mod_formula += "C(sig_mpep_prev, Diff) * error_mp_prev + "
        mod_formula += "1"

    else:
        mod_formula = "delta_ha_init ~ "
        mod_formula += "C(sig_mpep_prev, Diff) * error_mp_prev + "
        mod_formula += "C(sig_mpep_prev, Diff) * error_ep_prev + "
        mod_formula += "1"

    # NOTE: statsmodels
    mod = smf.ols(mod_formula, data=d)
    res_sm = mod.fit()

    # NOTE: pingouin
    y, X = patsy.dmatrices(mod_formula, d, return_type="dataframe")
    res_pg = pg.linear_regression(X, np.squeeze(y.to_numpy()), relimp=True)

    return res_sm, res_pg


def fit_reg_fb_1(d):
    mod_formula = "ha_end ~ "
    mod_formula += "C(sig_mp, Diff) * error_mp + "
    mod_formula += "np.log(trial) + "
    mod_formula += "1"

    # NOTE: statsmodels
    mod = smf.ols(mod_formula, data=d)
    res_sm = mod.fit()

    # NOTE: pingouin
    y, X = patsy.dmatrices(mod_formula, d, return_type="dataframe")
    res_pg = pg.linear_regression(X, np.squeeze(y.to_numpy()), relimp=True)

    return res_sm, res_pg


def fit_reg_fb_2(d):
    mod_formula = "fb_int ~ "
    mod_formula += "C(sig_mp, Diff) * error_mp + "
    mod_formula += "1"

    # NOTE: statsmodels
    mod = smf.ols(mod_formula, data=d)

    res_sm = mod.fit()

    # NOTE: pingouin
    y, X = patsy.dmatrices(mod_formula, d, return_type="dataframe")
    res_pg = pg.linear_regression(X, np.squeeze(y.to_numpy()), relimp=True)

    return res_sm, res_pg


def inspect_regression(g):
    d = load_all_data()
    # d = d.sort_values(["group", "subject", "trial"])
    d = d.loc[d["group"] == g]
    d = d.loc[np.isin(d["phase"], ["adaptation"])]
    d = (
        d.groupby(["trial", "sig_mp", "sig_ep"])[["ha_init", "ha_end", "rot"]]
        .mean()
        .reset_index()
    )
    d = prep_data_regression(d)

    res_ff_1_sm, res_ff_1_pg = fit_reg_ff_1(d)
    res_ff_2_sm, res_ff_2_pg = fit_reg_ff_2(d)
    fig_reg_ff(d, g, res_ff_1_sm, res_ff_2_sm)
    fig_reg_ff_mod(d, g, res_ff_1_sm, res_ff_2_sm)

    res_fb_1_sm, res_fb_1_pg = fit_reg_fb_1(d)
    res_fb_2_sm, res_fb_2_pg = fit_reg_fb_2(d)
    fig_reg_fb(d, g, res_fb_1_sm, res_fb_2_sm)
    fig_reg_fb_mod(d, g, res_fb_1_sm, res_fb_2_sm)

    # stats_to_file_sm(g, res_ff_1, res_ff_2, res_fb_1, res_fb_2)
    stats_to_file_pg(g, res_ff_1_pg, res_ff_2_pg, res_fb_1_pg, res_fb_2_pg)

    inspect_hit_miss(d)

    # TODO: inspect infinite uncertainty trials
    # TODO: inspect delta hand angle as function of uncertainty and error


def stats_to_file_pg(g, res_ff_1, res_ff_2, res_fb_1, res_fb_2):
    res_ff_1.rename(
        columns={"CI[2.5%]": "CI[2.5\%]", "CI[97.5%]": "CI[97.5\%]"}, inplace=True
    )

    res_ff_2.rename(
        columns={"CI[2.5%]": "CI[2.5\%]", "CI[97.5%]": "CI[97.5\%]"}, inplace=True
    )

    res_fb_1.rename(
        columns={"CI[2.5%]": "CI[2.5\%]", "CI[97.5%]": "CI[97.5\%]"}, inplace=True
    )

    res_fb_2.rename(
        columns={"CI[2.5%]": "CI[2.5\%]", "CI[97.5%]": "CI[97.5\%]"}, inplace=True
    )

    f = open("../stats_tables/stats_table_regression_group_" + str(g) + ".tex", "w")

    if g in (20, 18):
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"($\sigma_{\infty} - \sigma_{H}$):$\delta_{MP}$",
                    r"($\sigma_{H} - \sigma_{M}$):$\delta_{MP}$",
                    r"($\sigma_{M} - \sigma_{L}$):$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{\infty} - \sigma_{H}$",
                    r"$\sigma_{H} - \sigma_{M}$",
                    r"$\sigma_{M} - \sigma_{L}$",
                    r"Intercept",
                ]
            )
        )

    elif g in (15, 16):
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"($\sigma_{HH} - \sigma_{HL}$):$\delta_{EP}$",
                    r"($\sigma_{HL} - \sigma_{LH}$):$\delta_{EP}$",
                    r"($\sigma_{LH} - \sigma_{LL}$):$\delta_{EP}$",
                    r"$\delta_{EP}$",
                    r"($\sigma_{HH} - \sigma_{HL}$):$\delta_{MP}$",
                    r"($\sigma_{HL} - \sigma_{LH}$):$\delta_{MP}$",
                    r"($\sigma_{LH} - \sigma_{LL}$):$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{HH} - \sigma_{HL}$",
                    r"$\sigma_{HL} - \sigma_{LH}$",
                    r"$\sigma_{LH} - \sigma_{LL}$",
                    r"Intercept",
                ]
            )
        )

    else:
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"($\sigma_{\infty} - \sigma_{H}$):$\delta_{EP}$",
                    r"($\sigma_{H} - \sigma_{M}$):$\delta_{EP}$",
                    r"($\sigma_{M} - \sigma_{L}$):$\delta_{EP}$",
                    r"$\delta_{EP}$",
                    r"($\sigma_{\infty} - \sigma_{H}$):$\delta_{MP}$",
                    r"($\sigma_{H} - \sigma_{M}$):$\delta_{MP}$",
                    r"($\sigma_{M} - \sigma_{L}$):$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{\infty} - \sigma_{H}$",
                    r"$\sigma_{H} - \sigma_{M}$",
                    r"$\sigma_{M} - \sigma_{L}$",
                    r"Intercept",
                ]
            )
        )

    res_ff_1["names"] = beta_labels
    res_ff_2["names"] = beta_labels[:-1]

    f.write("\n\n")
    f.write(
        res_ff_1[
            ["names", "coef", "se", "T", "pval", "CI[2.5\%]", "CI[97.5\%]", "relimp"]
        ]
        .round(2)
        .to_latex(index=False, escape=False)
    )

    f.write("\n\n")
    f.write(
        res_ff_2[
            ["names", "coef", "se", "T", "pval", "CI[2.5\%]", "CI[97.5\%]", "relimp"]
        ]
        .round(2)
        .to_latex(index=False, escape=False)
    )

    if g in (15, 16):
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"$\sigma_{MP}$:$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{MP}$",
                    r"$\beta_{0}$",
                ]
            )
        )
    else:
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"($\sigma_{\infty} - \sigma_{H}$):$\delta_{MP}$",
                    r"($\sigma_{H} - \sigma_{M}$):$\delta_{MP}$",
                    r"($\sigma_{M} - \sigma_{L}$):$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{\infty} - \sigma_{H}$",
                    r"$\sigma_{H} - \sigma_{M}$",
                    r"$\sigma_{M} - \sigma_{L}$",
                    r"$\beta_{0}$",
                ]
            )
        )

    res_fb_1["names"] = beta_labels
    res_fb_2["names"] = beta_labels[:-1]

    # f.write('\n\n')
    # f.write(res_fb_1[[
    #     'names', 'coef', 'se', 'T', 'pval', 'CI[2.5\%]', 'CI[97.5\%]', 'relimp'
    # ]].round(2).to_latex(index=False, escape=False))

    f.write("\n\n")
    f.write(
        res_fb_2[
            ["names", "coef", "se", "T", "pval", "CI[2.5\%]", "CI[97.5\%]", "relimp"]
        ]
        .round(2)
        .to_latex(index=False, escape=False)
    )

    f.close()


def stats_to_file_sm(g, res_ff_1, res_ff_2, res_fb_1, res_fb_2):
    f = open("../stats_tables/stats_table_regression_group_" + str(g) + ".tex", "w")

    if res_ff_1.params.shape[0] == 9:
        beta_labels = list(
            reversed(
                [
                    r"log(Trial)",
                    r"($\sigma_{\infty} - \sigma_{H}$):$\delta_{MP}$",
                    r"($\sigma_{H} - \sigma_{M}$):$\delta_{MP}$",
                    r"($\sigma_{M} - \sigma_{L}$):$\delta_{MP}$",
                    r"$\delta_{MP}$",
                    r"$\sigma_{\infty} - \sigma_{H}$",
                    r"$\sigma_{H} - \sigma_{M}$",
                    r"$\sigma_{M} - \sigma_{L}$",
                    r"Intercept",
                ]
            )
        )
    else:
        beta_labels = list(
            reversed(
                [
                    "log(Trial)",
                    "(sig 4 - sig 3):Err EP",
                    "(sig 3 - sig 2):Err EP",
                    "(sig 2 - sig 1):Err EP",
                    "Err EP",
                    "(sig 4 - sig 3):Err MP",
                    "(sig 3 - sig 2):Err MP",
                    "(sig 2 - sig 1):Err MP",
                    "Err MP",
                    "sig 4 - sig 3",
                    "sig 3 - sig 2",
                    "sig 2 - sig 1",
                    "$\beta_{0}$",
                ]
            )
        )

    f.write(res_ff_1.summary(xname=beta_labels).tables[0].as_latex_tabular())
    f.write(res_ff_1.summary(xname=beta_labels).tables[1].as_latex_tabular())
    f.write(res_ff_1.summary(xname=beta_labels).tables[2].as_latex_tabular())
    f.write("\n\n\n\n")

    f.write(res_ff_2.summary(xname=beta_labels[:-1]).tables[0].as_latex_tabular())
    f.write(res_ff_2.summary(xname=beta_labels[:-1]).tables[1].as_latex_tabular())
    f.write(res_ff_2.summary(xname=beta_labels[:-1]).tables[2].as_latex_tabular())
    f.write("\n\n\n\n")

    if g in (15, 16):
        beta_labels = list(
            reversed(["log(Trial)", "sig MP:Err MP", "Err MP", "sig MP", "Intercept"])
        )
    else:
        beta_labels = list(
            reversed(
                [
                    "log(Trial)",
                    "(sig 4 - sig 3):Err MP",
                    "(sig 3 - sig 2):Err MP",
                    "(sig 2 - sig 1):Err MP",
                    "Err MP",
                    "sig 4 - sig 3",
                    "sig 3 - sig 2",
                    "sig 2 - sig 1",
                    "Intercept",
                ]
            )
        )

    f.write(res_fb_1.summary(xname=beta_labels).tables[0].as_latex_tabular())
    f.write(res_fb_1.summary(xname=beta_labels).tables[1].as_latex_tabular())
    f.write(res_fb_1.summary(xname=beta_labels).tables[2].as_latex_tabular())
    f.write("\n\n\n\n")

    f.write(res_fb_2.summary(xname=beta_labels[:-1]).tables[0].as_latex_tabular())
    f.write(res_fb_2.summary(xname=beta_labels[:-1]).tables[1].as_latex_tabular())
    f.write(res_fb_2.summary(xname=beta_labels[:-1]).tables[2].as_latex_tabular())

    f.close()


def inspect_washout(g):
    d = load_all_data()
    d = d.loc[d["group"] == g]
    d.loc[d["phase"] == "washout", "sig_mp"] = 5
    d.loc[d["phase"] == "washout", "sig_ep"] = 5
    d = d.loc[np.isin(d["phase"], ["adaptation", "washout"])]
    d = prep_data_regression(d)

    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(6, 6))
    sns.lineplot(
        data=d[d["phase"] == "washout"],
        x="trial",
        y="ha_init",
        hue="subject",
        estimator=None,
        ax=ax[0, 0],
    )
    plt.show()

    dd = d.groupby(["subject"]).apply(compute_washout_diff).reset_index()
    dd["sig_mpep_prev"] = dd["sig_mpep_prev"].astype("category")

    fig_washout(d, dd, g)

    dd["sig_mpep_prev"] = pd.factorize(dd["sig_mpep_prev"])[0]

    res = pg.rm_anova(
        data=dd, dv="diff", within="sig_mpep_prev", subject="subject", correction=True
    )
    res.style.to_latex("../stats_tables/stats_table_washout_group_" + str(g) + "_1.tex")

    res = pg.pairwise_tests(
        data=dd, dv="diff", within="sig_mpep_prev", subject="subject", padjust="bonf"
    )
    res = res[["A", "B", "T", "dof", "p-corr", "hedges"]].round(2)
    res.style.format(precision=2).to_latex(
        "../stats_tables/stats_table_washout_group_" + str(g) + "_2.tex"
    )


def compute_washout_diff(d):
    n_adapt = 10
    n_wash = 3

    wash = d.loc[
        (d["phase"] == "washout") & (d["trial"] < 180 + n_wash + 1), "ha_init"
    ].mean()

    x = {"sig_mpep_prev": [], "diff": []}
    for smpep in d["sig_mpep_prev"].unique():
        if smpep != (5, 5):
            adapt = d.loc[
                (d["phase"] == "adaptation")
                & (d["trial"] > 180 - n_adapt)
                & (d["sig_mpep_prev"] == smpep),
                "ha_init",
            ].mean()

            x["sig_mpep_prev"].append(smpep)
            x["diff"].append(adapt - wash)

    x = pd.DataFrame(x)

    return x


def fig_washout(d, dd, g):
    ddd = d.copy()
    d = d.groupby(["trial", "sig_mpep_prev", "phase"]).mean()

    exp = d.group.unique()
    exp = exp[~np.isnan(exp)][0]

    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 4))

    sns.lineplot(
        data=ddd,
        x="trial",
        y="ha_init",
        style="phase",
        hue="sig_mpep_prev",
        markers=True,
        legend=False,
        ax=ax[0, 0],
    )
    ax[0, 0].set_xlabel("Trial")
    ax[0, 0].set_ylabel("Initial movement vector\n(degrees)")
    axins = inset_axes(
        ax[0, 0], width="20%", height="20%", loc="upper right", borderpad=1.0
    )
    sns.barplot(data=dd, x="sig_mpep_prev", y="diff", ax=axins)
    if exp not in (15, 16):
        axins.set_xticklabels(
            labels=[
                r"$\sigma_{L}$",
                r"$\sigma_{M}$",
                r"$\sigma_{H}$",
                r"$\sigma_{\infty}$",
            ],
            rotation=90,
        )
    elif exp in (15, 16):
        axins.set_xticklabels(
            labels=[
                r"$\sigma_{LL}$",
                r"$\sigma_{LH}$",
                r"$\sigma_{HL}$",
                r"$\sigma_{HH}$",
            ],
            rotation=90,
        )
    axins.set_xlabel("")
    axins.set_ylabel("")

    sns.scatterplot(
        data=d,
        x="trial",
        y="ha_init",
        hue="error_mp_prev",
        style="phase",
        palette="cool",
        markers=True,
        legend=False,
        ax=ax[0, 1],
    )
    ax[0, 1].set_xlabel("Trial")
    ax[0, 1].set_ylabel("")
    norm = plt.Normalize(d["error_mp_prev"].min(), d["error_mp_prev"].max())
    sm = plt.cm.ScalarMappable(cmap="cool", norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm,
        cax=ax[0, 1].inset_axes([0.1, 0.1, 0.6, 0.05]),
        orientation="horizontal",
        ticks=np.round(np.arange(d.error_mp_prev.min(), d.error_mp_prev.max(), 5)),
    )
    ax[0, 1].annotate(
        text="Error MP", xy=(0.3, 0.175), xycoords="axes fraction", rotation=0
    )

    sns.lineplot(
        data=d,
        x="trial",
        y="ha_init",
        hue="error_ep_prev",
        style="phase",
        palette="cool",
        markers=True,
        legend=False,
        ax=ax[0, 2],
    )
    ax[0, 2].set_xlabel("Trial")
    ax[0, 2].set_ylabel("")
    norm = plt.Normalize(d["error_ep_prev"].min(), d["error_ep_prev"].max())
    sm = plt.cm.ScalarMappable(cmap="cool", norm=norm)
    sm.set_array([])
    fig.colorbar(
        sm,
        cax=ax[0, 2].inset_axes([0.1, 0.1, 0.6, 0.05]),
        orientation="horizontal",
        ticks=np.round(np.arange(d.error_mp_prev.min(), d.error_mp_prev.max(), 5)),
    )
    ax[0, 2].annotate(
        text="Error EP", xy=(0.3, 0.175), xycoords="axes fraction", rotation=0
    )

    labels = list(string.ascii_lowercase)
    for i, x in enumerate(ax.flatten()):
        x.annotate(text=labels[i], xy=(-0.0, 1.1), xycoords="axes fraction", size=14)

    plt.tight_layout()
    plt.savefig("../figures/fig_wash_grp_" + str(g) + ".tif")
    plt.savefig("../figures/fig_wash_grp_" + str(g) + ".pdf")
    plt.close()


def inspect_model_stats():
    d = pd.read_csv("../fits/fit_summary.csv")

    d["model_class"] = "undefined"
    d.loc[d["model"].str.contains("error-scale"), "model_class"] = "error-scale"
    d.loc[d["model"].str.contains("retention-scale"), "model_class"] = "retention-scale"
    d.loc[d["model"].str.contains("bias-scale"), "model_class"] = "bias-scale"
    d.loc[d["model"].str.contains("x-aim-scale"), "model_class"] = "x-aim-scale"
    d.loc[d["model"].str.contains("y-aim-scale"), "model_class"] = "y-aim-scale"

    d["model_state"] = "undefined"
    d.loc[d["model"].str.contains("one-state"), "model_state"] = "one-state"
    d.loc[d["model"].str.contains("two-state"), "model_state"] = "two-state"

    d["model_bound"] = "normal"
    d.loc[d["model"].str.contains("non-neg"), "model_bound"] = "non-neg"

    res_state = []
    res_class = []
    res_bound = []
    for g in [20, 19, 15]:
        dd = d[d["group"] == g]
        res_state.append(
            pg.pairwise_tests(
                data=dd,
                dv="bic",
                within="model_state",
                subject="subject",
                alternative="greater",
                padjust="bonf",
                return_desc=True,
            )
        )

        res_class.append(
            pg.pairwise_tests(
                data=dd[dd["model_state"] != "one-state"],
                dv="bic",
                within="model_class",
                subject="subject",
                alternative="greater",
                padjust="bonf",
                return_desc=True,
            )
        )

        res_bound.append(
            pg.pairwise_tests(
                data=dd[dd["model_state"] != "one-state"],
                dv="bic",
                within="model_bound",
                subject="subject",
                alternative="greater",
                padjust="bonf",
                return_desc=True,
            )
        )

    # print()
    # print(res_state[0][[
    #     'A', 'B', 'mean(A)', 'std(A)', 'mean(B)', 'std(B)', 'T', 'dof',
    #     'p-unc', 'hedges'
    # ]].round(2).style.to_latex())

    print()
    print(
        res_class[2][
            [
                "A",
                "B",
                "mean(A)",
                "std(A)",
                "mean(B)",
                "std(B)",
                "T",
                "dof",
                "p-corr",
                "hedges",
            ]
        ]
        .round(2)
        .style.format(precision=2)
        .to_latex()
    )

    # print()
    # print(res_bound[0][[
    #     'A', 'B', 'mean(A)', 'std(A)', 'mean(B)', 'std(B)', 'T', 'dof',
    #     'p-unc', 'hedges'
    # ]].round(2).style.to_latex())

    # NOTE: rank-based analysis
    def rank_models(x):
        x = x.sort_values(by="bic")
        x["rank"] = np.arange(1, x.shape[0] + 1, 1)
        return x

    rank_table = []
    for g in [20, 19, 15]:
        dd = d[d["group"] == g]
        dd = dd.groupby(["subject"]).apply(rank_models).reset_index(drop=True)
        dd = dd.sort_values(["model_class", "model_state", "model_bound"])
        rank_table.append(pd.crosstab(dd["model"], dd["rank"]))

    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 4))
    cm = sns.color_palette("Blues", as_cmap=True)
    sns.heatmap(
        rank_table[0],
        ax=ax[0, 0],
        annot=True,
        cbar=False,
        cmap=sns.color_palette("Blues", as_cmap=True),
    )
    sns.heatmap(
        rank_table[1],
        ax=ax[0, 1],
        annot=True,
        cbar=False,
        cmap=sns.color_palette("Greens", as_cmap=True),
    )
    sns.heatmap(
        rank_table[2],
        ax=ax[0, 2],
        annot=True,
        cbar=False,
        cmap=sns.color_palette("Reds", as_cmap=True),
    )
    ax[0, 1].set_yticks([])
    ax[0, 2].set_yticks([])
    ax[0, 1].set_ylabel("")
    ax[0, 2].set_ylabel("")
    ax[0, 0].set_title("Experiment 1")
    ax[0, 1].set_title("Experiment 2")
    ax[0, 2].set_title("Experiment 3")
    plt.tight_layout()
    plt.savefig("../figures/fig_model_rank.tif")
    plt.savefig("../figures/fig_model_rank.pdf")
    plt.close()
