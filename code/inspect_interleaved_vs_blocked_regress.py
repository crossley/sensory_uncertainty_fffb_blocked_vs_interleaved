from imports import *
from util_funcs import *

if __name__ == "__main__":
    # NOTE: prep data
    d = load_all_data_grand()

    d = d[d.group.isin([7, 8, 19])]

    print(d.groupby(["group", "phase"])["subject"].nunique())
    print(d.groupby(["group", "phase"])["trial"].max())
    print(d[d["group"] == 19].groupby(["subject", "sig_mpep_prev"])["trial"].count())

    d = d[d.phase.isin(["adaptation"])]
    d = d[d.sig_mpep_prev.isin([(1, 1), (3, 3)])]

    d["group_class"] = d["group"].map({7: "blocked", 8: "blocked", 19: "interleaved"})

    d["group"] = d["group"].astype("category")
    d["sig_mpep_prev"] = d["sig_mpep_prev"].astype("category")

    d["group"] = d["group"].cat.remove_unused_categories()
    d["phase"] = d["phase"].cat.remove_unused_categories()
    d["sig_mpep_prev"] = d["sig_mpep_prev"].cat.remove_unused_categories()

    # NOTE: for each group print the number of unique subject
    print(d.groupby(["group_class", "group"])["subject"].nunique())
    print(d.groupby(["group_class", "group"])["trial"].max())
    print(d.groupby(["group_class", "group"])["rot"].describe())

    # NOTE: check that everybody in the blocked conditions
    # got the same sequence of rotations and ditto for
    # interleaved.
    fig, ax = plt.subplots(3, 2, squeeze=False, figsize=(6, 6))
    sns.lineplot(
        data=d[d.group == 7],
        x="trial",
        y="rot",
        markers=True,
        ax=ax[0, 0],
    )
    sns.lineplot(
        data=d[d.group == 8],
        x="trial",
        y="rot",
        markers=True,
        ax=ax[1, 0],
    )
    sns.lineplot(
        data=d[d.group == 19],
        x="trial",
        y="rot",
        markers=True,
        ax=ax[2, 0],
    )
    sns.histplot(
        data=d[d.group == 7],
        x="rot",
        bins=np.arange(-30, 10, 2),
        ax=ax[0, 1],
    )
    sns.histplot(
        data=d[d.group == 8],
        x="rot",
        bins=np.arange(-30, 10, 2),
        ax=ax[1, 1],
    )
    sns.histplot(
        data=d[d.group == 19],
        x="rot",
        bins=np.arange(-30, 10, 2),
        ax=ax[2, 1],
    )
    [ax[i, 0].set_ylim([-30, 10]) for i in range(3)]
    [ax[i, 1].set_xlim([-30, 10]) for i in range(3)]
    plt.tight_layout()
    plt.show()

    dd = (
        d.groupby(
            ["group_class", "group", "phase", "trial", "sig_mpep_prev"], observed=True
        )[
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

    dd.trial = dd.trial + 1

    mod_formula = "ha_init ~ "
    mod_formula += "C(group_class, Diff)  * C(sig_mpep_prev, Diff) * np.log(trial) + "
    mod_formula += "1"

    # regression via statsmodels
    mod = smf.ols(mod_formula, data=dd)
    res = mod.fit()
    print(res.summary())

    dd['ha_init_pred'] = res.model.predict(res.params, res.model.exog)

    fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(14, 6))
    sns.lineplot(
        data=dd,
        x="trial",
        y="ha_init",
        hue="sig_mpep_prev",
        style="group_class",
        markers=True,
        ax=ax[0, 0],
    )
    sns.lineplot(
        data=dd,
        x="trial",
        y="ha_init_pred",
        hue="sig_mpep_prev",
        style="group_class",
        markers=True,
        ax=ax[0, 1],
    )
    plt.savefig("../figures/fig1.png")
    plt.close()

    # regression via pingouin
    y, X = patsy.dmatrices(mod_formula, dd, return_type="dataframe")
    res_pg = pg.linear_regression(X, np.squeeze(y.to_numpy()), relimp=True)
    print(res_pg)

    # NOTE: write results to latex table
    res_pg.rename(
        columns={"CI[2.5%]": "CI[2.5\%]", "CI[97.5%]": "CI[97.5\%]"}, inplace=True
    )

    res_pg = res_pg[
        ["names", "coef", "se", "T", "pval", "CI[2.5\%]", "CI[97.5\%]", "relimp"]
    ]

    beta_labels = list(
        [
            r"$\beta_0$",
            r"$Interleaved - Blocked$",
            r"$\sigma_L - \sigma_H$",
            r"$(Interleaved - Blocked)$:$(\sigma_L - \sigma_H)$",
            r"log(Trial)",
            r"$(Interleaved - Blocked)$:$log(Trial)$",
            r"$(\sigma_L - \sigma_H)$:$log(Trial)$",
            r"$(Interleaved - Blocked)$:$(\sigma_L - \sigma_H)$:$log(Trial)$",
        ]
    )

    res_pg["names"] = beta_labels

    res_pg.style.format(precision=2).to_latex(
        "../stats_tables/stats_table_regression.tex"
    )
