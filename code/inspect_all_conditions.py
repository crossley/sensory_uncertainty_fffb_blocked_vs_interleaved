from imports import *
from util_funcs import *

d = load_all_data_grand()

for grp in d.group.unique():
    dd = d[d["group"] == grp].copy()
    dd = dd[dd["phase"] != "baseline"]
    dd["phase"] = dd["phase"].cat.remove_unused_categories()
    dd["sig_mpep"] = dd["sig_mpep"].cat.remove_unused_categories()
    dd["sig_mpep_prev"] = dd["sig_mpep_prev"].cat.remove_unused_categories()
    dd["rot"] *= -1

    ddd = (
        dd.groupby(["group", "phase", "trial", "sig_mpep_prev"])[
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

    ddd["trial"] = ddd.groupby(["group"]).cumcount()

    fig, ax = plt.subplots(3, 2, figsize=(12, 8))
    sns.scatterplot(data=ddd, x="trial", y="rot", ax=ax[0, 0], alpha=0.1)
    sns.scatterplot(data=ddd, x="trial", y="rot", ax=ax[0, 1], alpha=0.1)
    sns.scatterplot(
        data=ddd,
        x="trial",
        y="ha_init",
        hue="sig_mpep_prev",
        style="phase",
        ax=ax[0, 0],
    )
    sns.scatterplot(
        data=ddd,
        x="trial",
        y="ha_end",
        hue="sig_mpep_prev",
        style="phase",
        ax=ax[0, 1],
    )
    sns.violinplot(
        data=ddd[ddd["phase"] == "adaptation"],
        x="sig_mpep_prev",
        y="delta_ha_init",
        ax=ax[1, 0],
    )
    sns.violinplot(
        data=ddd[ddd["phase"] == "adaptation"],
        x="sig_mpep_prev",
        y="fb_int",
        ax=ax[2, 0],
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
            ax=ax[1, 1],
        )
    for smpep in np.sort(ddd.sig_mpep_prev.unique()):
        sns.regplot(
            data=ddd.loc[
                (ddd["phase"] == "adaptation") & (ddd["sig_mpep_prev"] == smpep)
            ],
            x="error_mp",
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
    plt.suptitle(str(grp))
    plt.tight_layout()
    plt.savefig("../figures/tmp_" + str(grp) + ".pdf")
    plt.close()
    # plt.show()
