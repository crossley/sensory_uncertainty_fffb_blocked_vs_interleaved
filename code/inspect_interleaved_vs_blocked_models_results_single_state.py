from imports import *
from util_funcs_2 import *
import model_context_1s

if __name__ == "__main__":
    # NOTE: prep data
    d = load_all_data_grand()

    d = d[d.group.isin([7, 8, 19])]

    d["group_class"] = d["group"].map({
        7: "blocked",
        8: "blocked",
        19: "interleaved"
    })

    d["group"] = d["group"].astype("category")
    d["sig_mpep_prev"] = d["sig_mpep_prev"].astype("category")

    d["group"] = d["group"].cat.remove_unused_categories()
    d["phase"] = d["phase"].cat.remove_unused_categories()
    d["sig_mpep_prev"] = d["sig_mpep_prev"].cat.remove_unused_categories()

    dd = d[d.phase.isin(["adaptation"])]
    # dd = dd[dd.group.isin([19])]

    froot = "../fits/"

    d_subject_list = []
    d_params_list = []
    for file in os.listdir(froot):
        if ("context_1s" in file) and (file.endswith(".txt")):
            subject = file.split("_")[-1].split(".")[0]
            group = file.split("_")[-3]
            subject = int(subject)
            group = int(group)

            d_subject = dd[(dd["subject"] == subject)
                           & (dd["group"] == group)].copy()

            if not d_subject.empty:
                rot = d_subject.rot.to_numpy()
                sig_mp = d_subject.sig_mp.to_numpy()
                sig_ep = d_subject.sig_ep.to_numpy()
                group = d_subject.group.to_numpy()
                x_obs_mp = d_subject["ha_init"].to_numpy()
                x_obs_ep = d_subject["ha_end"].to_numpy()

                args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep)
                params = np.loadtxt(os.path.join(froot, file),
                                    delimiter=",")[:-2]

                p_names = np.array([
                    "alpha_ff",
                    "beta_ff",
                    "bias_ff",
                    "alpha_ff2",
                    "beta_ff2",
                    "bias_ff2",
                    "alpha_fb",
                    "beta_fb",
                    "xfb_init",
                    "gamma_fbint_1",
                    "gamma_fbint_2",
                    "gamma_fbint_3",
                    "gamma_fbint_4",
                    "gamma_ff_1",
                    "gamma_ff_2",
                    "gamma_ff_3",
                    "gamma_ff_4",
                ])

                d_params = pd.DataFrame({"params": params, "p_names": p_names})
                d_params["subject"] = np.unique(subject)[0]
                d_params["group"] = np.unique(group)[0]

                x_pred = model_context_1s.simulate(params, args)
                x_pred_mp = x_pred[1]
                x_pred_ep = x_pred[0]
                d_subject["ha_init_pred"] = x_pred_mp
                d_subject["ha_end_pred"] = x_pred_ep
                d_subject["model"] = "1s"
                d_params["model"] = "1s"

                d_subject_list.append(d_subject)
                d_params_list.append(d_params)

    d_subject = pd.concat(d_subject_list)
    d_params = pd.concat(d_params_list)

    # TODO: plot average with errors across subs
    d_pred_list = []
    model = "1s"
    sim_func = model_context_1s.simulate
    for g in d_params.group.unique():
        for s in d_params[(d_params.group == g)].subject.unique():
            params = d_params[(d_params.group == g)
                              & (d_params.subject == s)].copy()
            params = params.params.to_numpy()

            sig_mp = d_subject[(d_subject.group == g)
                               & (d_subject.subject == s)].sig_mp.to_numpy()
            sig_ep = d_subject[(d_subject.group == g)
                               & (d_subject.subject == s)].sig_ep.to_numpy()
            args = (rot, sig_mp, sig_ep)

            x_pred = sim_func(params, args)
            x_pred_mp = x_pred[1]
            x_pred_ep = x_pred[0]

            d_pred = pd.DataFrame({"ha_init": x_pred_mp, "ha_end": x_pred_ep})
            d_pred["trial"] = np.arange(1, d_pred.shape[0] + 1)
            d_pred["subject"] = s
            d_pred["group"] = g
            d_pred_list.append(d_pred)

    d_pred = pd.concat(d_pred_list)
    d_pred.group = d_pred.group.astype("category")
    d_pred = d_pred[d_pred.trial < 200]

    d_pred["sig_mpep_prev"] = d_pred["group"]
    d_pred["sig_mpep_prev"] = d_pred["sig_mpep_prev"].cat.rename_categories({
        7:
        "(1, 1)",
        8:
        "(3, 3)"
    })

    d_params["sig_mpep_prev"] = d_params["group"].astype("category")
    d_params["sig_mpep_prev"] = d_params[
        "sig_mpep_prev"].cat.rename_categories({
            7: "(1, 1)",
            8: "(3, 3)"
        })

    d_pred["Sensory uncertainty on trial t-1"] = d_pred["group"].map({
        7:
        "Low Uncertainty",
        8:
        "High Uncertainty"
    })

    d_params["Sensory uncertainty on trial t-1"] = d_params["group"].map({
        7:
        "Low Uncertainty",
        8:
        "High Uncertainty"
    })

    sns.set_palette("colorblind")
    fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(12, 4))
    sns.lineplot(
        data=d_pred,
        x="trial",
        y="ha_init",
        hue="Sensory uncertainty on trial t-1",
        markers=True,
        ax=ax[0, 0],
    )

    d_params_2 = d_params.copy()
    d_params_2 = d_params_2[[x in ["alpha_ff2"] for x in d_params_2.p_names]]

    sns.violinplot(
        data=d_params_2,
        x="p_names",
        y="params",
        hue="Sensory uncertainty on trial t-1",
        ax=ax[0, 1],
    )
    d_params_2 = d_params.copy()
    d_params_2 = d_params_2[[x in ["beta_ff2"] for x in d_params_2.p_names]]
    sns.violinplot(
        data=d_params_2,
        x="p_names",
        y="params",
        hue="Sensory uncertainty on trial t-1",
        ax=ax[0, 2],
    )
    ax[0, 1].set_xticklabels(["$\\alpha$"])
    ax[0, 2].set_xticklabels(["$\\beta$"])
    ax[0, 1].set_xlabel("")
    ax[0, 2].set_xlabel("")
    ax[0, 1].set_ylabel("Parameter Estimate (a.u.)")
    ax[0, 2].set_ylabel("Parameter Estimate (a.u.)")
    ax[0, 0].set_ylim([0, 12])
    plt.tight_layout()
    plt.savefig("../figures/fig2.png")
    plt.close()

    # NOTE: stats on parameter estimates
    x = d_params[(d_params.p_names == "alpha_ff2")
                 & (d_params.group == 7)].params.to_numpy()
    y = d_params[(d_params.p_names == "alpha_ff2")
                 & (d_params.group == 8)].params.to_numpy()
    res_alpha = pg.ttest(x, y, paired=True)

    x = d_params[(d_params.p_names == "beta_ff2")
                 & (d_params.group == 7)].params.to_numpy()
    y = d_params[(d_params.p_names == "beta_ff2")
                 & (d_params.group == 8)].params.to_numpy()
    res_beta = pg.ttest(x, y, paired=True)

    print(res_alpha)
    print(res_beta)
