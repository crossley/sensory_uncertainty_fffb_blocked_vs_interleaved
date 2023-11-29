from imports import *
from util_funcs import *

if __name__ == "__main__":
    # NOTE: prep data
    d = load_all_data_grand()

    d = d[d.group.isin([7, 8, 19])]
    # d = d[d.sig_mpep_prev.isin([(1, 1), (3, 3)])]

    d["group_class"] = d["group"].map({7: "blocked", 8: "blocked", 19: "interleaved"})

    d["group"] = d["group"].astype("category")
    d["sig_mpep_prev"] = d["sig_mpep_prev"].astype("category")

    d["group"] = d["group"].cat.remove_unused_categories()
    d["phase"] = d["phase"].cat.remove_unused_categories()
    d["sig_mpep_prev"] = d["sig_mpep_prev"].cat.remove_unused_categories()

    dd = (
        d.groupby(["group_class", "group", "phase", "sig_mpep_prev", "trial"])[
            "ha_init"
        ]
        .mean()
        .reset_index()
    )

    # dd = dd[dd["trial"] > 1]

    # NOTE: plot behaviour
    # fig, ax = plt.subplots(1, 2, squeeze=False, figsize=(16, 6))
    # sns.lineplot(
    #     data=d[d["group_class"] == "blocked"],
    #     x="trial",
    #     y="ha_init",
    #     style="phase",
    #     hue="sig_mpep_prev",
    #     markers=True,
    #     ax=ax[0, 0],
    # )
    # sns.lineplot(
    #     data=d[d["group_class"] == "interleaved"],
    #     x="trial",
    #     y="ha_init",
    #     style="phase",
    #     hue="sig_mpep_prev",
    #     markers=True,
    #     ax=ax[0, 1],
    # )
    # ax[0, 0].set_ylim(0, 14)
    # ax[0, 1].set_ylim(0, 14)
    # plt.show()

    # NOTE: fit context model
    modname = "context_model"

    bounds = (
        # alpha_ff, beta_ff, bias_ff,
        (0, 1),
        (0, 1),
        (-10, 10),
        # alpha_ff2, beta_ff2, bias_ff2,
        (0, 1),
        (0, 1),
        (-10, 10),
        # alpha_fb, beta_fb, xfb_init,
        (0, 1),
        (-10, 10),
        (-2, 2),
        # gamma_fbint_1, gamma_fbint_2, gamma_fbint_3, gamma_fbint_4,
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        # gamma_ff_1, gamma_ff_2, gamma_ff_3, gamma_ff_4,
        (0, 1),
        (0, 1),
        (0, 1),
        (0, 1),
        # temporal_discount,
        (0, 0),
    )

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
        "obj_func": obj_func_context,
        "bounds": bounds,
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

    froot = "../fits/"

    d = d[d.group.isin([19])]
    # d = d[d.subject.isin([1])]
    d = d[d.phase.isin(["adaptation"])]

    for grp in d["group"].unique():
        for sub in d[d["group"] == grp]["subject"].unique():
            print(grp, sub)

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
                func=fit_args["obj_func"],
                bounds=fit_args["bounds"],
                constraints=fit_args["constraints"],
                args=args,
                disp=fit_args["disp"],
                maxiter=fit_args["maxiter"],
                popsize=fit_args["popsize"],
                mutation=fit_args["mutation"],
                recombination=fit_args["recombination"],
                tol=fit_args["tol"],
                polish=fit_args["polish"],
                updating=fit_args["updating"],
                workers=fit_args["workers"],
            )

            fout = os.path.join(
                froot,
                "fit_results_context_model_"
                + "group_"
                + str(grp)
                + "_sub_"
                + str(sub)
                + ".txt",
            )
            with open(fout, "w") as f:
                tmp = np.concatenate((results["x"], [results["fun"]]))
                tmp = np.reshape(tmp, (tmp.shape[0], 1))
                np.savetxt(f, tmp.T, "%0.4f", delimiter=",", newline="\n")

    d_subject_list = []
    for file in os.listdir(froot):
        if file.startswith("fit_results_context_model"):
            subject = file.split("_")[-1].split(".")[0]
            group = file.split("_")[5]
            subject = int(subject)
            group = int(group)
            d_subject = d[(d["subject"] == subject) & (d["group"] == group)].copy()

            rot = d_subject.rot.to_numpy()
            sig_mp = d_subject.sig_mp.to_numpy()
            sig_ep = d_subject.sig_ep.to_numpy()
            group = d_subject.group.to_numpy()
            x_obs_mp = d_subject["ha_init"].to_numpy()
            x_obs_ep = d_subject["ha_end"].to_numpy()

            args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep, group, modname)
            params = np.loadtxt(os.path.join(froot, file), delimiter=",")[:-1]

            x_pred = simulate_context(params, args)
            x_pred_mp = x_pred[1]
            x_pred_ep = x_pred[0]
            xff = x_pred[3]

            # fig, ax = plt.subplots(1, 3, squeeze=False, figsize=(16, 6))
            # ax[0, 0].plot(x_obs_mp, label="human")
            # ax[0, 0].plot(x_pred_mp, label="model")
            # ax[0, 1].plot(x_obs_ep, label="human")
            # ax[0, 1].plot(x_pred_ep, label="model")
            # ax[0, 2].plot(xff[0, :], label="1")
            # ax[0, 2].plot(xff[1, :], label="2")
            # ax[0, 2].plot(xff[2, :], label="3")
            # ax[0, 2].plot(xff[3, :], label="4")
            # plt.legend()
            # plt.show()

            d_subject["ha_init_pred"] = x_pred_mp
            d_subject["ha_end_pred"] = x_pred_ep

            d_subject_list.append(d_subject)

            # fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(14, 8))
            # sns.scatterplot(
            #    data=d_subject,
            #    x="trial",
            #    y="ha_init",
            #    hue="sig_mp",
            #    style="phase",
            #    ax=ax[0, 0],
            # )
            # sns.scatterplot(
            #    data=d_subject,
            #    x="trial",
            #    y="ha_end",
            #    hue="sig_mp",
            #    style="phase",
            #    ax=ax[0, 1],
            # )
            # sns.scatterplot(
            #    data=d_subject,
            #    x="trial",
            #    y="ha_init_pred",
            #    hue="sig_mp",
            #    style="phase",
            #    ax=ax[1, 0],
            # )
            # sns.scatterplot(
            #    data=d_subject,
            #    x="trial",
            #    y="ha_end_pred",
            #    hue="sig_mp",
            #    style="phase",
            #    ax=ax[1, 1],
            # )
            # plt.show()

    d_subject = pd.concat(d_subject_list)

    fig, ax = plt.subplots(2, 2, squeeze=False, figsize=(14, 8))
    sns.lineplot(
        data=d_subject,
        x="trial",
        y="ha_init",
        hue="sig_mp_prev",
        style="phase",
        ax=ax[0, 0],
    )
    sns.lineplot(
        data=d_subject,
        x="trial",
        y="ha_end",
        hue="sig_mp_prev",
        style="phase",
        ax=ax[0, 1],
    )
    sns.lineplot(
        data=d_subject,
        x="trial",
        y="ha_init_pred",
        hue="sig_mp_prev",
        style="phase",
        ax=ax[1, 0],
    )
    sns.lineplot(
        data=d_subject,
        x="trial",
        y="ha_end_pred",
        hue="sig_mp_prev",
        style="phase",
        ax=ax[1, 1],
    )
    plt.show()
