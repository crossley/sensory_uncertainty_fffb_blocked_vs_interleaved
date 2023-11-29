from imports import *
from util_funcs_2 import *
import model_context_1f1s
import model_context_1fns
import model_context_nf1s
import model_context_nfns
import model_aim_y


if __name__ == "__main__":
    # NOTE: prep data
    d = load_all_data_grand()

    d = d[d.group.isin([7, 8, 19])]

    d["group_class"] = d["group"].map({7: "blocked", 8: "blocked", 19: "interleaved"})

    d["group"] = d["group"].astype("category")
    d["sig_mpep_prev"] = d["sig_mpep_prev"].astype("category")

    d["group"] = d["group"].cat.remove_unused_categories()
    d["phase"] = d["phase"].cat.remove_unused_categories()
    d["sig_mpep_prev"] = d["sig_mpep_prev"].cat.remove_unused_categories()

    dd = d[d.phase.isin(["adaptation", "washout"])]
    # dd = dd[dd.group.isin([19])]

    froot = "../fits/"

    d_subject_list = []
    d_params_list = []
    for file in os.listdir(froot):
        if file.endswith(".txt"):
            subject = file.split("_")[-1].split(".")[0]
            group = file.split("_")[-3]
            subject = int(subject)
            group = int(group)

            d_subject = dd[(dd["subject"] == subject) & (dd["group"] == group)].copy()

            if not d_subject.empty:
                rot = d_subject.rot.to_numpy()
                sig_mp = d_subject.sig_mp.to_numpy()
                sig_ep = d_subject.sig_ep.to_numpy()
                group = d_subject.group.to_numpy()
                x_obs_mp = d_subject["ha_init"].to_numpy()
                x_obs_ep = d_subject["ha_end"].to_numpy()

                args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep)
                params = np.loadtxt(os.path.join(froot, file), delimiter=",")[:-2]

                p_names = np.array(
                    [
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
                    ]
                )

                d_params = pd.DataFrame({"params": params, "p_names": p_names})
                d_params["subject"] = np.unique(subject)[0]
                d_params["group"] = np.unique(group)[0]

                if "context_1s" in file:
                    x_pred = model_context_1s.simulate(params, args)
                    x_pred_mp = x_pred[1]
                    x_pred_ep = x_pred[0]
                    d_subject["ha_init_pred"] = x_pred_mp
                    d_subject["ha_end_pred"] = x_pred_ep
                    d_subject["model"] = "1s"
                    d_params["model"] = "1s"

                if "context_1f1s" in file:
                    x_pred = model_context_1f1s.simulate(params, args)
                    x_pred_mp = x_pred[1]
                    x_pred_ep = x_pred[0]
                    d_subject["ha_init_pred"] = x_pred_mp
                    d_subject["ha_end_pred"] = x_pred_ep
                    d_subject["model"] = "1f1s"
                    d_params["model"] = "1f1s"

                if "context_1fns" in file:
                    x_pred = model_context_1fns.simulate(params, args)
                    x_pred_mp = x_pred[1]
                    x_pred_ep = x_pred[0]
                    d_subject["ha_init_pred"] = x_pred_mp
                    d_subject["ha_end_pred"] = x_pred_ep
                    d_subject["model"] = "1fns"
                    d_params["model"] = "1fns"

                if "context_nf1s" in file:
                    x_pred = model_context_nf1s.simulate(params, args)
                    x_pred_mp = x_pred[1]
                    x_pred_ep = x_pred[0]
                    d_subject["ha_init_pred"] = x_pred_mp
                    d_subject["ha_end_pred"] = x_pred_ep
                    d_subject["model"] = "nf1s"
                    d_params["model"] = "nf1s"

                if "context_nfns" in file:
                    x_pred = model_context_nfns.simulate(params, args)
                    x_pred_mp = x_pred[1]
                    x_pred_ep = x_pred[0]
                    d_subject["ha_init_pred"] = x_pred_mp
                    d_subject["ha_end_pred"] = x_pred_ep
                    d_subject["model"] = "nfns"
                    d_params["model"] = "nfns"

                if "aim_y" in file:
                    x_pred = model_aim_y.simulate(params, args)
                    x_pred_mp = x_pred[1]
                    x_pred_ep = x_pred[0]
                    d_subject["ha_init_pred"] = x_pred_mp
                    d_subject["ha_end_pred"] = x_pred_ep
                    d_subject["model"] = "aim_y"
                    d_params["model"] = "aim_y"

                d_subject_list.append(d_subject)
                d_params_list.append(d_params)

    d_subject = pd.concat(d_subject_list)
    d_params = pd.concat(d_params_list)

    # NOTE: plot results
    for grp in d_subject.group.unique():
        fig, ax = plt.subplots(2, 3, squeeze=False, figsize=(14, 8))
        ax = ax.flatten()
        models = ["1f1s", "1fns", "nf1s", "nfns", "aim_y"]
        for i in range(len(models)):
            dd = d_subject[(d_subject.group == grp) & (d_subject.model == models[i])]
            dd = dd[dd["trial"] < dd["trial"].max()]
            sns.lineplot(
                data=dd,
                x="trial",
                y="ha_init_pred",
                hue="sig_mp_prev",
                style="phase",
                markers=True,
                legend=False,
                ax=ax[i],
            )
            ax[i].set_title(models[i])
        sns.lineplot(
            # data=d[d["group_class"] == "interleaved"],
            data=d[d["group"] == grp],
            x="trial",
            y="ha_init",
            style="phase",
            hue="sig_mpep_prev",
            markers=True,
            ax=ax[-1],
        )
        [x.set_ylim([0, 14]) for x in ax]
        plt.suptitle(f"Group {grp}")
        plt.tight_layout()
        plt.show()

    # NOTE: validate on blocked conditions
    mean_params = d_params.groupby(["group", "model", "p_names"]).mean().reset_index()

    sig_mp_1 = [1] * rot.shape[0]
    sig_ep_1 = [1] * rot.shape[0]
    args_1 = (rot, sig_mp_1, sig_ep_1)

    sig_mp_3 = [3] * rot.shape[0]
    sig_ep_3 = [3] * rot.shape[0]
    args_3 = (rot, sig_mp_3, sig_ep_3)

    models = ["1f1s", "1fns", "nf1s", "nfns", "aim_y"]
    sim_func = [
        model_context_1f1s.simulate,
        model_context_1fns.simulate,
        model_context_nf1s.simulate,
        model_context_nfns.simulate,
        model_aim_y.simulate,
    ]

    fig, ax = plt.subplots(2, 3, squeeze=False, figsize=(14, 8))
    ax = ax.flatten()
    for i in range(len(models)):
        params = mean_params[mean_params.model == models[i]].copy()
        category = pd.CategoricalDtype(categories=p_names, ordered=True)
        params["p_names"] = params["p_names"].astype(category)
        params = params.sort_values("p_names")

        # fig2, ax2 = plt.subplots(1, 1, squeeze=False, figsize=(18, 8))
        # sns.barplot(data=params, x="p_names", y="params", hue="group", ax=ax2[0, 0])
        # for item in ax2[0, 0].get_xticklabels():
        #     item.set_rotation(45)
        # plt.show()

        params_1 = params[params.group == 7].copy()
        params_1 = params_1["params"].to_numpy()
        x_pred = sim_func[i](params_1, args_1)
        x_pred_mp = x_pred[1]
        x_pred_ep = x_pred[0]
        ax[i].plot(x_pred_mp, ".C0", label="1")

        params_3 = params[params.group == 8].copy()
        params_3 = params_3["params"].to_numpy()
        x_pred = sim_func[i](params_3, args_3)
        x_pred_mp = x_pred[1]
        x_pred_ep = x_pred[0]
        ax[i].plot(x_pred_mp, ".C2", label="3")

        ax[i].set_title(models[i])

    sns.lineplot(
        data=d[d["group_class"] == "blocked"],
        x="trial",
        y="ha_init",
        style="phase",
        hue="sig_mpep_prev",
        markers=True,
        ax=ax[-1],
    )

    [x.set_ylim([0, 12]) for x in ax]
    [x.legend() for x in ax]

    plt.tight_layout()
    plt.show()
