from imports import *
from util_funcs_2 import *
import model_context_1s

if __name__ == "__main__":
    # NOTE: prep data
    d = load_all_data_grand()

    # d = d[d.group.isin([7, 8, 19])]

    # NOTE: only fit single-state to blocked conditions
    # d = d[d.group.isin([7, 8])]
    d = d[d.group.isin([8])]
    d = d[d.subject.isin([38, 39, 40])]

    d = d[d.phase.isin(["adaptation"])]

    d["group_class"] = d["group"].map({7: "blocked", 8: "blocked", 19: "interleaved"})

    d["group"] = d["group"].astype("category")
    d["sig_mpep_prev"] = d["sig_mpep_prev"].astype("category")

    d["group"] = d["group"].cat.remove_unused_categories()
    d["phase"] = d["phase"].cat.remove_unused_categories()
    d["sig_mpep_prev"] = d["sig_mpep_prev"].cat.remove_unused_categories()

    froot = "../fits/"

    bounds = (
        # alpha_ff, beta_ff, bias_ff,
        (0, 0),  # NOTE: to make it a single state model
        (1, 1),  # NOTE: to make it a single state model
        (0, 0),  # NOTE: to make it a single state model
        # alpha_ff2, beta_ff2, bias_ff2,
        (0, 1),
        (0, 1),
        (-20, 20),
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

    sim_func_list = [
        model_context_1s.simulate,
    ]

    mod_name_list = [
        "model_context_1s",
    ]

    for i, sim_func in enumerate(sim_func_list):
        mod_name = mod_name_list[i]

        # to improve your chances of finding a global minimum use higher
        # popsize (default 15), with higher mutation (default 0.5) and
        # (dithering), but lower recombination (default 0.7). this has the
        # effect of widening the search radius, but slowing convergence.
        fit_args = {
            "obj_func": obj_func,
            "sim_func": sim_func,
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

        for grp in d["group"].unique():
            for sub in d[d["group"] == grp]["subject"].unique():
                print(mod_name, grp, sub)

                dd = d[(d["subject"] == sub) & (d["group"] == grp)][
                    [
                        "rot",
                        "ha_init",
                        "ha_end",
                        "trial_abs",
                        "group",
                        "sig_mp",
                        "sig_ep",
                    ]
                ]

                rot = dd.rot.to_numpy()
                sig_mp = dd.sig_mp.to_numpy()
                sig_ep = dd.sig_ep.to_numpy()
                group = dd.group.to_numpy()
                x_obs_mp = dd["ha_init"].to_numpy()
                x_obs_ep = dd["ha_end"].to_numpy()

                args = (rot, sig_mp, sig_ep, x_obs_mp, x_obs_ep, sim_func)

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
                    "fit_results_"
                    + mod_name
                    + "_group_"
                    + str(grp)
                    + "_sub_"
                    + str(sub)
                    + ".txt",
                )
                with open(fout, "w") as f:
                    tmp = np.concatenate((results["x"], [results["fun"]]))
                    tmp = np.reshape(tmp, (tmp.shape[0], 1))
                    np.savetxt(f, tmp.T, "%0.4f", delimiter=",", newline="\n")
