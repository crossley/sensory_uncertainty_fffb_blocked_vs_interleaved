from imports import *

def load_all_data_grand():
    # d1 = pd.read_csv("../data/exp1_2020.csv")
    # d2 = pd.read_csv("../data/exp2_2020.csv")
    # d3 = pd.read_csv("../data/exp3_2020.csv")
    # d4 = pd.read_csv("../data/exp4_2020.csv")
    # d5 = pd.read_csv("../data/exp5_2020.csv")
    d6 = pd.read_csv("../data/exp6_2020.csv")
    # d7 = pd.read_csv("../data/exp7_2020.csv")
    # d8 = pd.read_csv("../data/exp8_2021.csv")
    # d345 = pd.read_csv("../data/exp345_2021.csv")
    # d15 = pd.read_csv("../data/G15.csv")
    # d16 = pd.read_csv("../data/G16.csv")
    # d1718 = pd.read_csv("../data/G17_18.csv")
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

    # TODO: remove outlier (there is only one data point removed)
    d = d.loc[d["error_ep_prev"] > -15]

    return d


def obj_func(params, *args):
    obs = args

    rot = obs[0]
    sig_mp = obs[1]
    sig_ep = obs[2]
    x_obs_mp = obs[3]
    x_obs_ep = obs[4]
    sim_func = obs[5]

    args = (rot, sig_mp, sig_ep)

    x_pred = sim_func(params, args)
    x_pred_mp = x_pred[1]
    x_pred_ep = x_pred[0]

    sse_mp = np.sum((x_obs_mp - x_pred_mp) ** 2)
    sse_ep = np.sum((x_obs_ep - x_pred_ep) ** 2)

    sse = sse_mp + sse_ep

    return sse
