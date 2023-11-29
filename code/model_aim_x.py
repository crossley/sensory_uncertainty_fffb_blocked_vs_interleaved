from imports import *


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

    r = args[0]
    sig_mp = args[1]
    sig_ep = args[2]

    n_trials = r.shape[0]

    delta_ep = np.zeros(n_trials)
    delta_mp = np.zeros(n_trials)
    xff = np.zeros(n_trials)
    xff2 = np.zeros(n_trials)
    xfb = np.zeros(n_trials)
    yff = np.zeros(n_trials)
    yfb = np.zeros(n_trials)
    y = np.zeros(n_trials)

    xfb[0] = xfb_init

    # some condition files call a 4 a 0
    sig_mp[sig_mp == 0] = 4
    sig_ep[sig_ep == 0] = 4

    for i in range(n_trials - 1):
        ff_adapt_mp = 0.0
        ff_adapt_ep = 0.0
        ff_adapt_mp2 = 0.0
        ff_adapt_ep2 = 0.0

        yff[i] = xff[i] + xff2[i]

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

        # update ff state slow
        xff[i + 1] = beta_ff * xff[i] + ff_adapt_mp + ff_adapt_ep + bias_ff

        # update ff state fast
        if sig_mp[i] != 4:
            xff2[i + 1] = gamma_ff[sig_mp[i] - 1]
        else:
            xff2[i + 1] = xff[i]

        # clip the feedback gain to prevent instability
        xfb = np.clip(xfb, -2, 2)

    return (y, yff, yfb, xff, xfb)
