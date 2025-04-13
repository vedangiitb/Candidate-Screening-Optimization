def shortlisted_candidates(
    N, scores, match_tech, match_soft, experience, gender, costs,
    tau_t, tau_s, tau_p, tau_c, tau_e, tau_g, B, kmin, kmax,
    alpha=None
):
    if alpha is None:
        alpha = np.array([0.25, 0.2, 0.15, 0.15, 0.15, 0.1])

    weighted_scores = scores @ alpha
    c = -weighted_scores

    A_ub = []
    b_ub = []

    # Max number of candidates to shortlist
    A_ub.append(np.ones(N))
    b_ub.append(kmax)

    # Min number of candidates to shortlist
    A_ub.append(-np.ones(N))
    b_ub.append(-kmin)

    # Avg. technical skill constraint
    A_ub.append(-(match_tech - tau_t))
    b_ub.append(0)

    # Avg. soft skill constraint
    A_ub.append(-(match_soft - tau_s))
    b_ub.append(0)

    # Psychometric scores Threshold
    psych = scores[:, 2]
    A_ub.append(tau_p - psych)
    b_ub.append(0)

    # Communication skills Threshold
    comm = scores[:, 4]
    A_ub.append(tau_c - comm)
    b_ub.append(0)

    # Experience Threshold
    A_ub.append(tau_e - experience)
    b_ub.append(0)

    # Budget constraint
    A_ub.append(costs)
    b_ub.append(B)

    # Diversity constraint
    A_ub.append(-(gender - tau_g))
    b_ub.append(0)

    # Bounds for x_i (0 <= x_i <= 1)
    bounds = [(0, 1) for _ in range(N)]

    # Using linear programming to solve the optimization problem
    result = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub), bounds=bounds, method='highs')

    if result.success:
        return result.x, weighted_scores
    else:
        raise ValueError("Optimization failed:", result.message)
