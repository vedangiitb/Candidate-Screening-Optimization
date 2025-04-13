def golden_section_search(func, a=0.0, b=1.0, tol=1e-4):
    gr = (np.sqrt(5) + 1) / 2
    c = b - (b - a) / gr
    d = a + (b - a) / gr
    while abs(b - a) > tol:
        if func(c) < func(d):
            a = c
        else:
            b = d
        c = b - (b - a) / gr
        d = a + (b - a) / gr
    return (b + a) / 2, -func((b + a) / 2)

def misclassification_loss(x, y, theta, w_fp=1.0, w_fn=1.0):
    preds = (x >= theta).astype(int)
    fp = np.sum((preds == 1) & (y == 0))
    fn = np.sum((preds == 0) & (y == 1))
    return w_fp * fp + w_fn * fn

def project_to_simplex(v):
    if np.sum(v) == 1 and np.all(v >= 0): return v
    v = np.maximum(v, 0)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v)+1) > (cssv - 1))[0][-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    return np.maximum(v - theta, 0)

def joint_optimize(y, scores, match_tech, match_soft, experience, gender, costs,
                   tau_t, tau_s, tau_p, tau_c, tau_e, tau_g, B, kmin, kmax,
                   max_iters=100, eta=0.1, eps=1e-6):
    
    N, d = scores.shape
    alpha = np.ones(d) / d
    epsilon = 1e-5

    for iteration in range(max_iters):
        x_scores, _ = shortlisted_candidates(
            N=N, scores=scores, match_tech=match_tech, match_soft=match_soft,
            experience=experience, gender=gender, costs=costs,
            tau_t=tau_t, tau_s=tau_s, tau_p=tau_p, tau_c=tau_c,
            tau_e=tau_e, tau_g=tau_g, B=B, kmin=kmin, kmax=kmax,
            alpha=alpha, theta=0.5
        )

        def neg_f1(theta):
            preds = (x_scores >= theta).astype(int)
            if np.sum(preds) == 0: return 0
            return -f1_score(y, preds)

        theta_opt, f1 = golden_section_search(neg_f1)

        grad = np.zeros_like(alpha)
        for j in range(d):
            delta = np.zeros_like(alpha)
            delta[j] = epsilon
            alpha_perturbed = alpha + delta
            alpha_perturbed = project_to_simplex(alpha_perturbed)

            x_perturbed, _ = shortlisted_candidates(
                N=N, scores=scores, match_tech=match_tech, match_soft=match_soft,
                experience=experience, gender=gender, costs=costs,
                tau_t=tau_t, tau_s=tau_s, tau_p=tau_p, tau_c=tau_c,
                tau_e=tau_e, tau_g=tau_g, B=B, kmin=kmin, kmax=kmax,
                alpha=alpha_perturbed, theta=theta_opt
            )

            loss_plus = misclassification_loss(x_perturbed, y, theta_opt)
            loss_now = misclassification_loss(x_scores, y, theta_opt)
            grad[j] = (loss_plus - loss_now) / epsilon

        alpha_new = project_to_simplex(alpha - eta * grad)
        if np.linalg.norm(alpha_new - alpha) < eps:
            break
        alpha = alpha_new

    return alpha, theta_opt, f1
