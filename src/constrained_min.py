import numpy as np


def line_search(
    f,
    x0,
    bk,
    pk_resolver,
    alpha_resolver,
    obj_tol=1e-8,
    param_tol=1e-8,
    max_iter=10_000,
    verbose=True,
    algo_name="unknown",
):
    x = np.asarray(x0, dtype=float)
    f_ans = f(x)
    f_val = f_ans[0]
    grad = f_ans[1]
    trace = []
    trace.append({"k": 0, "x": x, "fx": f_val})
    success = False
    if verbose:
        print(f"{algo_name} iteration 0/{max_iter}, x={x}, f(x)={f_val}")
    for k in range(1, max_iter + 1):
        rhs = pk_resolver(f_ans)
        pk = np.linalg.solve(bk, rhs)
        alpha = alpha_resolver(f, x, f_val, grad, pk)[0]
        if alpha * np.linalg.norm(pk) < param_tol:
            success = True
            break
        x_new = x + alpha * pk
        f_ans = f(x_new)
        f_new = f_ans[0]
        grad = f_ans[1]
        if verbose:
            print(f"{algo_name} iteration {k}/{max_iter}, x={x_new}, f(x)={f_new}")
        trace.append({"k": k, "x": x_new, "fx": f_new})
        if abs(f_new - f_val) < obj_tol:
            x, f_val = x_new, f_new
            success = True
            break
        x, f_val = x_new, f_new
    return x, f_val, success, trace


def calculate_alpha_wolfe_conditions(
    f,
    x,
    f0,
    g0,
    p,
    wolfe_condition_constant=0.01,
    alpha0=1.0,
    backtrack_constant=0.5,
    max_ls=20,
):
    alpha = alpha0
    for _ in range(max_ls):
        ans = f(x + alpha * p, hessian_flag=False)
        f_new, g_new = ans[0], ans[1]
        if f_new > f0 + wolfe_condition_constant * alpha * np.dot(g0, p) or np.isnan(
            f_new
        ):
            alpha *= backtrack_constant
        else:
            break
    return alpha, f_new, g_new


def newton_pk_resolver(ans, fallback_to_gd=True):
    _, g, H = ans
    try:
        p = -np.linalg.solve(H, g)
        if np.dot(g, p) >= 0:
            raise ValueError("Newton direction is not descent")
        return p
    except (np.linalg.LinAlgError, ValueError):
        if fallback_to_gd:
            return -g
        raise


def make_barrier(t: float, func, ineq_constraints):

    def phi(x, hessian_flag: bool = True):
        f_val, f_grad, f_hess = func(x)

        val = t * f_val
        grad = t * f_grad.copy()
        if hessian_flag:
            hess = t * f_hess.copy()

        for g in ineq_constraints:
            g_val, g_grad, g_hess = g(x)
            if g_val >= 0:
                if hessian_flag:
                    return np.inf, None, None
                return np.inf, None
            val -= np.log(-g_val)
            grad += -1.0 / g_val * g_grad
            if hessian_flag:
                hess += np.outer(g_grad, g_grad) / (g_val**2) - g_hess / g_val

        if hessian_flag:
            return val, grad, hess
        return val, grad

    return phi


def make_kkt_pk_resolver(A_: np.ndarray, n):

    m_eq = A_.shape[0]

    if m_eq == 0:
        return newton_pk_resolver

    def pk_resolver(ans, fallback_to_gd: bool = True):
        _, g, H = ans
        KKT = np.block(
            [
                [H, A_.T],
                [A_, np.zeros((m_eq, m_eq))],
            ]
        )
        rhs = -np.concatenate([g, np.zeros(m_eq)])
        try:
            sol = np.linalg.solve(KKT, rhs)
            p = sol[:n]
            if np.dot(g, p) >= 0:
                raise ValueError
            return p
        except (np.linalg.LinAlgError, ValueError):
            if not fallback_to_gd:
                raise
            z = -g
            ATA_inv = np.linalg.inv(A_ @ A_.T)
            p_proj = z - A_.T @ (ATA_inv @ (A_ @ z))
            return p_proj

    return pk_resolver


def interior_pt(
    func,
    ineq_constraints,
    eq_constraints_mat,
    eq_constraints_rhs,
    x0,
    *,
    t0: float = 1.0,
    mu: float = 10.0,
    obj_tol: float = 1e-8,
    param_tol: float = 1e-8,
    outer_tol: float = 1e-12,
    max_outer_steps: int = 50,
    max_inner_steps: int = 100,
    verbose: bool = True,
):

    x = np.asarray(x0, dtype=float).copy()
    A = np.atleast_2d(eq_constraints_mat).astype(float)
    b = np.asarray(eq_constraints_rhs, dtype=float)
    m_ineq = len(ineq_constraints)
    n, n_eq = x.size, A.shape[0]

    if m_ineq and any(c(x)[0] >= 0 for c in ineq_constraints):
        raise ValueError("x0 is not strictly feasible (some g_i(x0) ≥ 0).")
    if n_eq and np.linalg.norm(A @ x - b) > 1e-10:
        raise ValueError("x0 does not satisfy the equality constraints.")

    t = t0
    trace = []
    for outer in range(max_outer_steps):

        if verbose:
            print(f"\n── Barrier outer step {outer}  (t = {t:.3e}) ──")

        phi_t = make_barrier(t, func, ineq_constraints)
        pk_resolver = make_kkt_pk_resolver(A, n)

        x, _, success, _ = line_search(
            phi_t,
            x,
            np.identity(n),
            pk_resolver,
            calculate_alpha_wolfe_conditions,
            obj_tol=obj_tol,
            param_tol=param_tol,
            max_iter=max_inner_steps,
            verbose=verbose,
            algo_name="IPM-Newton",
        )
        if not success:
            raise RuntimeError("Inner Newton failed to converge.")
        trace.append({"x1": x[0], "x2": x[1], "f_x": func(x)[0], "t": t})
        if m_ineq == 0 or m_ineq / t < outer_tol:
            if verbose:
                print("Duality gap small enough – terminating.")
            break

        t *= mu

    return x, trace
