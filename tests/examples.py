import numpy as np


def get_quadratic_probel():

    def f_obj(x, hessian_flag: bool = True):
        x_, y_, z_ = x
        val = x_**2 + y_**2 + (z_ + 1.0) ** 2
        grad = np.array([2.0 * x_, 2.0 * y_, 2.0 * (z_ + 1.0)])
        if not hessian_flag:
            return val, grad
        hess = np.diag([2.0, 2.0, 2.0])
        return val, grad, hess

    def g_x_nonneg(x):
        val = -x[0]
        grad = np.array([-1.0, 0.0, 0.0])
        hess = np.zeros((3, 3))
        return val, grad, hess

    def g_y_nonneg(x):
        val = -x[1]
        grad = np.array([0.0, -1.0, 0.0])
        hess = np.zeros((3, 3))
        return val, grad, hess

    def g_z_nonneg(x):
        val = -x[2]
        grad = np.array([0.0, 0.0, -1.0])
        hess = np.zeros((3, 3))
        return val, grad, hess

    ineq_constraints = [g_x_nonneg, g_y_nonneg, g_z_nonneg]

    A_eq = np.array([[1.0, 1.0, 1.0]])
    b_eq = np.array([1.0])

    x0 = np.array([0.3, 0.3, 0.4])
    return f_obj, ineq_constraints, A_eq, b_eq, x0


def get_linear_problem():

    def f_obj(x, hessian_flag: bool = True):

        val = -(x[0] + x[1])
        grad = np.array([-1.0, -1.0])
        if not hessian_flag:
            return val, grad
        hess = np.zeros((2, 2))
        return val, grad, hess

    def g1_feasible_line(x):
        val = -x[0] - x[1] + 1.0
        grad = np.array([-1.0, -1.0])
        hess = np.zeros((2, 2))
        return val, grad, hess

    def g2_y_le_1(x):
        val = x[1] - 1.0
        grad = np.array([0.0, 1.0])
        hess = np.zeros((2, 2))
        return val, grad, hess

    def g3_x_le_2(x):
        val = x[0] - 2.0
        grad = np.array([1.0, 0.0])
        hess = np.zeros((2, 2))
        return val, grad, hess

    def g4_y_nonneg(x):
        val = -x[1]
        grad = np.array([0.0, -1.0])
        hess = np.zeros((2, 2))
        return val, grad, hess

    ineq_constraints = [g1_feasible_line, g2_y_le_1, g3_x_le_2, g4_y_nonneg]

    A_eq = np.empty((0, 2))
    b_eq = np.empty((0,))

    x0 = np.array([1.0, 0.5])

    return f_obj, ineq_constraints, A_eq, b_eq, x0
