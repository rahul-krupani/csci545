import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.optimize import minimize

from collision import line_sphere_intersection
from drawing import WireframeSphere
from fk import fk


# arm link lengths
link_lengths = np.array([0.7, 1.0, 1.0])

# joint angle limits
bounds = [(-np.pi, np.pi)] * 3

# obstacle parameters
obstacle_center = np.array([0.6, 0.5, 0])
obstacle_radius = 0.2

# initial state (in configuration space)
q0 = np.array([0, 0, 1.86])

# goal state (in task space)
p_d  = np.array([0.1, 1.33, 0])

def objective(x):
    """
    Objective function that we want to minimize.

    :param x: state in configuration space, as numpy array with dtype
        np.float64
    :returns: a scalar value with type np.float64 representing the objective
        cost for x
    """
    # FILL in your code here
    position = np.zeros(3, dtype=np.float64)
    for index in range(len(x)):
        position[0] += link_lengths[index] * np.cos(sum(x[:index + 1]))
        position[1] += link_lengths[index] * np.sin(sum(x[:index + 1]))

    return np.dot((p_d - position).T, (p_d - position)) / 2

def constraint1(x):
    """
    Collision constraint for the first link.

    As an inequality constraint, the constraint is satisfied (meaning there is
    no collision) if the return value is non-negative.

    :param x: state in configuration space, as numpy array with dtype
        np.float64
    :returns: constraint output as a scalar value of type np.float64
    """
    # FILL in your code here
    p1 = np.array([0., 0., 0.])
    p2 = fk(angles=x[:1], link_lengths=link_lengths)

    return -1*line_sphere_intersection(p1=p1, p2=p2, c=obstacle_center, r=obstacle_radius)

def constraint2(x):
    """
    Collision constraint for the second link.

    As an inequality constraint, the constraint is satisfied (meaning there is
    no collision) if the return value is non-negative.

    :param x: state in configuration space, as numpy array with dtype
        np.float64
    :returns: constraint output as a scalar value of type np.float64
    """
    # FILL in your code here
    p1 = fk(angles=x[:1], link_lengths=link_lengths)
    p2 = fk(angles=x[:2], link_lengths=link_lengths)

    return -1*line_sphere_intersection(p1=p1, p2=p2, c=obstacle_center, r=obstacle_radius)

def constraint3(x):
    """
    Collision constraint for the third link.

    As an inequality constraint, the constraint is satisfied (meaning there is
    no collision) if the return value is non-negative.

    :param x: state in configuration space, as numpy array with dtype
        np.float64
    :returns: constraint output as a scalar value of type np.float64
    """
    # FILL in your code here
    p1 = fk(angles=x[:2], link_lengths=link_lengths)
    p2 = fk(angles=x[:1], link_lengths=link_lengths)

    return -1*line_sphere_intersection(p1=p1, p2=p2, c=obstacle_center, r=obstacle_radius)

# build constraints
con1 = {'type': 'ineq', 'fun': constraint1}
con2 = {'type': 'ineq', 'fun': constraint2}
con3 = {'type': 'ineq', 'fun': constraint3}
constraints = ([con1, con2, con3])

def solve_ik_with_cons(obj, q0, bnds, cons):
    """
    Call the scipy solver.

    :param obj: objective function
    :param q0: initial guess for solution
    :param bnds: list of lower and upper bound tuples for each parameter
    :param cons: optimization constraints
    :returns: solution state that minimizes the objective function
    """
    # show initial objective
    print('Initial SSE Objective: ' + str(objective(q0)))
    print(q0)

    # call optimizer
    solution = minimize(obj, q0, method='SLSQP', constraints=cons, bounds=bnds)
    x = solution.x

    # show final objective
    print('Final SSE Objective: ' + str(objective(x)))

    # print solution
    print('Solution')
    print('x1 = ' + str(x[0]))
    print('x2 = ' + str(x[1]))
    print('x3 = ' + str(x[2]))

    return x

def plot_solution(x):
    """
    Plot IK solution.

    :param x: solution state as a vector of joint angles
    :returns: None
    """
    fig = plt.figure()
    #ax = fig.gca(projection='3d')
    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # start position
    ax.scatter(0, 0, 0, color='r', s=100)

    # desired position
    ax.scatter(p_d[0], p_d[1], p_d[2], color='g', s=100)

    # plot robot
    points1 = fk(x[:1], link_lengths[:1])
    plt.plot([0, points1[0]], [0, points1[1]], [0, points1[2]], color='k')
    for pp in range(1, len(x)):
       points0 = fk(x[:pp], link_lengths[:pp])
       points1 = fk(x[:pp+1], link_lengths[:pp+1])
       plt.plot([points0[0], points1[0]],
                [points0[1], points1[1]],
                [points0[2], points1[2]],
                color='k')

    # add obstacle as sphere
    ax.plot_wireframe(
        *WireframeSphere(obstacle_center, obstacle_radius),
        color='k',
        alpha=0.5)
    ax.set(xlim=(-2, 2), ylim=(-2, 2), zlim=(-2, 2))

    # show result
    plt.show()

if __name__ == '__main__':
    solution = solve_ik_with_cons(objective, q0, bounds, constraints)
    plot_solution(solution)
