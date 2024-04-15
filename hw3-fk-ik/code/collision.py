import numpy as np


def line_sphere_intersection(p1, p2, c, r):
    """
    Implements the line-sphere intersection algorithm.
    https://en.wikipedia.org/wiki/Line-sphere_intersection

    :param p1: start of line segment
    :param p2: end of line segment
    :param c: sphere center
    :param r: sphere radius
    :returns: discriminant (value under the square root) of the line-sphere
        intersection formula, as a np.float64 scalar
    """
    # FILL in your code here
    # [u dot (o-c)]^2 - |u|^2(|o-c|^2-r^2)

    u = p2 - p1
    u = u/np.linalg.norm(u)
    a = np.dot(u,p1-c)**2
    b = (np.linalg.norm(p1-c)**2)-r**2

    discriminant = (a-b)

    return discriminant