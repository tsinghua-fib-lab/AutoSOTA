"""
Defines active schedulers

Active schedulers are iterables which allow us to define which FBPINN subdomains
are active/fixed at each training step.

Each scheduler must inherit from the ActiveScheduler base class.
Each scheduler must define the NotImplemented methods.

This module is used by constants.py (and subsequently trainers.py)
"""

import numpy as np


class ActiveScheduler:
    """Base scheduler class to be inherited by different schedulers"""

    def __init__(self, all_params, n_steps):
        self.n_steps = n_steps
        self.m = all_params["static"]["decomposition"]["m"]
        self.xd = all_params["static"]["decomposition"]["xd"]

    def __len__(self):
        return self.n_steps

    def __iter__(self):
        """
        Returns None if active array not to be changed, otherwise active array.
        active is an array of length m, where each value corresponds
        to the state of each model (i.e. subdomain), which can be one of:

        0 = inactive (but still trained if it overlaps with active models)
        1 = active
        2 = fixed
        """

        raise NotImplementedError


class AllActiveSchedulerND(ActiveScheduler):
    "All models are active and training all of the time"

    def __iter__(self):
        for i in range(self.n_steps):
            if i == 0:
                yield np.ones(self.m, dtype=int)
            else:
                yield None


class _SubspacePointSchedulerRectangularND(ActiveScheduler):
    "Slowly expands radially outwards from a point in a subspace of a rectangular domain (in x units)"

    def __init__(self, all_params, n_steps, point, iaxes):
        super().__init__(all_params, n_steps)

        point = np.array(point)# point in constrained axes
        iaxes = list(iaxes)# unconstrained axes

        # validation
        if point.ndim != 1: raise Exception("ERROR: point.ndim != 1")
        if len(point) > self.xd: raise Exception("ERROR: len(point) > self.xd")
        if len(iaxes) + len(point) != self.xd: raise Exception("ERROR: len(iaxes) + len(point) != self.xd")

        # set set attributes
        self.point = point# (cd)
        self.iaxes = iaxes# (ucd)

        # get xmins, xmaxs
        self.xmins0 = all_params["static"]["decomposition"]["xmins0"].copy()# (m, xd)
        self.xmaxs0 = all_params["static"]["decomposition"]["xmaxs0"].copy()# (m, xd)

    def _get_radii(self, point, xmins, xmaxs):
        "Get the shortest distance from a point to a hypperectangle"

        # make sure subspace dimensions match with point
        assert xmins.shape[1] == xmaxs.shape[1] == point.shape[0]

        # broadcast point
        point = np.expand_dims(point, axis=0)# (1, cd)

        # whether point is inside model
        c_inside = (point >= xmins) & (point <= xmaxs)# (m, cd) point is broadcast
        c_inside = np.product(c_inside, axis=1).astype(bool)# (m) must be true across all dims

        # get closest point on rectangle to point
        pmin = np.clip(point, xmins, xmaxs)# (m, cd) point is broadcast

        # get furthest point on rectangle to point
        dmin, dmax = point-xmins, point-xmaxs# (m, cd) point is broadcast
        ds = np.stack([dmin, dmax], axis=0)# (2, m, cd)
        i = np.argmax(np.abs(ds), axis=0, keepdims=True)# (1, m, cd)
        pmax = point-np.take_along_axis(ds, i, axis=0)[0]# (m, cd) point is broadcast

        # get radii
        rmin = np.sqrt(np.sum((pmin-point)**2, axis=1))# (m) point is broadcast
        rmax = np.sqrt(np.sum((pmax-point)**2, axis=1))# (m) point is broadcast

        # set rmin=0 where point is inside model
        rmin[c_inside] = 0.

        return rmin, rmax

    def __iter__(self):

        # slice constrained axes
        ic = [i for i in range(self.xd) if i not in self.iaxes]
        xmins, xmaxs = self.xmins0[:,ic], self.xmaxs0[:,ic]# (m, cd)

        # get subspace radii
        rmin, rmax = self._get_radii(self.point, xmins, xmaxs)
        r_min, r_max = rmin.min(), rmax.max()

        # initialise active array, start scheduling
        active = np.zeros(self.m, dtype=int)# (m)
        for i in range(self.n_steps):

            # advance radius
            rt = r_min + (r_max-r_min)*(i/(self.n_steps))

            # get filters
            c_inactive = (active == 0)
            c_active   = (active == 1)# (m) active filter
            c_radius = (rt >= rmin) & (rt < rmax)# (m) circle inside box
            c_to_active = c_inactive & c_radius# c_radius is broadcast
            c_to_fixed = c_active & (~c_radius)# c_radius is broadcast

            # set values
            if c_to_active.any() or c_to_fixed.any():
                active[c_to_active] = 1
                active[c_to_fixed] = 2
                yield active
            else:
                yield None

class PointSchedulerRectangularND(_SubspacePointSchedulerRectangularND):
    "Slowly expands outwards from a point in the domain (in x units)"

    def __init__(self, all_params, n_steps, point):
        xd = all_params["static"]["decomposition"]["xd"]
        if len(point) != xd: raise Exception(f"ERROR: point incorrect shape {point.shape}")
        super().__init__(all_params, n_steps, point, iaxes=[])

class LineSchedulerRectangularND(_SubspacePointSchedulerRectangularND):
    "Slowly expands outwards from a line in the domain (in x units)"

    def __init__(self, all_params, n_steps, point, iaxis):
        xd = all_params["static"]["decomposition"]["xd"]
        if xd < 2: raise Exception("ERROR: requires nd >=2")
        if len(point) != xd-1: raise Exception(f"ERROR: point incorrect shape {point.shape}")
        super().__init__(all_params, n_steps, point, iaxes=[iaxis])

class PlaneSchedulerRectangularND(_SubspacePointSchedulerRectangularND):
    "Slowly expands outwards from a plane in the domain (in x units)"

    def __init__(self, all_params, n_steps, point, iaxes):
        xd = all_params["static"]["decomposition"]["xd"]
        if xd < 3: raise Exception("ERROR: requires nd >=3")
        if len(point) != xd-2: raise Exception(f"ERROR: point incorrect shape {point.shape}")
        super().__init__(all_params, n_steps, point, iaxes=iaxes)



if __name__ == "__main__":

    from fbpinns.decompositions import RectangularDecompositionND

    x = np.array([-6,-4,-2,0,2,4,6])

    subdomain_xs1 = [x]
    d1 = RectangularDecompositionND
    ps_ = d1.init_params(subdomain_xs1, [3*np.ones_like(x) for x in subdomain_xs1], (0,1))
    all_params1 = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}, "nm": tuple(len(x) for x in subdomain_xs1)}

    subdomain_xs2 = [x, x]
    d2 = RectangularDecompositionND
    ps_ = d2.init_params(subdomain_xs2, [3*np.ones_like(x) for x in subdomain_xs2], (0,1))
    all_params2 = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}, "nm": tuple(len(x) for x in subdomain_xs2)}

    subdomain_xs3 = [x, x, x]
    d3 = RectangularDecompositionND
    ps_ = d3.init_params(subdomain_xs3, [3*np.ones_like(x) for x in subdomain_xs3], (0,1))
    all_params3 = {"static":{"decomposition":ps_[0]}, "trainable":{"decomposition":ps_[1]}, "nm": tuple(len(x) for x in subdomain_xs3)}

    # test point
    for all_params in [all_params1, all_params2, all_params3]:
        xd, nm = all_params["static"]["decomposition"]["xd"], all_params["nm"]

        print("Point")
        point = np.array([0]*xd)
        A = PointSchedulerRectangularND(all_params, 100, point)
        for i, active in enumerate(A):
            if active is not None:
                print(i)
                print(active.reshape(nm))
        print()

    # test line
    for all_params in [all_params2, all_params3]:
        xd, nm = all_params["static"]["decomposition"]["xd"], all_params["nm"]

        print("Line")
        point = np.array([0]*(xd-1))
        A = LineSchedulerRectangularND(all_params, 100, point, 0)
        for i, active in enumerate(A):
            if active is not None:
                print(i)
                print(active.reshape(nm))
        print()

    # test plane
    for all_params in [all_params3]:
        xd, nm = all_params["static"]["decomposition"]["xd"], all_params["nm"]

        print("Plane")
        point = np.array([0]*(xd-2))
        A = PlaneSchedulerRectangularND(all_params, 100, point, [0,1])
        for i, active in enumerate(A):
            if active is not None:
                print(i)
                print(active.reshape(nm))
        print()