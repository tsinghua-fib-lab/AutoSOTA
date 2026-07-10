# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 16:16:53 2021
Last update Monday Jan 31 1:15 pm 2025

This file uses the rpy2 package to call the generalized random forest R
package via the `grf` library.
"""

import gc
import uuid
import numpy as np

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector
from rpy2.robjects import numpy2ri

# --- R package setup --------------------------------------------------------

# import R's "base" package
base = importr("base")

# import R's "utils" package
utils = rpackages.importr("utils")

# select a mirror for R packages
utils.chooseCRANmirror(ind=1)  # select the first mirror in the list

# R package names
packnames = ("grf",)

# Selectively install what needs to be installed
names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
if len(names_to_install) > 0:
    utils.install_packages(StrVector(names_to_install))

# import the grf package
grf = importr("grf")

# activate automatic numpy <-> R conversion
numpy2ri.activate()


# --- Python wrapper classes -------------------------------------------------


class regression_forest:
    """
    Wrapper around grf::regression_forest for nuisance regression (gamma).
    """

    def __init__(self):
        self.f = None
        # unique prefix for R objects
        self.id = "a" + str(uuid.uuid4()).replace("-", "_")

    def fit(self, X, Y):
        gc.collect()
        Y = Y.reshape(len(Y), 1)

        # assign data to R
        robjects.r.assign(self.id + "rX", X)
        robjects.r.assign(self.id + "rY", Y)


        robjects.r(
            f"""
            {self.id}f <- regression_forest(
                {self.id}rX,
                {self.id}rY,
                tune.parameters = "all",
                num.threads = 1,
                seed = 42
            )
            """
        )
        # we don't really need to keep the R object in self.f,
        # just record that fit was called
        self.f = True
        return self

    def predict(self, X):
        # assign new X to R
        robjects.r.assign(self.id + "rXp", X)

        # IMPORTANT: return the predictions *directly* as the last expression,
        # not via an assignment like "pred = ...", which would return NULL.
        yhat = robjects.r(
            f"""
            predict({self.id}f, {self.id}rXp)$predictions
            """
        )
        yhat = np.array(yhat)
        return yhat

    def clear(self):
        # remove R objects associated with this instance
        robjects.r(
            f"""
            if (exists("{self.id}f"))   rm({self.id}f)
            if (exists("{self.id}rX"))  rm({self.id}rX)
            if (exists("{self.id}rY"))  rm({self.id}rY)
            if (exists("{self.id}rXp")) rm({self.id}rXp)
            gc()
            """
        )
        gc.collect()


class regression_forest2:
    """
    Wrapper around grf::regression_forest typically used for GPS estimation.
    Uses fixed R object names (g, rX2, rY2, rXp2) but is used in a single
    fit -> predict cycle inside DDMLCT, so this is OK.
    """

    def __init__(self):
        self.f = None

    def fit(self, X, Y):
        gc.collect()
        Y = Y.reshape(len(Y), 1)

        robjects.r.assign("rX2", X)
        robjects.r.assign("rY2", Y)


        robjects.r(
            """
            g <- regression_forest(
                rX2,
                rY2,
                tune.parameters = "all",
                num.threads = 1,
                seed = 42
            )
            """
        )
        self.f = True
        return self

    def predict(self, X):
        robjects.r.assign("rXp2", X)

        # again, return predictions directly as the last expression
        yhat = robjects.r(
            """
            predict(g, rXp2)$predictions
            """
        )
        yhat = np.array(yhat)

        # clean up R objects
        robjects.r(
            """
            if (exists("g"))    rm(g)
            if (exists("rXp2")) rm(rXp2)
            if (exists("rX2"))  rm(rX2)
            if (exists("rY2"))  rm(rY2)
            gc()
            """
        )
        gc.collect()
        return yhat
