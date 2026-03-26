import os
import dill
import numpy as np
from neuralsurv.prediction import NeuralSurvPredict
from neuralsurv.fit import NeuralSurvFit
from neuralsurv.tools import NeuralSurvTools


class NeuralSurv(NeuralSurvPredict, NeuralSurvFit, NeuralSurvTools):

    def __init__(
        self,
        model,
        model_params_init,
        alpha_prior,
        beta_prior,
        rho,
        num_points_integral_em,
        num_points_integral_cavi,
        batch_size,
        max_iter_em,
        max_iter_cavi,
        output_dir,
        overwrite_em=False,
        overwrite_cavi=False,
    ):

        self.model = model
        self.model_params_init = model_params_init
        self.alpha_prior = alpha_prior
        self.beta_prior = beta_prior
        self.rho = rho
        self.num_points_integral_em = num_points_integral_em
        self.num_points_integral_cavi = num_points_integral_cavi
        self.batch_size = batch_size
        self.max_iter_em = max_iter_em
        self.max_iter_cavi = max_iter_cavi
        self.output_dir = output_dir
        self.neuralsurv_dir = output_dir + "/fit_neuralsurv.pkl"
        self.overwrite_em = overwrite_em
        self.overwrite_cavi = overwrite_cavi

        NeuralSurvPredict.__init__(self)
        NeuralSurvFit.__init__(self)
        NeuralSurvTools.__init__(self)

    @classmethod
    def load_or_create(
        cls,
        model,
        model_params_init,
        alpha_prior,
        beta_prior,
        rho,
        num_points_integral_em,
        num_points_integral_cavi,
        batch_size,
        max_iter_em,
        max_iter_cavi,
        output_dir,
        overwrite_em=False,
        overwrite_cavi=False,
    ):
        neuralsurv_dir = output_dir + "/fit_neuralsurv.pkl"
        if os.path.isfile(neuralsurv_dir):
            with open(neuralsurv_dir, "rb") as f:
                obj = dill.load(f)
            # Overwrite after loading
            obj.num_points_integral_em = num_points_integral_em
            obj.num_points_integral_cavi = num_points_integral_cavi
            obj.batch_size = batch_size
            obj.max_iter_em = max_iter_em
            obj.max_iter_cavi = max_iter_cavi
            obj.overwrite_em = overwrite_em
            obj.overwrite_cavi = overwrite_cavi
            obj.model = model
            obj.output_dir = output_dir
            obj.neuralsurv_dir = neuralsurv_dir
            obj.model_params_init = model_params_init
            return obj
        else:
            return cls(
                model,
                model_params_init,
                alpha_prior,
                beta_prior,
                rho,
                num_points_integral_em,
                num_points_integral_cavi,
                batch_size,
                max_iter_em,
                max_iter_cavi,
                output_dir,
                overwrite_em,
                overwrite_cavi,
            )

    def save(self):
        print("Saving...")

        # Temporarily remove the model (otherwise can lead to version issue)
        model = self.model
        model_params_init = self.model_params_init
        theta_MAP = self.theta_MAP
        Z = self.Z
        self.model = None
        self.model_params_item = None
        self.model_params_init = None
        self.Z = None
        self.theta_map = np.array(theta_MAP)

        # Save the object without the model
        with open(self.neuralsurv_dir, "wb") as f:
            dill.dump(self, f)

        # Restore the model
        self.model = model
        self.model_params_init = model_params_init
        self.theta_map = theta_MAP
        self.Z = Z
