"""
Variable module
"""
from typing import List, Dict
import numpy as np
from regmod.variable import Variable
from regmod.parameter import Parameter
from regmod.prior import Prior, GaussianPrior, LinearUniformPrior
from regmod.models import Model, PoissonModel, GaussianModel
from regmod.data import Data
from regmod.utils import sizes_to_sclices


class ModelVariables:
    def __init__(self,
                 variables: List[Variable],
                 model_type: str = "Poisson"):
        self.variables = variables
        self.var_names = [variable.name for variable in self.variables]
        self.var_dict = {var_name: self.variables[i]
                         for i, var_name in enumerate(self.var_names)}
        self.model_type = model_type
        if self.model_type not in ["Poisson", "Linear"]:
            raise ValueError("`model_type` has to be 'Poisson' or 'Linear'.")

    def add_priors(self, priors: Dict[str, Prior]):
        for var_name, prior in priors.items():
            if var_name in self.var_names:
                self.var_dict[var_name].add_priors(prior)

    def get_model(self, data: Data) -> Model:
        if self.model_type == "Poisson":
            model = PoissonModel(data, param_specs={"lam": {"variables": self.variables,
                                                            "use_offset": True}})
        else:
            model = GaussianModel(data, param_specs={"mu": {"variables": self.variables,
                                                            "use_offset": True}})
        return model

    def result_to_priors(self,
                         result: Dict[str, np.ndarray],
                         min_var: float = 0.1) -> Dict[str, Prior]:
        mean = result["coefs"]
        sd = np.sqrt(np.maximum(min_var, np.diag(result["vcov"])))
        slices = sizes_to_sclices([var.size for var in self.variables])
        return {
            var_name: GaussianPrior(mean=mean[slices[i]], sd=sd[slices[i]])
            for i, var_name in enumerate(self.var_names)
        }


class SeasonalityModelVariables(ModelVariables):
    def __init__(self,
                 variables: List[Variable],
                 col_time: str,
                 smooth_order: int,
                 **kwargs):
        super().__init__(variables, **kwargs)
        if not col_time in self.var_names:
            raise ValueError(f"SeasonalityModelVariables must include {col_time}.")
        self.time_var = self.var_dict[col_time]
        self.smooth_order = smooth_order

    def get_model(self, data: Data) -> PoissonModel:
        self.time_var.check_data(data)
        mat = np.vstack([
            self.time_var.spline.design_dmat(self.time_var.spline.knots[0], order=i) -
            self.time_var.spline.design_dmat(self.time_var.spline.knots[-1] + 1, order=i, r_extra=True)
            for i in range(self.smooth_order)
        ])
        self.time_var.add_priors(LinearUniformPrior(mat=mat, ub=0.0, lb=0.0))
        return super().get_model(data)


class TimeModelVariables(ModelVariables):
    def __init__(self, variables: List[Variable], **kwargs):
        super().__init__(variables, **kwargs)
        if "time" not in self.var_names:
            raise ValueError("TimeModelVariables must include 'time'.")
        self.time_var = self.var_dict["time"]
