"""
Variable module
"""
from typing import List, Dict
import numpy as np
import regmod


class ModelVariables:
    def __init__(self,
                 variables: List[regmod.variable.Variable],
                 model_type: str = "Poisson"):
        self.variables = variables
        self.var_names = [variable.name for variable in self.variables]
        self.var_dict = {var_name: self.variables[i]
                         for i, var_name in enumerate(self.var_names)}
        self.model_type = model_type
        if self.model_type not in ["Poisson", "Linear"]:
            raise ValueError("`model_type` has to be 'Poisson' or 'Linear'.")

    def add_priors(self, priors: Dict[str, regmod.prior.Prior]):
        for var_name, prior in priors.items():
            if var_name in self.var_names:
                self.var_dict[var_name].add_priors(prior)

    def get_model(self, data: regmod.data.Data) -> regmod.model.Model:
        if self.model_type == "Poisson":
            model = regmod.model.PoissonModel(data, self.variables, use_offset=True)
        else:
            model = regmod.model.LinearModel(data, self.variables, use_offset=True)
        return model

    def result_to_priors(self,
                         result: Dict[str, np.ndarray],
                         min_var: float = 0.1) -> Dict[str, regmod.prior.Prior]:
        mean = result["coefs"]
        sd = np.sqrt(np.maximum(min_var, np.diag(result["vcov"])))
        slices = regmod.utils.sizes_to_sclices([var.size for var in self.variables])
        return {
            var_name: regmod.prior.GaussianPrior(mean=mean[slices[i]], sd=sd[slices[i]])
            for i, var_name in enumerate(self.var_names)
        }


class SeasonalityModelVariables(ModelVariables):
    def __init__(self, variables: List[regmod.variable.Variable], **kwargs):
        super().__init__(variables, **kwargs)
        if not ("week" in self.var_names or "month" in self.var_names):
            raise ValueError("SeasonalityModelVariables must include 'week' or 'month'.")
        dtime = "week" if "week" in self.var_names else "month"
        self.dtime_var = self.var_dict[dtime]

    def get_model(self, data: regmod.data.Data) -> regmod.model.PoissonModel:
        self.dtime_var.check_data(data)
        mat = np.vstack([
            self.dtime_var.spline.design_dmat(self.dtime_var.spline.knots[0], order=i) -
            self.dtime_var.spline.design_dmat(self.dtime_var.spline.knots[-1] + 1, order=i, r_extra=True)
            for i in range(1)
        ])
        self.dtime_var.add_priors(regmod.prior.LinearUniformPrior(mat=mat, ub=0.0, lb=0.0))
        return super().get_model(data)


class TimeModelVariables(ModelVariables):
    def __init__(self, variables: List[regmod.variable.Variable], **kwargs):
        super().__init__(variables, **kwargs)
        if "time" not in self.var_names:
            raise ValueError("TimeModelVariables must include 'time'.")
        self.time_var = self.var_dict["time"]
