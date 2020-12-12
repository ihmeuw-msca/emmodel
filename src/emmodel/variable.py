"""
Variable module
"""
from typing import List, Dict
import numpy as np
import regmod


class ModelVariables:
    def __init__(self, variables: List[regmod.variable.Variable]):
        self.variables = variables
        self.var_names = [variable.name for variable in self.variables]
        self.var_dict = {var_name: self.variables[i]
                         for i, var_name in enumerate(self.var_names)}

    def add_priors(self, priors: Dict[str, regmod.prior.Prior]):
        for var_name, prior in priors.items():
            if var_name in self.var_dict.keys():
                self.var_dict[var_name].add_priors(prior)

    def get_model(self, data: regmod.data.Data) -> regmod.model.PoissonModel:
        return regmod.model.PoissonModel(data, self.variables, use_offset=True)

    def result_to_priors(self, result: Dict[str, np.ndarray]) -> Dict[str, regmod.prior.Prior]:
        mean = result["coefs"]
        sd = np.sqrt(np.diag(result["vcov"]))
        slices = regmod.utils.sizes_to_sclices([var.size for var in self.variables])
        return {
            var_name: regmod.prior.GaussianPrior(mean=mean[slices[i]], sd=sd[slices[i]])
            for i, var_name in enumerate(self.var_names)
        }


class YearModelVariables(ModelVariables):
    def __init__(self, variables: List[regmod.variable.Variable]):
        super().__init__(variables)
        if not ("week" in self.var_names or "month" in self.var_names):
            raise ValueError("YearModelVariables must include 'week' or 'month'.")
        tunit = "week" if "week" in self.var_names else "month"
        self.tunit_var = self.var_dict[tunit]

    def get_model(self, data: regmod.data.Data) -> regmod.model.PoissonModel:
        self.tunit_var.check_data(data)
        mat = np.vstack([
            self.tunit_var.spline.design_dmat(self.tunit_var.spline.knots[0], order=i) -
            self.tunit_var.spline.design_dmat(self.tunit_var.spline.knots[-1] + 1, order=i, r_extra=True)
            for i in range(1)
        ])
        self.tunit_var.add_priors(regmod.prior.LinearUniformPrior(mat=mat, ub=0.0, lb=0.0))
        return regmod.model.PoissonModel(data, self.variables, use_offset=True)


class TimeModelVariables(ModelVariables):
    def __init__(self, variables: List[regmod.variable.Variable]):
        super().__init__(variables)
        if "time" not in self.var_names:
            raise ValueError("TimeModelVariables must include 'time'.")
        self.time_var = self.var_dict["time"]
