"""
Cascade module
"""
from copy import deepcopy
from typing import List, Any, Dict, Union
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
import regmod

from emmodel.variable import ModelVariables
from emmodel.model import ExcessMortalityModel


@dataclass
class CascadeSpecs:
    model_variables: List[ModelVariables]
    prior_masks: Dict[str, np.ndarray]
    level_masks: List[float]


@dataclass
class Cascade:
    df: pd.DataFrame
    specs: CascadeSpecs
    level_id: int = 0
    children: List["Cascade"] = field(default_factory=list)
    name: str = "unknown"

    model_variables: List[ModelVariables] = field(default_factory=list, init=False)
    model: ExcessMortalityModel = field(default=None, init=False)
    priors: Dict = field(default_factory=dict, init=False)

    def __post_init__(self):
        self.model_variables = deepcopy(self.specs.model_variables)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_children(self, children: Union["Cascacde", List["Cascade"]]):
        if isinstance(children, Cascade):
            self.children.append(children)
        elif isinstance(children, list) and all([isinstance(child, Cascade) for child in children]):
            self.children.extend(children)
        else:
            raise ValueError("Invalid children type.")

    def to_list(self) -> List:
        current_list = [self]
        if not self.is_leaf():
            current_list.append([
                child.to_list[0]
                for child in self.children
            ])
        return current_list

    def set_priors(self, priors: Dict[str, regmod.prior.Prior]):
        self.priors = priors

    def add_priors_to_variables(self):
        for variables in self.model_variables:
            variables.add_priors(self.priors)

    def get_model(self):
        self.model = ExcessMortalityModel(self.df, self.model_variables)

    def get_priors(self) -> Dict[str, regmod.prior.Prior]:
        priors = {}
        for i, variables in enumerate(self.model_variables):
            priors.update(variables.result_to_priors(self.model.results[i]))
        for var_name, mask in self.specs.prior_masks.items():
            priors[var_name].sd *= mask
        for prior in priors.values():
            prior.sd *= self.specs.level_masks[self.level_id]
        return priors

    def run_models(self):
        self.add_priors_to_variables()
        self.get_model()
        self.model.run_models()

        if not self.is_leaf():
            priors = self.get_priors()
            for child in self.children:
                child.set_priors(priors)
                child.run_models()
