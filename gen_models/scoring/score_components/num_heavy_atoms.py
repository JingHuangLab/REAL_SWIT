import numpy as np
from rdkit.Chem.Descriptors import HeavyAtomCount
from typing import List

from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_summary import ComponentSummary

from scoring.score_transformations import TransformationFactory
from utils.enums.transformation_type_enum import TransformationTypeEnum


class HeavyAtomScore(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.transform_function = self._assign_transformation(
            self.parameters.specific_parameters
        )

    def calculate_score(self, molecules: List) -> ComponentSummary:
        raw_score = self._calculate_heavy_atoms(molecules)
        transformed_score = self.transform_function(raw_score, self.parameters.specific_parameters)
        score_summary = ComponentSummary(
            total_score=transformed_score,
            parameters=self.parameters
        )
        return score_summary

    def _calculate_heavy_atoms(self, query_mols) -> np.array:
        heavy_atom_counts = []
        for mol in query_mols:
            try:
                count = HeavyAtomCount(mol)
            except:
                count = 0
            heavy_atom_counts.append(count)
        return np.array(heavy_atom_counts, dtype=np.float32)
    
    def _assign_transformation(self, specific_parameters: {}):
        transformation_type = TransformationTypeEnum()
        factory = TransformationFactory()
        if self.parameters.specific_parameters[self.component_specific_parameters.TRANSFORMATION]:
            transform_function = factory.get_transformation_function(specific_parameters)
        else:
            self.parameters.specific_parameters[
                self.component_specific_parameters.TRANSFORMATION_TYPE] = transformation_type.NO_TRANSFORMATION
            transform_function = factory.no_transformation
        return transform_function

