import logging
from typing import List
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit.Chem.rdMolDescriptors import CalcNumRings

from scoring.component_parameters import ComponentParameters
from scoring.score_components.base_score_component import BaseScoreComponent
from scoring.score_summary import ComponentSummary
from scoring.score_transformations import TransformationFactory
from utils.enums.transformation_type_enum import TransformationTypeEnum

class MaxLinkerLength(BaseScoreComponent):
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self._transformation_function = self._assign_transformation(self.parameters.specific_parameters)
        self.logger = logging.getLogger(__name__)
        self.bad_linker_patts = [Chem.MolFromSmarts('I' + '~[A;!R]'*n + '~I') for n in range(20, 0, -1)]

    def calculate_score(self, molecules: List) -> ComponentSummary:
        
        scores = []
        for mol in molecules:
            if not mol:
                self.logger.warning(f'failed to convert {mol} to mol')
                score = 10
            else:
                score = self.max_linker_length(mol)
            scores.append(score)
        total_score = self._transformation_function(scores, self.parameters.specific_parameters)
        score_summary = ComponentSummary(total_score=total_score, parameters=self.parameters)
        return score_summary
    
    def split_mol(self,mol):
        if not mol:
            self.logger.warning(f'pass None to split_mol, return None')
            #print(f'pass None to split_mol, return None')
            return None, None

        if not CalcNumRings(mol):
            return None, [mol]

        for bond in mol.GetBonds():
            bond.SetIntProp("orig_idx", bond.GetIdx())

        try:
            scaffold = GetScaffoldForMol(mol)
        except:
            self.logger.warning(f'get murcko scaffold failed for {Chem.MolToSmiles(mol)}')
            #print(f'get murcko scaffold failed for {Chem.MolToSmiles(mol)}')
            return None, [mol]
        else:
            pass

        cut_bonds = list()
        for bond in scaffold.GetBonds():
            if scaffold.GetAtomWithIdx(bond.GetBeginAtomIdx()).IsInRing() + scaffold.GetAtomWithIdx(bond.GetEndAtomIdx()).IsInRing() == 1:
                cut_bonds.append(bond.GetIntProp("orig_idx"))

        if not cut_bonds:
            return None, [mol]

        try:
            frgs = Chem.GetMolFrags(Chem.FragmentOnBonds(mol, cut_bonds, addDummies=True), asMols=True)
        except:
            self.logger.warning(f'get GetMolFrags failed for {Chem.MolToSmiles(mol)}')
            #print(f'get GetMolFrags failed for {Chem.MolToSmiles(mol)}')
            return None, [mol]
        else:
            linkers, nonlinkers = list(), list()

            for frg in frgs:
                if not CalcNumRings(frg):
                    frg = Chem.ReplaceSubstructs(frg, Chem.MolFromSmiles('*'), Chem.MolFromSmiles('I'), replaceAll=True)[0]
                    Chem.GetSymmSSSR(frg)
                    linkers.append(frg)
                else:
                    nonlinkers.append(frg)

            return linkers, nonlinkers

    def max_linker_length(self,mol):
        linkers = self.split_mol(mol)[0]
        if not linkers:
            return 0

        for patt in self.bad_linker_patts:
            if any([m.HasSubstructMatch(patt) for m in linkers]):
                return patt.GetNumAtoms() - 2

        return 0
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