import numpy as np
import random

class TestingSetCreator:
    def __init__(self, random_seed):
        self.building_blocks = self.create_building_blocks()
        self.random_seed = random_seed

    def create_building_blocks(self):
        """
        Returns testing set with nan-values. Simplifies the testing procedure. 
        Limits itself to three columns. The third column is always a random continuous variable.

        Parameters
        ----------
        complete: int - how many complete rows should the testing set have.
        spat1: int - how many rows with pattern [np.nan, 1, 1]
        spat2: int - how many rows with pattern [1, np.nan, 1]
        spat3: int - how many rows with pattern [1, 1, np.nan]
        mpat1: int - how many rows with pattern [np.nan, np.nan, 1]
        mpat2: int - how many rows with pattern [np.nan, 1, np.nan]
        mpat3: int - how many rows with pattern [1, np.nan, np.nan]
        allnan: int - how many rows with pattern [np.nan, np.nan, np.nan]

        Returns
        -------
        dict with building blocks
        """
        building_blocks = {}

        building_blocks["complete"] = [1, 1, random.random()]
        building_blocks["spat1"] = [np.nan, 1, random.random()]
        building_blocks["spat2"] = [1, np.nan, random.random()]
        building_blocks["spat3"] = [1, 1, np.nan]

        building_blocks["mpat1"] = [np.nan, np.nan, random.random()]
        building_blocks["mpat2"] = [np.nan, 1, np.nan]
        building_blocks["mpat3"] = [1, np.nan, np.nan]
        building_blocks["allnan"] = [np.nan, np.nan, np.nan]

        return building_blocks

    #def add_building_block(self, building_block, building_block_name):
    #    self.building_blocks[building_block_name] = building_block

    def test_set(self, complete=0, spat1=0, spat2=0, spat3=0, mpat1=0, mpat2=0, mpat3=0, allnan=0, seed=23):
        random.seed(seed)
        attrs_name = ["complete", "spat1", "spat2", "spat3", "mpat1", "mpat2", "mpat3", "allnan"]
        attrs = [complete, spat1, spat2, spat3, mpat1, mpat2, mpat3, allnan]
        ret_set = []

        for idx, attr in enumerate(attrs):
            for i in range(attr):
                tmp_block = list(self.building_blocks[attrs_name[idx]])
                if attrs_name[idx] in ["complete", "spat1", "spat2", "mpat1"]:
                    tmp_block[2] = random.random()
                ret_set.append(tmp_block)
        return ret_set
