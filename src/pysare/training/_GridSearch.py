
import pandas as pd
import numpy as np 
from tqdm.autonotebook import tqdm

class GridSearch():
    r"""Creates and executes grid searches.

    
    """
    def __init__(self, jobs:pd.DataFrame = None, parameters: dict = None) -> None:

        if isinstance(jobs, pd.DataFrame):
            self.jobs = jobs
        elif jobs is None:
            self.jobs = None
        else:
            raise ValueError("Argument jobs must be either DataFrame or None")
        
        if isinstance(parameters, dict):
            for key, values in parameters.items():
                self.add_parameter(key, values)
        elif parameters is not None:
            raise ValueError("Argument parameters must be either dict or None")
        
        self.result = None

    def add_parameter(self, key, values):
        r"""Adds a parameter/column to the list of jobs.
        
        
        """
        if self.jobs is None:
            self.jobs = pd.DataFrame(data={key: values})
        else:
            if np.isscalar(values[0]):
                new_col =  np.array(values).repeat(len(self.jobs), axis=0)
            else:
                new_col = list(values)
                new_col.append([])
                new_col =  np.array(new_col,dtype=object)[:-1].repeat(len(self.jobs), axis=0)
            self.jobs = pd.concat([self.jobs]*len(values), ignore_index=True)
            self.jobs[key] = new_col

    # def add_generator(self, key, generator):
    #     pass
    # def add_random(self, key, data, method):
    #     pass
    
    # def generate(self, num_rows, num_blocks):
    #     pass

    def run_jobs(self,experiment, suppress_printing=False):
        r"""Runs the grid search.
        
        Parameters
        ----------
        
        experiment : callable
            A function that takes one job (row in jobs) as input and outputs 
            one or two dicts. The first dict is aggregated in the main process 
            and included in the return results and the second dict is only 
            saved to file.
        """
        res = []

        
        for ind, job in tqdm(self.jobs.iterrows(), total=self.jobs.shape[0]):
            # with nostdout(supress_printing):
                res.append(experiment(**job))

        self.result = pd.concat((self.jobs,pd.DataFrame.from_records(res)), axis=1)
        return self.result