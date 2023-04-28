import pandas as pd
from bamt.networks.continuous_bn import ContinuousBN
from gmr import GMM
from bamt.preprocessors import Preprocessor




class BNGenerator:
    def __init__(self, bn_structure: dict, n_components: int, gmm_parameters: dict):
        self.bn_structure = bn_structure
        self.n_components = n_components
        self.gmm_parameters = gmm_parameters
    def get_sample(self, size: int):
        GMM_model = GMM(n_components=self.n_components, priors=self.gmm_parameters['weights'], means=self.gmm_parameters['means'], covariances=self.gmm_parameters['covs'])
        gmm_sample = GMM_model.sample(1000)
        sample_df = pd.DataFrame(gmm_sample, columns=self.bn_structure['V'])
        bn = ContinuousBN(use_mixture=True)
        p = Preprocessor([])
        _, _ = p.apply(sample_df)
        bn.add_nodes(p.info)
        bn.set_structure(edges=self.bn_structure['E'])
        bn.fit_parameters(sample_df)
        sample = bn.sample(size)
        return sample






