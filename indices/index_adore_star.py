import pyterrier as pt

from pyterrier_dr import STAR, FlexIndex
encoder = STAR('/root/nfs/jpq/provided_models/adore_star/')
index = FlexIndex('adore_star')
pipeline = encoder >> index
pipeline.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())
