import pyterrier as pt

from pyterrier_dr import STAR, FlexIndex
encoder = STAR('/root/nfs/jpq/provided_models/star/')
index = FlexIndex('star')
pipeline = encoder >> index
pipeline.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())
