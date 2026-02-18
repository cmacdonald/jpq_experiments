import pyterrier as pt

from pyterrier_dr import Dragon, FlexIndex
encoder = Dragon()
index = FlexIndex('dragon')
pipeline = encoder >> index
pipeline.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())
