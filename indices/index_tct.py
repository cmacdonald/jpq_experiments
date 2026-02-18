import pyterrier as pt

from pyterrier_dr import TctColBert, FlexIndex
encoder = TctColBert.hnp()
index = FlexIndex('tct-hnp')
pipeline = encoder >> index
pipeline.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())
