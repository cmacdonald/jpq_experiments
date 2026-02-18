import pyterrier as pt

from pyterrier_dr import TasB, FlexIndex
encoder = TasB.dot()
index = FlexIndex('tasb')
pipeline = encoder >> index
pipeline.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())
