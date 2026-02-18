import pyterrier as pt
import torch
#torch.set_float32_matmul_precision('high')
from pyterrier_dr.lion import LionLlamaDense
from pyterrier_dr import FlexIndex
encoder = LionLlamaDense()
encoder.model.compile()
index = FlexIndex('lion')
pipeline = encoder >> index
pipeline.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())
