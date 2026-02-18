import pyterrier as pt
import torch
torch.set_float32_matmul_precision('high')
from pyterrier_dr import RepLLama, FlexIndex
encoder = RepLLama.v1_7b()
encoder.model.compile()
index = FlexIndex('repllama7b')
pipeline = encoder >> index
pipeline.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())
