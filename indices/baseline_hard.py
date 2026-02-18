import sys
if len(sys.argv) == 1:
    print("Usage: %s [tasb|repllama|star|e5|dragon|lion] flexindexpath" % sys.argv[0])
    sys.exit(1)
modelname, path = sys.argv[1:]

import os
import pyterrier as pt
import pyterrier_dr
from pyterrier.measures import *

if modelname == 'tasb':
    model = pyterrier_dr.TasB()
if modelname == 'tct-hnp':
    model = pyterrier_dr.TctColBert.hnp()
elif modelname == 'repllama':
    import torch
    torch.set_float32_matmul_precision('high')
    model = pyterrier_dr.RepLLama.v1_7b()
elif modelname == 'lion':
    import torch
    torch.set_float32_matmul_precision('high')
    model = pyterrier_dr.LionLlamaDense()
elif modelname == 'e5':
    model = pyterrier_dr.E5()
elif modelname == 'dragon':
    model = pyterrier_dr.Dragon()
elif modelname == 'adore_star':
    model = pyterrier_dr.STAR('/root/nfs/jpq/provided_models/adore_star/')
elif modelname == 'star':
    model = pyterrier_dr.STAR('/root/nfs/jpq/provided_models/star/')
else:
    raise KeyError("unknown modelname " + modelname)

index = pyterrier_dr.FlexIndex(path)
retr_pipe = model >> index.np_retriever()

def get_exp(ds):
    eval_metrics=[RR@10, Recall(rel=2)@100, Recall@100, nDCG@10, "mrt"]
    if ds == 'trec2019':
        dsid = 'msmarco-passage/trec-dl-2019/judged'
    elif ds == 'trec2020':
        dsid = 'msmarco-passage/trec-dl-2020/judged' 
    elif ds == 'dev':
        eval_metrics=[RR@10, Recall@100, "mrt"]
        dsid = 'msmarco-passage/dev/small'
    elif ds == 'hard':
        dsid = "msmarco-passage/trec-dl-hard"
    dataset = pt.get_dataset('irds:'+dsid)
    return dataset.get_topics(), dataset.get_qrels(), eval_metrics

for ds in ['hard']:
    t,q,e = get_exp(ds)
    save_dir = "./baselines/"+ds + "_" + modelname
    os.makedirs(save_dir, exist_ok=True)
    df = pt.Experiment(
            [retr_pipe],
            t,
            q,
            eval_metrics=e,
            save_dir = save_dir,
            names=["baseline_" + modelname]
    )
    print(ds)
    print(df)
    df.to_csv(save_dir + "/" + 'metrics.csv')

