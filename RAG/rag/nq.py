import random
import json
from pathlib import Path
from functools import cache
from collections import defaultdict
from collections.abc import Iterator, Callable

import datasets
import pandas as pd
import pyterrier as pt
from tqdm import tqdm
from ir_datasets.formats import GenericDocPair


@cache
def _load(split: str) -> datasets.Dataset:
    return datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "nq", split=split)

def get_topics(split: str) -> pd.DataFrame:
    dataset = _load(split)
    df = dataset.to_pandas().drop("golden_answers", axis=1)
    return df.rename({"id": "qid", "question": "query"}, axis=1)

def get_answers(split: str) -> pd.DataFrame:
    dataset = _load(split)
    data = []
    index = []
    for idx, item in enumerate(dataset):
        qid = item["id"]
        gold_answers = item["golden_answers"]
        for gold_answer in gold_answers:
            data.append([qid, gold_answer])
            index.append(idx)
    
    return pd.DataFrame(data, index=index, columns=["qid", "gold_answer"])

def _init_bm25() -> pt.Transformer:
    sparse_index = pt.Artifact.from_hf('pyterrier/ragwiki-terrier')
    bm25 = (
        pt.rewrite.tokenise()
        >> sparse_index.bm25(
            include_fields=["docno", "text", "title"],
            num_results=100,
        )
        >> pt.rewrite.reset()
    )
    return bm25

def _try_import_pyserini_has_answers() -> Callable:
    try:
        from pyserini.eval.evaluate_dpr_retrieval import has_answers
    except Exception as e:
        raise ImportError("require `pyserini` for reproducing DPR data processing") from e
    
    return has_answers


class NQDevSet:
    """Dev set for JPQ."""
    def __init__(self, qrels_fp: str) -> None:
        self._qrels_fp = Path(qrels_fp)
        self._qrels = None

        if self._qrels_fp.exists():
            self._load_qrels()

    def get_topics(self) -> pd.DataFrame:
        topics = get_topics("dev")
        qrels = self._load_qrels()

        qids = set(qrels.qid)
        return topics[topics.qid.isin(qids)]

    def get_qrels(self) -> pd.DataFrame:
        if not self._qrels_fp.exists():
            self._save_qrels()

        return self._load_qrels()

    def _load_qrels(self) -> pd.DataFrame:
        if self._qrels is not None:
            return self._qrels

        self._qrels = pd.read_json(self._qrels_fp)
        self._qrels.docno = self._qrels.docno.astype("string")
        return self._qrels

    def _save_qrels(self) -> None:
        qrels = self._prepare_qrels()
        qrels.to_json(self._qrels_fp)

    def _prepare_qrels(self) -> pd.DataFrame:
        has_answers = _try_import_pyserini_has_answers()
        bm25 = _init_bm25()

        dev_topics = get_topics("dev")
        dev_answers = get_answers("dev")
        qid2answers = defaultdict(list)

        for row in dev_answers.itertuples():
            qid2answers[row.qid].append(str(row.gold_answer))

        qrels = []
        for row in tqdm(
            dev_topics.itertuples(),
            total=len(dev_topics),
            desc="preparing qrels",
        ):
            for item in bm25.search(row.query).itertuples():
                if has_answers(
                    item.text,
                    qid2answers[row.qid],
                    tokenizer=None,
                    regex=True,
                ):
                    qrels.append(
                        {
                            "qid": row.qid,
                            "docno": item.docno,
                            "label": 1,
                            "iteration": 0,
                        }
                    )

        return pd.DataFrame(qrels)


class NQTrainSet:
    """Training set for JPQ."""
    def __init__(self, docpairs_fp: str) -> None:
        self._docpairs_fp = Path(docpairs_fp)
        self._docpairs = None

        if self._docpairs_fp.exists():
            self._load_docpairs()

    def queries(self) -> pd.DataFrame:
        return get_topics("train")

    def docpairs_iter(self) -> Iterator[GenericDocPair]:
        if not self._docpairs_fp.exists():
            self._save_docpairs()
        
        yield from self._load_docpairs()

    def _load_docpairs(self) -> list[GenericDocPair]:
        if self._docpairs is not None:
            return self._docpairs

        with open(self._docpairs_fp, "rt") as f:
            data = json.load(f)

        self._docpairs = [GenericDocPair(**item) for item in data]
        return self._docpairs

    def _save_docpairs(self) -> None:
        docpairs = [docpair._asdict() for docpair in self._prepare_docpairs()]
        _random = random.Random(42)
        _random.shuffle(docpairs)

        with open(self._docpairs_fp, "wt") as f:
            json.dump(docpairs, f)

    def _prepare_docpairs(self) -> Iterator[GenericDocPair]:
        has_answers = _try_import_pyserini_has_answers()
        bm25 = _init_bm25()
        train_topics = get_topics("train")
        train_answers = get_answers("train")
        qid2answers = defaultdict(list)
        _random = random.Random(42)

        for row in train_answers.itertuples():
            qid2answers[row.qid].append(str(row.gold_answer))

        for row in tqdm(
            train_topics.itertuples(),
            total=len(train_topics),
            desc="preparing docpairs",
        ):
            pos_items = []
            neg_items = []
            for item in bm25.search(row.query).itertuples():
                if has_answers(
                    item.text,
                    qid2answers[row.qid],
                    tokenizer=None,
                    regex=True,
                ):
                    pos_items.append(item)
                else:
                    neg_items.append(item)

            if not pos_items or not neg_items:
                continue

            for pos_item in pos_items:
                neg_item = _random.choice(neg_items)
                yield GenericDocPair(
                    query_id=row.qid,
                    doc_id_a=pos_item.docno,
                    doc_id_b=neg_item.docno,
                )
