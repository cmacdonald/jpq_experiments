import argparse
from pathlib import Path
from dataclasses import dataclass, field

import torch
import pandas as pd
import pyterrier as pt
import pyterrier_rag as ptr
import pyterrier_dr as ptd

import rag.prompt
import rag.nq as nq

_PIPELINES = []
_NAMES = []


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", type=str, choices=["e5", "tct-colbert", "bm25"], default="e5")
    parser.add_argument("--save_dir", type=Path, default=None)
    parser.add_argument("--eval_direct_inference", action="store_true")
    parser.add_argument("--reuse_results", action="store_true")
    parser.add_argument("--topk", type=int, nargs="+", default=[5, 10])
    parser.add_argument("--llm", type=str, default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--generation_batch_size", type=int, default=32)
    parser.add_argument("--encoder_batch_size", type=int, default=256)
    parser.add_argument("--vllm_max_model_len", type=int, default=16384)
    parser.add_argument("--vllm_max_num_batched_tokens", type=int, default=65536)
    parser.add_argument("--hnsw_neighbours", type=int, default=128)
    parser.add_argument("--hnsw_ef_construction", type=int, default=100)
    parser.add_argument("--hnsw_ef_search", type=int, default=32)
    parser.add_argument("--eval_jpq", action="store_true")
    parser.add_argument("--eval_flat", action="store_true")
    parser.add_argument("--eval_hnsw", action="store_true")
    parser.add_argument("--jpq_index_path", type=str, default=None)
    parser.add_argument("--jpq_encoder_path", type=str, default=None)
    parser.add_argument("--flat_index_path", type=str, default=None)
    return parser.parse_args()


@dataclass
class Config:
    retriever: str = "e5"
    save_dir: Path = Path("results")
    topk: list[int] = field(default_factory=lambda: [5, 10])
    eval_direct_inference: bool = False
    reuse_results: bool = False
    llm: str = "Qwen/Qwen2.5-3B-Instruct"
    generation_batch_size: int = 32
    encoder_batch_size: int = 256
    vllm_max_model_len: int | None = None
    vllm_max_num_batched_tokens: int = 65536
    hnsw_neighbours: int = 128
    hnsw_ef_construction: int = 100
    hnsw_ef_search: int = 32
    eval_jpq: bool = True
    eval_flat: bool = True
    eval_hnsw: bool = True
    jpq_index_path: str = None
    jpq_encoder_path: str = None
    flat_index_path: str = None


def load_e5(model_path: str | None = None) -> ptd.E5:
    from sentence_transformers import SentenceTransformer

    e5 = ptd.E5()
    if model_path:
        e5.model = SentenceTransformer(model_path, device="cuda")
    return e5


def load_tct_colbert(model_path: str | None = None) -> ptd.TctColBert:
    tct = ptd.TctColBert.hnp(device="cuda")
    if model_path:
        tct.model = tct.model.from_pretrained(model_path).eval().to("cuda")
    return tct


def load_backend(
    model: str,
    vllm=True,
    vllm_gpu_memory_utilization: float = 0.8,
    vllm_max_model_len: int | None = None,
    vllm_max_num_batched_tokens: int = 65536,
    generation_batch_size: int = 256,
) -> ptr.Backend:
    if vllm:
        return ptr.VLLMBackend(
            model,
            model_args={
                "gpu_memory_utilization": vllm_gpu_memory_utilization,
                "max_model_len": vllm_max_model_len,
                "max_num_seqs": generation_batch_size,
                "max_num_batched_tokens": vllm_max_num_batched_tokens,
            },
            generation_args={
                "max_tokens": 32,
                "temperature": 0,
            },
        )

    backend = ptr.HuggingFaceBackend(
        model,
        model_args={
            "torch_dtype": torch.bfloat16,
        },
        generation_args={
            "do_sample": False,
            "max_new_tokens": 32,
            "temperature": None,
            "top_p": None,
            "top_k": None,
        },
    )
    backend.tokenizer.padding_side = "left"
    return backend


def init_bm25_pipeline(
    config: Config,
    text_loader: pt.Transformer,
    reader: pt.Transformer,
) -> None:
    sparse_index = pt.Artifact.from_hf("pyterrier/ragwiki-terrier")

    for k in config.topk:
        bm25 = pt.rewrite.tokenise() >> sparse_index.bm25(num_results=k) >> pt.rewrite.reset()
        pipeline_bm25 = bm25 >> text_loader >> reader
        _PIPELINES.append(pipeline_bm25)
        _NAMES.append(f"BM25-top{k}")


def init_dense_pipelines(
    config: Config,
    text_loader: pt.Transformer,
    reader: pt.Transformer,
) -> None:
    if config.retriever == "e5":
        load_model = load_e5
        prefix = "E5"
    elif config.retriever == "tct-colbert":
        load_model = load_tct_colbert
        prefix = "TCT"
    else:
        raise NotImplementedError(f"unsupported {config.retriever=}")

    if config.eval_jpq:
        try:
            from pyterrier_dr.jpq import JPQIndex
        except ImportError as e:
            raise ImportError("make sure the JPQ extension of pyterrier-dr is installed") from e

        index_jpq = JPQIndex(config.jpq_index_path)
        encoder_jpq = load_model(config.jpq_encoder_path)

    if config.eval_flat or config.eval_hnsw:
        index_flex = ptd.FlexIndex(config.flat_index_path)
        encoder = load_model()

    for k in config.topk:
        if config.eval_jpq:
            pipeline = (
                encoder_jpq.query_encoder(batch_size=config.encoder_batch_size)
                >> index_jpq.retriever_pq(topk=k)
                >> text_loader
                >> reader
            )
            _PIPELINES.append(pipeline)
            _NAMES.append(f"{prefix}-JPQ-top{k}")

        if config.eval_hnsw:
            pipeline = (
                encoder.query_encoder(batch_size=config.encoder_batch_size)
                >> index_flex.faiss_hnsw_retriever(
                    num_results=k,
                    neighbours=config.hnsw_neighbours,
                    ef_construction=config.hnsw_ef_construction,
                    ef_search=config.hnsw_ef_search,
                )
                >> text_loader
                >> reader
            )
            _PIPELINES.append(pipeline)
            _NAMES.append(f"{prefix}-HNSW-top{k}")

        if config.eval_flat:
            pipeline = (
                encoder.query_encoder(batch_size=config.encoder_batch_size)
                >> index_flex.np_retriever(num_results=k)
                >> text_loader
                >> reader
            )
            _PIPELINES.append(pipeline)
            _NAMES.append(f"{prefix}-Flat-top{k}")


def init_direct_inference_pipeline(
    backend: ptr.Backend,
    batch_size: int = 64,
) -> None:
    reader = rag.prompt.get_reader(
        backend=backend,
        batch_size=batch_size,
        do_rag=False,
    )

    _PIPELINES.append(reader)
    _NAMES.append("NoRAG")


def read_csv(fp: str, **kwargs) -> pd.DataFrame:
    df = pd.read_csv(fp, index_col=0)
    df.qanswer = df.qanswer.astype("string").fillna("")
    return df


def main(config: Config) -> None:
    text_loader = pt.Artifact.from_hf("pyterrier/ragwiki-terrier").text_loader(["docno", "title", "text"])
    backend = load_backend(
        config.llm,
        vllm_max_model_len=config.vllm_max_model_len,
        generation_batch_size=config.generation_batch_size,
        vllm_max_num_batched_tokens=config.vllm_max_num_batched_tokens,
    )
    reader = rag.prompt.get_reader(
        backend=backend,
        batch_size=config.generation_batch_size,
    )

    if config.retriever == "bm25":
        init_bm25_pipeline(config, text_loader, reader)
    else:
        init_dense_pipelines(config, text_loader, reader)

    if config.eval_direct_inference:
        init_direct_inference_pipeline(backend=backend, batch_size=config.generation_batch_size)

    save_dir: Path = config.save_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    with open(save_dir / "args", "wt") as f:
        f.write(str(config.__dict__))

    results: pd.DataFrame = pt.Experiment(
        _PIPELINES,
        nq.get_topics("test"),
        nq.get_answers("test"),
        [ptr.measures.F1, ptr.measures.EM, "mrt"],
        verbose=True,
        names=_NAMES,
        save_dir=save_dir,
        save_mode="reuse" if config.reuse_results else "overwrite",
        save_format=(read_csv, pd.DataFrame.to_csv),
    )

    print(results)
    results.to_csv(save_dir / "results.csv")


if __name__ == "__main__":
    args = parse_args()
    print(args, flush=True)

    if args.eval_jpq and args.jpq_index_path is None:
        raise ValueError(f"require `jpq_index_path`")

    if (args.eval_flat or args.eval_hnsw) and args.flat_index_path is None:
        raise ValueError("require `flat_index_path`")

    if not isinstance(args.topk, list):
        args.topk = [args.topk]

    if args.save_dir is None:
        args.save_dir = Path("results") / args.llm.split("/")[-1]

    main(Config(**args.__dict__))
