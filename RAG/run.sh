#!/bin/bash

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Example to run Flat, HNSW, JPQ (Zero-Shot), BM25, and Direct Inference
E5_FLAT_INDEX_PATH=data/ragwiki-e5.flex
E5_JPQ_INDEX_PATH=data/ragwiki-jpq-e5
E5_JPQ_ENCODER_PATH=data/e5-jpq-msmarco 

TCT_FLAT_INDEX_PATH=data/ragwiki-tct-colbert.flex
TCT_JPQ_INDEX_PATH=data/ragwiki-jpq-tct
TCT_JPQ_ENCODER_PATH=data/tct-colbert-jpq-msmarco


for LLM in Qwen/Qwen2.5-3B-Instruct meta-llama/Llama-3.2-3B-Instruct; do
    python -m rag.run --llm $LLM \
    --retriever e5 \
    --flat_index_path $E5_FLAT_INDEX_PATH \
    --jpq_index_path $E5_JPQ_INDEX_PATH \
    --jpq_encoder_path $E5_JPQ_ENCODER_PATH \
    --eval_jpq \
    --eval_flat \
    --eval_hnsw

    python -m rag.run --llm $LLM \
    --retriever tct-colbert \
    --flat_index_path $TCT_FLAT_INDEX_PATH \
    --jpq_index_path $TCT_JPQ_INDEX_PATH \
    --jpq_encoder_path $TCT_JPQ_ENCODER_PATH \
    --eval_jpq \
    --eval_flat \
    --eval_hnsw

    python -m rag.run --llm $LLM \
    --retriever bm25 \
    --eval_direct_inference
done
