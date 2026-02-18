import pyterrier as pt
from transformers import PreTrainedTokenizerBase
from pyterrier_rag.readers import Reader
from pyterrier_rag.prompt import Concatenator, PromptTransformer
from pyterrier_rag.backend import TextGenerator, VLLMBackend, HuggingFaceBackend


# See Also: https://github.com/yuwvandy/KG-LLM-MDQA/blob/main/Pipeline/prompt.py
_PROMPT = """Given the following documents:
{qcontext}

Answer the following question: {query}

Your answer should be concise (no more than 6 words) and no explanation is needed.
Answer:"""


_PROMPT_NO_CONTEXT = """\
Answer the following question: {query}

Your answer should be concise (no more than 6 words) and no explanation is needed.
Answer:"""


class PromptTransformerV2(PromptTransformer):
    def __init__(self, tokenizer: PreTrainedTokenizerBase, raw_prompt: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.raw_prompt = raw_prompt
        self.tokenizer = tokenizer

    def create_prompt(self, fields: dict) -> str:
        prompt = self.raw_prompt.format(**{k: fields[k] for k in self.input_fields})
        if self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt},
                ],
                add_generation_prompt=True,
                tokenize=False,
            )

        return prompt


def format_doc(title: str, text: str) -> str:
    title = title.strip('"')
    text = text.removeprefix(title).lstrip()
    return f"Title: {title}\nText: {text}\n"


def get_reader(
    backend: VLLMBackend | HuggingFaceBackend,
    batch_size: int = 32,
    *,
    add_concatenator: bool = True,
    in_fields: list[str] = ["title", "text"],
    do_rag: bool = True,
) -> pt.Transformer:
    if isinstance(backend, VLLMBackend):
        tokenizer = backend.model.get_tokenizer()
    elif isinstance(backend, HuggingFaceBackend):
        tokenizer = backend.tokenizer
    else:
        raise NotImplementedError(f"{type(backend)=}")

    if do_rag:
        prompt_transformer = PromptTransformerV2(
            tokenizer,
            raw_prompt=_PROMPT,
            model_name_or_path=backend.model_id,
        )
    else:
        prompt_transformer = PromptTransformerV2(
            tokenizer,
            raw_prompt=_PROMPT_NO_CONTEXT,
            model_name_or_path=backend.model_id,
            input_fields=["query"],
        )

    reader = Reader(backend=backend, prompt=prompt_transformer)

    if not isinstance(reader.backend, TextGenerator):
        raise NotImplementedError

    reader.backend.batch_size = batch_size

    if add_concatenator and do_rag:
        concatenator = Concatenator(in_fields=in_fields, intermediate_format=format_doc)
        return concatenator >> reader

    return reader
