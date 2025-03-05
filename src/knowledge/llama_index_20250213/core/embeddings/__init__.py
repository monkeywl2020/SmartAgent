from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.embeddings.mock_embed_model import MockEmbedding
from llama_index.core.embeddings.multi_modal_base import MultiModalEmbedding
from llama_index.core.embeddings.wl_custom_embeding import wlMultiModalEmbedding # 新增自定义多模态Embedding
from llama_index.core.embeddings.pooling import Pooling
from llama_index.core.embeddings.utils import resolve_embed_model

__all__ = [
    "BaseEmbedding",
    "MockEmbedding",
    "MultiModalEmbedding",
    "wlMultiModalEmbedding",# 新增自定义多模态Embedding
    "Pooling",
    "resolve_embed_model",
]
