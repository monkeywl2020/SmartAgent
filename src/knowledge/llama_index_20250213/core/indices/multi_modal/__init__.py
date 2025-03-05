"""Vector-store based data structures."""

from llama_index.core.indices.multi_modal.base import MultiModalVectorStoreIndex
from llama_index.core.indices.multi_modal.wl_multi_vectorstoreindex import wlMultiModalVectorStoreIndex # 新增多模态的index
from llama_index.core.indices.multi_modal.retriever import (
    MultiModalVectorIndexRetriever,
)

__all__ = [
    "MultiModalVectorStoreIndex",
    "wlMultiModalVectorStoreIndex", # 新增多模态的index
    "MultiModalVectorIndexRetriever",
]
