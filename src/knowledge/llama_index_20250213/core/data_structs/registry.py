"""Index registry."""

from typing import Dict, Type

from llama_index.core.data_structs.data_structs import (
    KG,
    EmptyIndexStruct,
    IndexDict,
    IndexGraph,
    IndexList,
    IndexLPG,
    IndexStruct,
    KeywordTable,
    MultiModelIndexDict,
    wlMultiModelIndexDict,# 新增多模态数据结构
)
from llama_index.core.data_structs.document_summary import IndexDocumentSummary
from llama_index.core.data_structs.struct_type import IndexStructType
from llama_index.core.data_structs.table import PandasStructTable, SQLStructTable

INDEX_STRUCT_TYPE_TO_INDEX_STRUCT_CLASS: Dict[IndexStructType, Type[IndexStruct]] = {
    IndexStructType.TREE: IndexGraph,
    IndexStructType.LIST: IndexList,
    IndexStructType.KEYWORD_TABLE: KeywordTable,
    IndexStructType.VECTOR_STORE: IndexDict,
    IndexStructType.SQL: SQLStructTable,
    IndexStructType.PANDAS: PandasStructTable,
    IndexStructType.KG: KG,
    IndexStructType.SIMPLE_LPG: IndexLPG,
    IndexStructType.EMPTY: EmptyIndexStruct,
    IndexStructType.DOCUMENT_SUMMARY: IndexDocumentSummary,
    IndexStructType.MULTIMODAL_VECTOR_STORE: MultiModelIndexDict,
    IndexStructType.WL_MULTIMODAL_VECTOR_STORE: wlMultiModelIndexDict,# 新增自定义的多模态索引类型 wlMultiModelIndexDict
}
