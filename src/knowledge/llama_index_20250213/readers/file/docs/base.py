"""Docs parser.

Contains parsers for docx, pdf files.

"""

import io
import logging
import struct
import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import traceback
import re
import os
from tenacity import retry, stop_after_attempt

from fsspec import AbstractFileSystem

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.readers.base import BaseReader
from llama_index.core.readers.file.base import get_default_fs, is_default_fs
from llama_index.core.schema import Document

from llama_index.readers.file.docs.pdf_2_markdown import PdfToMarkdown
from llama_index.readers.file.docs.imgFromPdf import ExtractGraphFromPdf

logger = logging.getLogger(__name__)

RETRY_TIMES = 3

# 自定义的 pdf reader，根据用户的问题查找pdf资料，并且给出页面的图片信息。
class wlPDFReader(BaseReader):
    """PDF parser."""

    def __init__(self, return_full_document: Optional[bool] = False) -> None:
        """
        Initialize PDFReader.  默认返回多个 Document 列表
        """
        self.return_full_document = return_full_document
        #print(f"wlPDFReader::__init__() beging")

    # 下面的函数，使用 PdfToMarkdown 工具，将pdf解析成Markdown格式，
    # 将pdf的每一页作为一个Document存储，不支持将整个pdf变成一个Document，因为页信息会丢失，后面没有办法跟截图对应起来
    @retry(
        stop=stop_after_attempt(RETRY_TIMES),
    )
    def load_data(
        self,
        file: Path,
        outPut_file: str = ".",
        save_png: bool = False,
        page_overlap: int = 200,  # 新增参数：页重叠大小 默认200 字符
        excluded_content: list[tuple[str, str]] = None,
        extra_info: Optional[Dict] = None,
    ) -> List[Document]:
        """Parse file."""
        #print(f"wlPDFReader::load_data() file[{str(file)}] outPut_file[{outPut_file}] save_png[{save_png}]",flush=True)
        logger.info(f"wlPDFReader::load_data() file[{str(file)}] outPut_file[{outPut_file}] save_png[{save_png}]")
        try:
            if not isinstance(file, Path):
                file = Path(file)
            
            #----------------------------------------------------
            #  新增处理， wlPdfReader 输入的文件名，需要将图片保存到这个文件名命名的文件夹中去
            #----------------------------------------------------
            filename_without_extension = file.stem  # 获取文件名，不带扩展名
            new_output_file = Path(outPut_file) / filename_without_extension  # 输出的文本文件路径
            #print(f"wlPDFReader::load_data()===2222 file[{str(file)}] output_file[{outPut_file}] png_output_file[{new_output_file}]  save_png[{save_png}]",flush=True)

            # 初始化 PdfToMarkdown 
            pdf_to_md = PdfToMarkdown(pdf_path = str(file), out_dir = outPut_file)

            #------------------------------------
            if save_png is True:
                #将pdf的每一页截图保存起来,png的保存需要使用 文件名来创建文件夹，否则多个文件都会放到一起被覆盖掉
                pdf_to_png = ExtractGraphFromPdf(pdf_path = str(file), output_folder = new_output_file)
                pdf_to_png.save_pdf_to_png()
            #------------------------------------

            # 解析pdf，解析后。内容会存放在 out_dir 这个路径下
            pdf_to_md.process_pdf(save_to_txt=True)

            #获取解析的结果
            doc = pdf_to_md.get_Markdown_content()

            # 返回的文档内容
            docs = []
            # 下面代码是因为pdf解析出来每一页都当做不同的doc处理的。否则所有的pdf处理成1个doc，那么pdf中的页面内容就会丢失

            # 用于存储上一页的重叠内容
            prev_overlap_content = ""
            
            # 初始化 SentenceSplitter 用于句子分割
            sentence_splitter = SentenceSplitter(chunk_size=page_overlap, chunk_overlap=0)

            #遍历文档每一页
            for page in doc:
                txt_in_page = ""
                # metadata 中存放当前 页面的编号和存成图片的文件名,还有文件的路径
                page_num = page["page_num"] # 页编号，就是pdf从第一页开始的计数，这个不是实际pdf中的页面
                file_png_name = "page_" + str(page_num) + ".png" # 这个是当前页面截图保存的png文件名
                file_path_str = str(new_output_file) + "/images/" + file_png_name #png输出文件的文件路径
                metadata = {"page_img_Num": page_num, "page_img_name": file_png_name,"page_img_path":file_path_str}# 保存到metadata数据中去
                #print("-----------------metadata:\n{metadata}")
                if extra_info is not None:
                    metadata.update(extra_info)# 将传入的额外的信息添加到 metadata去

                #遍历每一页中的 内容
                for item in page['content']:
                    if 'text' in item:
                        txt_in_page = txt_in_page + item['text'] + '\n'
                    elif 'table' in item:
                        txt_in_page = txt_in_page + '[TABLE]\n' + item['table'] + '\n'

                # --------------------------------
                # 添加上一页的重叠部分到当前页的开头
                # --------------------------------
                if prev_overlap_content:
                    txt_in_page = prev_overlap_content + "\n" + txt_in_page

                # 使用 SentenceSplitter 对当前页末尾内容进行切分，确保重叠部分是完整句子
                sentences = sentence_splitter.split_text(txt_in_page)
                
                # 提取重叠部分
                overlap_sentences = []
                char_count = 0
                # 反向遍历 所有的句子。
                for sentence in reversed(sentences): 
                    char_count += len(sentence)# 计算token
                    overlap_sentences.insert(0, sentence)  # 从末尾往前插入
                    if char_count >= page_overlap:
                        break

                # 更新上一页的重叠内容
                prev_overlap_content = "\n".join(overlap_sentences)

                # --------------------------------
                # 处理 excluded_content
                # --------------------------------
                if excluded_content:
                    for pattern_type, pattern_value in excluded_content:
                        if pattern_type == 'str':
                            # 如果是普通字符串，直接替换为空
                            txt_in_page = txt_in_page.replace(pattern_value, '')
                        elif pattern_type == 'regex':
                            # 如果是正则表达式，使用 re.sub 替换为空
                            txt_in_page = re.sub(pattern_value, '', txt_in_page)

                # 添加每一页的内容 ，每一页作为   Document 存储，存储文本内容，同时存储此页面的pdf截图  
                docs.append(Document(text=txt_in_page, metadata=metadata))
            
            logger.info(f"wlPDFReader::load_data() finished! docs[{docs}]")
            return docs
        except Exception as e:
            print(f"异常类型: {type(e)}")
            print(f"异常名称: {e.__class__.__name__}")
            print(f"异常信息: {e}")
            traceback.print_exc()  # 打印堆栈信息
            raise  # 重新抛出异常，触发重试机制 

class PDFReader(BaseReader):
    """PDF parser."""

    def __init__(self, return_full_document: Optional[bool] = False) -> None:
        """
        Initialize PDFReader.
        """
        self.return_full_document = return_full_document

    @retry(
        stop=stop_after_attempt(RETRY_TIMES),
    )
    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        #print(f"PDFReader::load_data() file[{str(file)}]")
        """Parse file."""
        if not isinstance(file, Path):
            file = Path(file)

        try:
            import pypdf
        except ImportError:
            raise ImportError(
                "pypdf is required to read PDF files: `pip install pypdf`"
            )
        fs = fs or get_default_fs()
        with fs.open(str(file), "rb") as fp:
            # Load the file in memory if the filesystem is not the default one to avoid
            # issues with pypdf
            stream = fp if is_default_fs(fs) else io.BytesIO(fp.read())

            # Create a PDF object
            pdf = pypdf.PdfReader(stream)

            # Get the number of pages in the PDF document
            num_pages = len(pdf.pages)

            docs = []

            # This block returns a whole PDF as a single Document
            if self.return_full_document:
                metadata = {"file_name": file.name}
                if extra_info is not None:
                    metadata.update(extra_info)

                # Join text extracted from each page
                text = "\n".join(
                    pdf.pages[page].extract_text() for page in range(num_pages)
                )

                docs.append(Document(text=text, metadata=metadata))

            # This block returns each page of a PDF as its own Document
            else:
                # Iterate over every page

                for page in range(num_pages):
                    # Extract the text from the page
                    page_text = pdf.pages[page].extract_text()
                    page_label = pdf.page_labels[page]

                    metadata = {"page_label": page_label, "file_name": file.name}
                    if extra_info is not None:
                        metadata.update(extra_info)

                    docs.append(Document(text=page_text, metadata=metadata))

            return docs


class DocxReader(BaseReader):
    """Docx parser."""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Parse file."""
        if not isinstance(file, Path):
            file = Path(file)

        try:
            import docx2txt
        except ImportError:
            raise ImportError(
                "docx2txt is required to read Microsoft Word files: "
                "`pip install docx2txt`"
            )

        if fs:
            with fs.open(str(file)) as f:
                text = docx2txt.process(f)
        else:
            text = docx2txt.process(file)
        metadata = {"file_name": file.name}
        if extra_info is not None:
            metadata.update(extra_info)

        return [Document(text=text, metadata=metadata or {})]


class HWPReader(BaseReader):
    """Hwp Parser."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.FILE_HEADER_SECTION = "FileHeader"
        self.HWP_SUMMARY_SECTION = "\x05HwpSummaryInformation"
        self.SECTION_NAME_LENGTH = len("Section")
        self.BODYTEXT_SECTION = "BodyText"
        self.HWP_TEXT_TAGS = [67]
        self.text = ""

    def load_data(
        self,
        file: Path,
        extra_info: Optional[Dict] = None,
        fs: Optional[AbstractFileSystem] = None,
    ) -> List[Document]:
        """Load data and extract table from Hwp file.

        Args:
            file (Path): Path for the Hwp file.

        Returns:
            List[Document]
        """
        import olefile

        if fs:
            logger.warning(
                "fs was specified but HWPReader doesn't support loading "
                "from fsspec filesystems. Will load from local filesystem instead."
            )

        if not isinstance(file, Path):
            file = Path(file)
        load_file = olefile.OleFileIO(file)
        file_dir = load_file.listdir()
        if self.is_valid(file_dir) is False:
            raise Exception("Not Valid HwpFile")

        result_text = self._get_text(load_file, file_dir)
        result = self._text_to_document(text=result_text, extra_info=extra_info)
        return [result]

    def is_valid(self, dirs: List[str]) -> bool:
        if [self.FILE_HEADER_SECTION] not in dirs:
            return False

        return [self.HWP_SUMMARY_SECTION] in dirs

    def get_body_sections(self, dirs: List[str]) -> List[str]:
        m = []
        for d in dirs:
            if d[0] == self.BODYTEXT_SECTION:
                m.append(int(d[1][self.SECTION_NAME_LENGTH :]))

        return ["BodyText/Section" + str(x) for x in sorted(m)]

    def _text_to_document(
        self, text: str, extra_info: Optional[Dict] = None
    ) -> Document:
        return Document(text=text, extra_info=extra_info or {})

    def get_text(self) -> str:
        return self.text

        # 전체 text 추출

    def _get_text(self, load_file: Any, file_dirs: List[str]) -> str:
        sections = self.get_body_sections(file_dirs)
        text = ""
        for section in sections:
            text += self.get_text_from_section(load_file, section)
            text += "\n"

        self.text = text
        return self.text

    def is_compressed(self, load_file: Any) -> bool:
        header = load_file.openstream("FileHeader")
        header_data = header.read()
        return (header_data[36] & 1) == 1

    def get_text_from_section(self, load_file: Any, section: str) -> str:
        bodytext = load_file.openstream(section)
        data = bodytext.read()

        unpacked_data = (
            zlib.decompress(data, -15) if self.is_compressed(load_file) else data
        )
        size = len(unpacked_data)

        i = 0

        text = ""
        while i < size:
            header = struct.unpack_from("<I", unpacked_data, i)[0]
            rec_type = header & 0x3FF
            (header >> 10) & 0x3FF
            rec_len = (header >> 20) & 0xFFF

            if rec_type in self.HWP_TEXT_TAGS:
                rec_data = unpacked_data[i + 4 : i + 4 + rec_len]
                text += rec_data.decode("utf-16")
                text += "\n"

            i += 4 + rec_len

        return text
