from collections import Counter
import re
import os
import json
import logging
import argparse
from tqdm import tqdm  # 导入tqdm库
from typing import Optional
from pdfminer.high_level import extract_pages,extract_text
from pdfminer.layout import LTImage, LTRect, LTTextContainer
# 导入pdfminer库，用于解析PDF内容
import pdfplumber
from pathlib import Path

#from imgFromPdf import ExtractGraphFromPdf
from llama_index.readers.file.docs.imgFromPdf import ExtractGraphFromPdf

# 导入日志模和 从pdf中提取图模块
#from log import logger
logger = logging.getLogger(__name__)


#=======================================================================================
#  PdfToMarkdown 这个类的作用： 将 pdf 转成 Markdown格式的 txt文档
#
#=======================================================================================
class PdfToMarkdown:
    # pdf_path 是pdf文件的路径（相对路径绝对路径都可以，系统默认支持）
    def __init__(self, pdf_path: str, out_dir: str):
        self.pdf_path = pdf_path
        self.doc = []  # 用于存储最终解析的文档内容

        # 创建输出目录，如果没有的话，支持相对路径和绝对路径
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)  # 创建目录
            logger.info(f"Output directory created: {out_dir}")
        else:
            logger.info(f"Output directory already exists: {out_dir}")
        
        pdf_path = Path(pdf_path) # 转成  Path 类
        if pdf_path.is_file(): # 如果是带路径的 文件 名
            filename_without_extension = pdf_path.stem  # 获取文件名，不带扩展名
            self.output_file = os.path.join(out_dir, f'{filename_without_extension}.txt')  # 输出的文本文件路径
        else:
            # 如果不是文件是路径
            self.output_file = os.path.join(out_dir, 'parsed_document.txt')  # 输出的文本文件路径
        logger.info(f"PdfToMarkdown::__init__: Output filepath {self.output_file}")

    # 删除 PDF 中的 (cid:xxx) 占位符
    def rm_cid(self, text: str) -> str:
        text = re.sub(r'\(cid:\d+\)', '', text) # 删除 PDF 中的 (cid:xxx) 占位符
        return text

    #删除长度大于21位的十六进制字符串
    def rm_hexadecimal(self, text: str) -> str:
        text = re.sub(r'[0-9A-Fa-f]{21,}', '', text) # 删除长度大于21位的十六进制字符串
        return text

    # 删除连续的无意义符号，并替换为制表符
    def rm_continuous_placeholders(self, text: str) -> str:
        text = re.sub(r'[.\- —。_*]{7,}', '\t', text) # 删除连续的无意义符号，并替换为制表符
        text = re.sub(r'\n{3,}', '\n\n', text)  # 将超过3个换行符替换为2个换行符
        return text

    # 将多个清洗规则依次应用于文本
    def clean_paragraph(self, text: str) -> str:
        text = self.rm_cid(text) # 删除 PDF 中的 (cid:xxx) 占位符
        text = self.rm_hexadecimal(text) # 删除长度大于21位的十六进制字符串
        text = self.rm_continuous_placeholders(text) # 删除连续的无意义符号，并替换为制表符
        return text
        
    # 去除重复的表格和文本元素并合并误分段的文本行
    # - 移除重复的表格和文本
    # - 合并误分段的文本行，确保同一段落不会被拆分到多个对象中。
    # - 清洗最终文本内容。
    def postprocess_page_content(self, page_content: list) -> list:
        # rm repetitive identification for table and text
        # Some documents may repeatedly recognize LTRect and LTTextContainer
        table_obj = [p['obj'] for p in page_content if 'table' in p]
        tmp = []
        for p in page_content:
            repetitive = False
            if 'text' in p:
                # 检查当前文本是否在表格中重复出现
                for t in table_obj:
                    if t.bbox[0] <= p['obj'].bbox[0] and p['obj'].bbox[1] <= t.bbox[1] and t.bbox[2
                            ] <= p['obj'].bbox[2] and p['obj'].bbox[3] <= t.bbox[3]:
                        repetitive = True
                        break

            if not repetitive:
                tmp.append(p)  # 添加非重复项
        page_content = tmp

        # merge paragraphs that have been separated by mistake # 合并误分段的文本
        new_page_content = []
        for p in page_content:
            if new_page_content and 'text' in new_page_content[-1] and 'text' in p and abs(
                    p.get('font-size', 12) - 
                    new_page_content[-1].get('font-size', 12)) < 2 and p['obj'].height < p.get('font-size', 12) + 1:
                # Merge those lines belonging to a paragraph
                new_page_content[-1]['text'] += f' {p["text"]}'  # 合并属于同一段落的文本
                # new_page_content[-1]['font-name'] = p.get('font-name', '')
                new_page_content[-1]['font-size'] = p.get('font-size', 12)
            else:
                p.pop('obj')  # 删除对象引用，减少冗余信息
                new_page_content.append(p)

        # 应用文本清洗
        for i in range(len(new_page_content)):
            if 'text' in new_page_content[i]:
                new_page_content[i]['text'] = self.clean_paragraph(new_page_content[i]['text'])

        return new_page_content

    #分析 PDF 文本字体和大小信息
    def get_font(self, element):
        from pdfminer.layout import LTChar, LTTextContainer

        fonts_list = []
        for text_line in element:
            if isinstance(text_line, LTTextContainer):
                for character in text_line:
                    if isinstance(character, LTChar):
                        fonts_list.append((character.fontname, character.size))  # 获取字体及大小

        fonts_list = list(set(fonts_list))
        if fonts_list:
            counter = Counter(fonts_list)  # 统计频率
            most_common_fonts = counter.most_common(1)[0][0]  # 返回最常见的字体
            return most_common_fonts
        else:
            return []
        
    #提取表格内容
    def extract_tables(self, pdf, page_num):
        table_page = pdf.pages[page_num]
        tables = table_page.extract_tables()  # 提取表格内容
        return tables

    #将表格内容转换为 Markdown 风格的表格字符串格式
    def table_converter(self, table):
        table_string = ''
        for row_num in range(len(table)):
            row = table[row_num]
            cleaned_row = [
                item.replace('\n', ' ') if item is not None and '\n' in item else 'None' if item is None else item
                for item in row
            ]  # 清理表格内容
            table_string += ('|' + '|'.join(cleaned_row) + '|' + '\n')  # 转换为字符串格式
        table_string = table_string[:-1]
        return table_string

    #根据图描述正则表达式进行匹配
    def is_image_description(self, text: str) -> bool:
        pattern = r'(图[\dA-Za-z]{1,2}-\d{1,2}[^\s]*)'
        return re.search(pattern, text)

    def process_page_content_with_image_markdown(self, page_content: list) -> list:
        """处理页面内容，识别图描述并插入 Markdown 图片链接"""
        processed_content = []
        for item in page_content:
            if 'text' in item:
                text = item['text']
                # 如果文本包含有效的图描述，则替换为 Markdown 格式
                if self.is_image_description(text):
                    image_filename = self.generate_image_filename(text)
                    if image_filename:
                        # 创建 Markdown 格式的图片链接
                        markdown_image = f"![图片描述]({image_filename})"
                        processed_content.append({'text': markdown_image})
                    else:
                        item['text'] = self.clean_paragraph(text)
                        processed_content.append(item)
                else:
                    item['text'] = self.clean_paragraph(text)
                    processed_content.append(item)
        return processed_content

    def save_doc_to_file(self):
        """保存文档内容到文本文件"""
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for page in self.doc:
                f.write(f'Page {page["page_num"]}\n')
                f.write("=" * 20 + "\n")
                for item in page['content']:
                    if 'text' in item:
                        f.write(item['text'] + '\n')
                    elif 'table' in item:
                        f.write('[TABLE]\n' + item['table'] + '\n')
                    elif 'image' in item:
                        f.write(f'[IMAGE] {item["image"]}\n')
                f.write("\n\n")

    #解析 PDF 文件并提取内容
    def process_pdf(self, save_to_txt: Optional[bool] = False):
        logger.info(f"PdfToMarkdown::process_pdf: pdf_path[{self.pdf_path}] save_to_txt[{save_to_txt}]")
        # 使用 'with' 语句打开 PDF 文件，自动管理文件关闭
        with pdfplumber.open(self.pdf_path) as pdf:
            logger.info(f"PdfToMarkdown::process_pdf: pdf[{self.pdf_path}] opened by pdfplumber!")

            # 获取页面列表
            pages = list(extract_pages(self.pdf_path))  # 提前将页面提取出来

            # 遍历PDF的每一页
            # 遍历所有页面
            for i, page_layout in tqdm(enumerate(pages), total=len(pages), desc="Extract Markdown from pages", unit="page"):
            #for i, page_layout in enumerate(extract_pages(self.pdf_path)):
                logger.info(f"PdfToMarkdown::process_pdf: extract_pages [{i}]!")
                page = {'page_num': page_layout.pageid, 'content': []}
                logger.info(f"PdfToMarkdown::process_pdf: extract_pages--------- :\n{page}")

                elements = []  # 存储每一页的元素（文本框、矩形框、表格等）
                for element in page_layout:
                    elements.append(element)

                # Init params for table
                table_num = 0
                tables = []

                # 遍历每一页的元素，进行分类处理
                for element in elements:
                    # 如果元素是矩形（可能是表格的边界）
                    if isinstance(element, LTRect):
                        logger.info(f"PdfToMarkdown::process_pdf: elements LTRect !")
                        if not tables:
                            logger.info(f"PdfToMarkdown::process_pdf: elements LTRect: extract_tables !")
                            tables = self.extract_tables(pdf, i) # 如果还没有提取表格，调用函数提取该页的表格
                            logger.info(f"PdfToMarkdown::process_pdf: elements LTRect: tables[{tables}]!")
                        if table_num < len(tables): # 如果表格数量小于当前提取的表格数
                            table_string = self.table_converter(tables[table_num]) # 转换表格为字符串
                            table_num += 1
                            if table_string: # 如果转换成功
                                page['content'].append({'table': table_string, 'obj': element})
                                logger.info(f"PdfToMarkdown::process_pdf: elements LTRect: [table: {table_string} obj:{element}!]")

                    # 如果元素是文本容器
                    elif isinstance(element, LTTextContainer):
                        logger.info(f"PdfToMarkdown::process_pdf: elements LTTextContainer !")

                        text = element.get_text() # 获取该元素的文本内容
                        text_pos = element.bbox # 获取文本的位置信息
                        logger.info(f"=======>: elements[{text}],pos[{text_pos}]!")
                        # Todo: Further analysis using font
                        font = self.get_font(element) # 获取该元素的字体信息

                        if text.strip(): # 如果文本非空
                            new_content_item = {'text': text, 'obj': element} # 创建字典存储文本及其相关对象
                            if font:
                                new_content_item['font-size'] = round(font[1])# 获取字体大小并保存
                                # new_content_item['font-name'] = font[0]
                            page['content'].append(new_content_item)# 将文本内容添加到页面的内容中

                    # 如果需要提取图片
                    elif isinstance(element, LTImage):
                        logger.info(f"PdfToMarkdown::process_pdf: elements LTImage !")
                    else:
                        pass # 其他情况忽略

                # 去除重复的元素并清洗文本
                page['content'] = self.postprocess_page_content(page['content'])

                self.doc.append(page)

            # 保存文档内容到文件,如果设置了保存文件的参数那么就保存文件。
            if save_to_txt is True:
                self.save_doc_to_file()
            logger.info(f"Parsing PDF completed!")
        
        logger.info("PdfToMarkdown::process_pdf: PDF document closed.")

    # 返回 doc内容， 这个内容是一个一个的page组成的list
    def get_Markdown_content(self) -> list:
        return self.doc

    def process_pdf_with_markitdown(self):
        # 直接将pdf提取成 text内容
        text_content = extract_text(self.pdf_path)
        with open(self.output_file, 'w', encoding='utf-8') as f:
            f.write(text_content)
            f.write("\n\n")
        
        logger.info(f"Parsing PDF use extract_text, completed!")


def main():
    '''
    参数举例：
        pdf_path 是pdf的文件的路径含文件名，可以是相当路径或者绝对路径
        -o 输出目录  这个是指定输出目录的参数，可以是相对路径或者绝对路径，指定后 解析的pdf的marddown文档会存放在此目录，同时会将pdf的截图放到此目录的images文件夹下

        # 使用举例，解析 1.pdf文档，输出到当前目录的文件 1 下面。
        python pdf_2_markdown.py 1.pdf -o 1

        python pdfParser/pdf_2_markdown.py 深度学习入门基于Python的理论与实现.pdf -o deeplearning

        python pdfParser/pdf_2_markdown.py LeeDL_Tutorial_v.1.1.5.pdf -o LeeHY_DL
    '''
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown")
    parser.add_argument("pdf_path", help="Path to the PDF file")
    #parser.add_argument("image_dir", help="Directory to save extracted images")
    # 在 argparse 中，-o 和 --outputdir 是同一个参数的 不同形式。它们分别是 短选项 和 长选项：
    # -o 是 短选项，通常是一个字符（此例中是 o），用于简化命令行输入。
    # --outputdir 是 长选项，用于更加清晰地指定参数的意义。
    #     如果在命令行中你使用 -o 或 --outputdir，都可以通过 args.outputdir 来获取参数值。
    #    你 不能 通过 args.o 来获取，因为 argparse 将 -o 与 --outputdir 都映射到 args.outputdir，而不是 args.o。
    parser.add_argument("-o","--outputdir", default=".", help="Output directory")# -- 表示是长连接符，可选参数

    args = parser.parse_args()

    pdf_to_md = PdfToMarkdown(pdf_path = args.pdf_path, out_dir = args.outputdir)

    #------------------------------------
    # 将pdf的每一页转成图片png
    pdf_to_png = ExtractGraphFromPdf(pdf_path = args.pdf_path, output_folder = args.outputdir)
    pdf_to_png.save_pdf_to_png()
    #------------------------------------
    # 原始方法
    if 1:
        pdf_to_md.process_pdf()

    #直接使用 pdfminer 的extract_text 提取文本，不做处理
    if 0:
        pdf_to_md.process_pdf_with_markitdown()

#######################################################################################
''' 
 __name__ 是一个特殊变量，它表示当前模块的名字。
    - 如果当前模块是被直接运行的（即作为主程序），那么 __name__ 的值是 "__main__"。
    - 如果当前模块是作为一个模块被导入到其他 Python 文件中的，__name__ 的值会是该模块的名字（即模块的文件名）。

if __name__ == "__main__": 这行判断的意思是：
    - 如果当前脚本是直接运行（而不是作为模块导入），则执行 main() 函数。
    - 如果当前脚本是作为模块导入到其他脚本中，则不会执行 main() 函数。
'''
#######################################################################################
if __name__ == "__main__":
    main()
