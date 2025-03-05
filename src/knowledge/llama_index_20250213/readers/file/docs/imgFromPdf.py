import fitz  # PyMuPDF 库，用于解析PDF内容
import os
import json
import re
import logging
import cv2
import argparse
from tqdm import tqdm  # 导入tqdm库
from typing import Dict, List, Tuple

#from log import logger
logger = logging.getLogger(__name__)


# 坐标转换：PDF -> OpenCV 图像坐标
'''
PyMuPDF (PDF) 坐标系单位 和 openCV的屏幕坐标坐标系不一样，单位也不一样。

--------------PyMuPDF (PDF) 坐标系单位，它和pdf的坐标系不一样，和opencv一样-------
单位：点 (pt)

1 pt = 1/72 英寸。
PDF 使用 矢量坐标，与显示设备分辨率无关。

原点在左上角 (0, 0)
坐标系 X 轴向右增长 Y 轴向下增长

在 PyMuPDF 中，bbox（边界框）是一个表示矩形区域的坐标信息，返回的是 两点坐标：

scss
复制代码
(x0, y0, x1, y1)
1. 各坐标的定义
(x0, y0) —— 左上角坐标
(x1, y1) —— 右下角坐标
这些坐标单位是 pt (points)，并遵循 PyMuPDF 的坐标系统：

--------------OpenCV 坐标系单位-------
OpenCV 图像坐标系单位
单位：像素 (px)
1英寸 = 2.54厘米
例如300 DPI意味着在1英寸(2.54厘米)的长度内有300个像素点

像素大小取决于图像分辨率，例如：300 DPI 或 72 DPI。
原点位置：左上角 (0, 0)

坐标系向右为 X 轴递增，向下为 Y 轴递增。
-------------------------------------------
  ----目前截图是用的  300dpi pix = page.get_pixmap(dpi=300)  # 高分辨率截图

PDF 到图像的缩放因子：

DPI = 300：
python
复制代码
scale_x = 300 / 72  # X 缩放比例
scale_y = 300 / 72  # Y 缩放比例

-------
PDF 坐标转换到图像像素坐标：

python
复制代码
image_x = pdf_x * scale_x
image_y = pdf_y * scale_y


-------
图像像素坐标转换回 PDF 坐标：

python
复制代码
pdf_x = image_x / scale_x
pdf_y = image_y / scale_y

'''
#=======================================================================================
#  ExtractGraphFromPdf 这个类的作用： 是从 pdf中提取 图
#  这个图是通过图描述来提取的，也就是图描述必须要有一定的格式。
#  目前这个格式如下：
#          图X-X 或 图XX-XX, 例如：图1-1, 图2-2, 图3-3, 图A-12 等等
#  一般格式  图A-12 图的描述  这种，匹配到了这个内容后。就会使用opencv检测图。
#  图是通过矩形边框来检测的，所以图必须是被框在矩形边框里面。这样就会将这个图截取保存下来。
#  同时也需要保存pdf每一页的截图。目前是300dpi分辨率
#=======================================================================================
class ExtractGraphFromPdf:
    def __init__(self, pdf_path: str, output_folder: str = None):
        self.pdf_path = pdf_path # pdf文件路径
        if output_folder is not None:
            self.output_folder = os.path.join(output_folder, 'images')  # images 子目录
        else:
            self.output_folder = "images" # images 子目录

        #self.output_folder = output_folder # pdf提取出来的图的路径
        self.scale = 300 / 72  # DPI/pt 缩放比例，opencv的图的dpi是300，PyMuPDF的单位是pt，1pt = 1/72英寸，所以dpi转成pt的是要除这个比例
        self.image_count = 0   # 图片计数
        self.image_descriptions: Dict[str, str] = {} # 图的说明集合

        # 确保输出目录存在
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        logger.info(f"ExtractGraphFromPdf::__init__: Start processing PDF files: {self.pdf_path}")

    #------------------------------------------
    # 坐标转换：PDF -> OpenCV 图像坐标
    #------------------------------------------
    def pdf_to_image_coords(self, pdf_x: float, pdf_y: float) -> Tuple[int, int]:
        """将 PDF 坐标转换为图像坐标，pdf的坐标是pt，需要乘上缩放比率才会转成dpi """
        image_x = pdf_x * self.scale
        image_y = pdf_y * self.scale 
        return int(image_x), int(image_y)

    #------------------------------------------
    # 坐标转换：OpenCV 图像 -> PDF 坐标
    #------------------------------------------
    def image_to_pdf_coords(self, image_x: int, image_y: int) -> Tuple[float, float]:
        """将图像坐标转换为 PDF 坐标，dpi除以缩放比例，才会转成pt"""
        pdf_x = image_x / self.scale
        pdf_y = image_y / self.scale 
        return pdf_x, pdf_y

    #------------------------------------------
    # 提取页面中的图片描述
    #------------------------------------------
    def extract_page_captions(self, page: fitz.Page) -> List[Dict[str, str]]:
        """从 PDF 页面中提取图片描述"""
        captions = []
        text_blocks = page.get_text("dict")['blocks']

        for block in text_blocks:
            if 'lines' in block:
                for line in block['lines']:
                    # 合并同一行的文本 spans
                    full_text = "".join([span['text'] for span in line['spans']])
                    bbox = block['bbox']  # 使用整个 block 的边界框

                    ''' r'(图[\dA-Za-z]{1,2}-\d{1,2}[^\s]*)' 这个正则表达式：
                        1： 图 - 匹配中文字符"图"
                        2： [\dA-Za-z]{1,2} - 匹配1到2个字符，这些字符可以是：
                            - \d 任意数字
                            - A-Z 任意大写字母
                            - a-z 任意小写字母
                        3：- - 匹配一个连字符
                        4：\d{1,2} - 匹配1到2个数字
                        6：[^\s]* - 匹配0个或多个非空白字符
                            - ^ 在方括号内表示"非"
                            - \s 表示任何空白字符（空格、制表符、换行符等）
                            - * 表示匹配0次或多次
                    '''

                    # 匹配描述格式：图X-X 或 图XX-XX
                    if re.match(r'(图[\dA-Za-z]{1,2}-\d{1,2}[^\s]*)', full_text):
                        captions.append({'text': full_text, 'bbox': bbox})
                        logger.info(f"ExtractGraphFromPdf::extract_page_captions: Match graph Description: {full_text}")
        return captions

    # 本函数将pdf文件的每一页保存为png图片    
    def save_pdf_to_png(self, pdf_path:str= None, dpi: int = 300) -> None:
        if pdf_path is None:
            pdf_path = self.pdf_path
            logger.info(f"ExtractGraphFromPdf::save_pdf_to_png:use self pdf_path[{pdf_path}]")
            
        # 使用 'with' 语句打开 PDF 文件，自动管理文件关闭
        with fitz.open(pdf_path) as doc:
            logger.info(f"ExtractGraphFromPdf::save_pdf_to_png: Total number of pages in the document: {len(doc)}")
        
            # 遍历所有页面
            for page_num in tqdm(range(len(doc)), desc="Processing pages", unit="page"):
            #for page_num in range(len(doc)):
                page = doc.load_page(page_num)  # 加载当前页面

                # 计算缩放矩阵
                zoom = dpi/72  # 将DPI转换为缩放比例
                matrix = fitz.Matrix(zoom, zoom)
                
                # 使用matrix参数创建pixmap
                pix = page.get_pixmap(matrix=matrix)
                #pix = page.get_pixmap(dpi)  # 高分辨率截图
                page_image_path = os.path.join(self.output_folder, f"page_{page_num + 1}.png")
                pix.save(page_image_path)

            logger.info(f"ExtractGraphFromPdf::save_pdf_to_png： 处理完成，共截图保存 {len(doc)} 张图片，描述存储在目录: {self.output_folder}")   

        # 文件会在 'with' 语句结束时自动关闭
        logger.info("ExtractGraphFromPdf::save_pdf_to_png: PDF document closed.")

    # 截图pdf整个页面，用于分析图，返回图片的路径
    def save_page_to_png(self, page: fitz.Page, page_num: int, dpi: int = 300) -> str:
        # 计算缩放矩阵
        zoom = dpi/72  # 将DPI转换为缩放比例
        matrix = fitz.Matrix(zoom, zoom)
        
        # 使用matrix参数创建pixmap
        pix = page.get_pixmap(matrix=matrix)
        #pix = page.get_pixmap(dpi)  # 高分辨率截图
        page_image_path = os.path.join(self.output_folder, f"page_{page_num + 1}.png")
        pix.save(page_image_path)
        return page_image_path

    #------------------------------------------
    # 使用 OpenCV 检测边框并截取框图
    #------------------------------------------
    def extract_rect_with_caption(self, image: cv2.Mat, captions: List[Dict[str, str]]) -> Dict[str, str]:
        """使用 OpenCV 检测并提取带有标题的矩形区域"""
        logger.info(f"ExtractGraphFromPdf::extract_rect_with_caption: captions:{captions}")

        #将彩色图像转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用高斯模糊去噪声
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 边缘检测
        edges = cv2.Canny(blurred, 50, 150)  

        # 查找轮廓
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_descriptions = {}  # 保存描述

        # 筛选矩形框
        for contour in contours:
            # 计算轮廓的边界框
            x, y, w, h = cv2.boundingRect(contour)

            # 筛选条件：宽高大于100像素且接近矩形
            if w > 100 and h > 100:
                # 计算轮廓的近似多边形
                approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

                # 判断是否为矩形框（顶点数为4）
                if len(approx) == 4:  # 判断是否为矩形

                    logger.info(f"ExtractGraphFromPdf::extract_rect_with_caption: find a rectangle. maybe a graph.")
                    logger.info(f"ExtractGraphFromPdf::extract_rect_with_caption:(opencv coordinates):({x}, {y}, {x+w},{y+h})")

                    # 扩展裁剪区域（增加边距）
                    padding = 50
                    x_start = max(0, x - padding)
                    y_start = max(0, y - padding)
                    x_end = min(image.shape[1], x + w + padding)
                    y_end = min(image.shape[0], y + h + padding + 50)

                    # 裁剪矩形区域和描述
                    cropped = image[y_start:y_end, x_start:x_end]
                    
                    # 坐标转换：图像 -> PDF 坐标
                    pdf_x_start, pdf_y_start = self.image_to_pdf_coords(x, y)
                    pdf_x_end, pdf_y_end = self.image_to_pdf_coords(x + w, y + h)
                    logger.info(f"ExtractGraphFromPdf::extract_rect_with_caption: image (pdf coordinates):({pdf_x_start}, {pdf_y_start}, {pdf_x_end},{pdf_y_end})")

                    # 匹配描述
                    caption = "无内容"
                    min_distance = float('inf')
                    logger.info(f"====captions: {captions}")
                    for desc in captions:
                        desc_y0 = desc['bbox'][1]  # PyMuPDF 坐标 ,左上
                        desc_y1 =  desc['bbox'][3] # PyMuPDF 坐标 ,右下
                        desc_center_y = (desc_y0 + desc_y1) / 2  # 取描述的垂直中心点

                        # 计算描述与矩形框的垂直距离（绝对值）这用的是pt单位
                        distance = abs(desc_center_y - pdf_y_end)
                        logger.info(f"矩形下边(PDF单位pt): {pdf_y_end}, 图的描述中心位置(PDF单位pt): {desc_center_y}, 距离: {distance}")
                        logger.info(f"====desc['text']: {desc['text']}")
                        
                        # 更新最近描述
                        max_distance_pt = 300 / (300 / 72)  # 300 px -> pt
                        if pdf_y_end < desc_y0:  # 框必须在描述上方
                            if 0 <= distance <= max_distance_pt and distance < min_distance:
                                logger.info(f"====process  caption!")
                                self.image_count += 1 # 处理的图片数量加 1
                                # 替换特殊空格并更新描述
                                caption = desc['text'].replace('\u3000', ' ')
                                min_distance = distance

                    # 使用描述前缀作为文件名
                    logger.info(f"ExtractGraphFromPdf::extract_rect_with_caption: caption ({caption})")
                    safe_caption = re.sub(r'[\/:*?"<>|\s]', '_', caption.split()[0]) #匹配caption 中的前面一部分内容，作为图片名称
                    image_name = f"{safe_caption}.png"
                    image_path = os.path.join(self.output_folder, image_name)#全路径
                    cv2.imwrite(image_path, cropped) #将图的框的范围扩展后截图保存

                    # 添加描述
                    image_name_no_suffix = safe_caption  # 不要png 后缀
                    image_descriptions[image_name_no_suffix] = caption
                    
                    logger.info(f"ExtractGraphFromPdf::extract_rect_with_caption:Save graph image {image_name}，desc: {caption}")

        return image_descriptions

    #------------------------------------------
    # 提取pdf中的图部分内容，并将描述整理成json格式
    #------------------------------------------
    def extract_images_with_captions(self) -> None:
        """提取 PDF 中的图像以及描述，并保存为 JSON 文件"""
        # 使用 'with' 语句打开 PDF 文件，自动管理文件关闭
        with fitz.open(self.pdf_path) as doc:
            logger.info(f"ExtractGraphFromPdf::extract_images_with_captions:Total number of pages in the document: {len(doc)}")
        
            # 遍历所有页面
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)  # 加载当前页面

                # 截图整个页面，用于后续分析
                page_image_path = self.save_page_to_png(page, page_num, dpi=300)
                #logger.info(f"页面大小(pdf单位pt): 宽 {page.rect.width} pt, 高 {page.rect.height} pt")

                # 提取页面描述
                captions = self.extract_page_captions(page)
                logger.info(f"ExtractGraphFromPdf::extract_images_with_captions: Page[{page_num}] Extract to description:{captions}")

                # 如果没有找到描述，就不进行边框检测了
                if len(captions) == 0:
                    logger.info(f"ExtractGraphFromPdf::extract_images_with_captions: No graph description found in page {page_num}")
                    continue

                # 使用 OpenCV 检测边框
                img = cv2.imread(page_image_path)
                descriptions = self.extract_rect_with_caption(img, captions)
                self.image_descriptions.update(descriptions)

            # 保存描述到 JSON 文件
            json_file = os.path.join(self.output_folder, os.path.splitext(os.path.basename(self.pdf_path))[0] + ".json")
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(self.image_descriptions, f, ensure_ascii=False, indent=4)

            logger.info(f"处理完成，共提取 {self.image_count} 张图片，描述存储在 {json_file}")

        # 文件会在 'with' 语句结束时自动关闭
        logger.info("ExtractGraphFromPdf::extract_images_with_captions: PDF document closed.")

#======================================================================================
def main():
    # 使用 argparse 解析命令行输入
    # 示例：python imgFromPdf.py 深度学习入门基于Python的理论与实现.pdf 
    parser = argparse.ArgumentParser(description="从 PDF 文件中提取带有标题的图像。")
    parser.add_argument("pdf_path", help="PDF 文件的路径（可以是完整路径或相对路径）")
    
    args = parser.parse_args()

    # 示例调用
    #pdf_path = "深度学习入门基于Python的理论与实现.pdf"
    # 初始化提取器并开始提取
    extractor = ExtractGraphFromPdf(args.pdf_path)
    # 提取图像和描述
    extractor.extract_images_with_captions()

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
