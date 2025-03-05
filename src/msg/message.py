from typing import List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, field_validator, model_validator

DEFAULT_SYSTEM_MESSAGE = 'You are a helpful assistant.'

ROLE = 'role'
CONTENT = 'content'
NAME = 'name'

SYSTEM = 'system'
USER = 'user'
ASSISTANT = 'assistant'
FUNCTION = 'function'
TOOL = 'tool'

FILE = 'file'
IMAGE = 'image'
AUDIO = 'audio'
VIDEO = 'video'

#继承自 pydantic.BaseModel，提供了字典式访问和操作的能力，增强了模型的灵活性。
class BaseModelCompatibleDict(BaseModel):
    #允许像字典一样访问属性
    # 例如： obj['key'] = value
    #        value = obj['key']
    def __getitem__(self, item):
        return getattr(self, item)

    #允许像字典一样设置属性
    def __setitem__(self, key, value):
        setattr(self, key, value)
    
    #调用 BaseModel 的 model_dump 和 model_dump_json 方法，但默认排除 None 值，保持输出简洁。
    def model_dump(self, **kwargs): # 返回python 字典 字典数据结构
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        return super().model_dump(**kwargs)# 解析的时候添加了 exclude_none = True 参数，会把 值为none的过滤掉

    def model_dump_json(self, **kwargs): # 返回JSON 字符串（文本格式）
        if 'exclude_none' not in kwargs:
            kwargs['exclude_none'] = True
        return super().model_dump_json(**kwargs) # 解析的时候添加了 exclude_none = True 参数，会把 值为none的过滤掉

    #模拟字典的 get 方法，安全地获取属性值，未找到时返回默认值
    def get(self, key, default=None):
        try:
            value = getattr(self, key)
            if value:
                return value
            else:
                return default
        except AttributeError:
            return default

    #将模型的 model_dump() 输出转换为字符串
    def __str__(self):
        return f'{self.model_dump()}'

#表示一个函数调用对象 ，包含：
# name: 函数名。
# arguments: 函数参数。
class FunctionCall(BaseModelCompatibleDict):
    name: str
    arguments: str

    def __init__(self, name: str, arguments: str):
        super().__init__(name=name, arguments=arguments)

    #返回对象的调试信息，格式化为字符串形式
    def __repr__(self):
        return f'FunctionCall({self.model_dump()})'

#表示多模态内容的单个条目，可以是文本、图片、文件、音频或视频, 但只能是一种
class ContentItem(BaseModelCompatibleDict):
    text: Optional[str] = None
    image: Optional[str] = None
    file: Optional[str] = None
    audio: Optional[str] = None
    video: Optional[Union[str, list]] = None

    def __init__(self,
                 text: Optional[str] = None,
                 image: Optional[str] = None,
                 file: Optional[str] = None,
                 audio: Optional[str] = None,
                 video: Optional[Union[str, list]] = None):
        super().__init__(text=text, image=image, file=file, audio=audio, video=video)

    '''
    @model_validator 是 Pydantic 的装饰器，用于定义模型级别的验证逻辑（与字段级别的验证 @field_validator 不同）。
    当使用 mode='after' 时，验证器会在所有字段验证完成后以及模型实例创建完成后执行。
    '''
    @model_validator(mode='after')
    def check_exclusivity(self):  # 检查提供的字段中是否仅有一个非空，否则抛出 ValueError
        provided_fields = 0
        if self.text is not None:
            provided_fields += 1
        if self.image:
            provided_fields += 1
        if self.file:
            provided_fields += 1
        if self.audio:
            provided_fields += 1
        if self.video:
            provided_fields += 1

        #text, image, file, audio, video 中必须有且仅有一个字段非空（通过模型验证器检查）
        if provided_fields != 1:
            raise ValueError("Exactly one of 'text', 'image', 'file', 'audio', or 'video' must be provided.")
        return self

    def __repr__(self):
        return f'ContentItem({self.model_dump()})'
    
    #返回内容的类型和值，结果为元组
    def get_type_and_value(self) -> Tuple[Literal['text', 'image', 'file', 'audio', 'video'], str]:
        # self.model_dump() 调用 Pydantic 的 model_dump() 方法，将当前模型实例转化为一个字典
        # 字典的 .items() 方法返回一个键值对的迭代器，每个元素是一个 (key, value) 元组。
        # (t, v), 表示从迭代器中获取第一个键值对（一个元组），并将元组解包到变量 t 和 v。
        # , 的作用是确保迭代器中只有一个键值对，否则会引发解包错误。
        (t, v), = self.model_dump().items()
        assert t in ('text', 'image', 'file', 'audio', 'video')
        return t, v

    # 返回ContentItem的 type
    @property
    def type(self) -> Literal['text', 'image', 'file', 'audio', 'video']:
        t, v = self.get_type_and_value()
        return t

    # 返回ContentItem的 value
    @property
    def value(self) -> str:
        t, v = self.get_type_and_value()
        return v


class Message(BaseModelCompatibleDict):
    role: str
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None
    function_call: Optional[FunctionCall] = None
    tool_calls: Optional[List[dict]] = None
    id: Optional[str] = None  # 新增字段
    extra: Optional[dict] = None

    def __init__(self,
                 role: str,
                 content: Optional[Union[str, List[ContentItem]]],
                 name: Optional[str] = None,
                 function_call: Optional[FunctionCall] = None,
                 tool_calls: Optional[dict] = None,
                 extra: Optional[dict] = None,
                 **kwargs):
        if content is None:
            content = ''
        super().__init__(role=role, content=content, name=name, function_call=function_call, tool_calls = tool_calls, extra=extra,**kwargs)

    #返回对象的调试信息，便于调试。
    def __repr__(self):
        return f'Message({self.model_dump()})'
    
    '''
    @field_validator('role') 是 Pydantic 的一个装饰器，用来定义一个字段验证器（Field Validator）。这个装饰器的作用是：
    对指定的字段（这里是 role）的值进行验证和处理。
    如果验证失败，可以抛出异常；如果验证通过，可以返回处理后的值。
    '''
    @field_validator('role')
    def role_checker(cls, value: str) -> str:
        if value not in [USER, ASSISTANT, SYSTEM, FUNCTION,TOOL]: #确保 role 的值在指定范围内（USER, ASSISTANT, SYSTEM, FUNCTION,TOOL）
            raise ValueError(f'{value} must be one of {",".join([USER, ASSISTANT, SYSTEM, FUNCTION,TOOL])}')
        return value
