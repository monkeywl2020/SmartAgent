import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from loguru import logger


from ..utils.common import save_url_to_local_work_dir,has_chinese_chars,print_traceback,json_loads
from ..settings import DEFAULT_WORKSPACE

from ..msg.message import ContentItem


#这是一个全局字典，用于存储注册的工具类，工具类的名称作为键，工具类本身作为值。TOOL_REGISTRY 用来保存已注册的所有工具。
NUWA_TOOL_REGISTRY = {}

#register_tool 装饰器
# register_tool 是一个用于注册工具类的装饰器。它会检查工具名称是否已存在于 TOOL_REGISTRY 中。如果已经存在，则根据 allow_overwrite 参数决定是抛出异常还是覆盖已有工具。
#   name: 注册工具时使用的工具名称。
#   allow_overwrite: 是否允许覆盖已有同名工具。
def nuwa_register_tool(name, allow_overwrite=False):

    def decorator(cls):
        if name in NUWA_TOOL_REGISTRY:
            if allow_overwrite:
                logger.warning(f'Tool `{name}` already exists! Overwriting with class {cls}.')
            else:
                raise ValueError(f'Tool `{name}` already exists! Please ensure that the tool name is unique.')
        if cls.name and (cls.name != name):
            raise ValueError(f'{cls.__name__}.name="{cls.name}" conflicts with @register_tool(name="{name}").')
        cls.name = name
        NUWA_TOOL_REGISTRY[name] = cls
        logger.info(f"nuwa_register_tool: NUWA_TOOL_REGISTRY:{NUWA_TOOL_REGISTRY}")

        return cls

    return decorator

# 用于验证给定的工具参数是否符合 OpenAI 的工具调用标准，特别是验证参数格式是否符合 JSON schema。
# 该函数通过 jsonschema 库验证给定参数是否符合 OpenAI 工具调用要求。 
# 参数对象是否包含 name、description 和 parameters 字段。 
# parameters 是否符合 JSON schema。
# 如果符合规范，返回 True；否则返回 False。
def is_tool_schema(obj: dict) -> bool:
    """
    Check if obj is a valid JSON schema describing a tool compatible with OpenAI's tool calling.
    Example valid schema:
    {
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city and state, e.g. San Francisco, CA"
          },
          "unit": {
            "type": "string",
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location"]
      }
    }
    """
    import jsonschema
    try:
        assert set(obj.keys()) == {'name', 'description', 'parameters'}
        assert isinstance(obj['name'], str)
        assert obj['name'].strip()
        assert isinstance(obj['description'], str)
        assert isinstance(obj['parameters'], dict)

        assert set(obj['parameters'].keys()) == {'type', 'properties', 'required'}
        assert obj['parameters']['type'] == 'object'
        assert isinstance(obj['parameters']['properties'], dict)
        assert isinstance(obj['parameters']['required'], list)
        assert set(obj['parameters']['required']).issubset(set(obj['parameters']['properties'].keys()))
    except AssertionError:
        return False
    try:
        jsonschema.validate(instance={}, schema=obj['parameters'])
    except jsonschema.exceptions.SchemaError:
        return False
    except jsonschema.exceptions.ValidationError:
        pass
    return True

# BaseTool 类是所有工具类的基类，定义了工具的基本结构和方法。每个具体的工具类都需要继承自 BaseTool，并实现其 call 方法。
#   name: 工具的名称，必须唯一且由 register_tool 装饰器指定。
#   description: 工具的描述信息。
#   parameters: 工具的参数，可以是列表或字典形式，定义了工具需要的输入。
class NuwaBaseTool(ABC):
    name: str = ''
    description: str = ''
    parameters: Union[List[dict], dict] = []

    def __init__(self, cfg: Optional[dict] = None):
        self.cfg = cfg or {}
        #初始化时会检查 name 是否已设置。如果没有设置，则抛出 ValueError 异常。
        if not self.name: 
            raise ValueError(
                f'You must set {self.__class__.__name__}.name, either by @register_tool(name=...) or explicitly setting {self.__class__.__name__}.name'
            )
        
        # 增加新处理，qwen的tool和openai的tool或者其他格式的tool，
        
        #如果 parameters 是字典类型，则会验证其是否符合 OpenAI 工具调用要求。
        if isinstance(self.parameters, dict):
            if not is_tool_schema({'name': self.name, 'description': self.description, 'parameters': self.parameters}):
                raise ValueError(
                    'The parameters, when provided as a dict, must confirm to a valid openai-compatible JSON schema.')

    @abstractmethod
    def call(self, params: Union[str, dict], **kwargs) -> Union[str, list, dict, List[ContentItem]]:
        """The interface for calling tools.

        Each tool needs to implement this function, which is the workflow of the tool.

        Args:
            params: The parameters of func_call.
            kwargs: Additional parameters for calling tools.

        Returns:
            The result returned by the tool, implemented in the subclass.
        """
        raise NotImplementedError

    ################################################
    # 输出openAI 格式的 dict
    ################################################
    def openAI_format(self)  -> Dict[str, Any]:
        # 只有参数部分有区别，如果是list就要转换，如果dict默认就应该是openAI格式
        parameters = self.parameters

        if isinstance(parameters, list):
            params_dict = {} # 参数
            required_params = []# 必填参数
            
            # 遍历所有的参数
            for param in parameters:
                # 构建属性描述
                param_properties = {
                    "type": param['type'],
                    "description": param['description'],
                }

                # 如果有 enum，添加到属性中
                if 'enum' in param:
                    param_properties["enum"] = param['enum']   

                # 添加属性到 params_dict
                params_dict[param['name']] = param_properties

                # 判断是否必填：支持 'required' 和 'optional'
                is_required = param.get('required', not param.get('optional', False))
                if is_required:
                    required_params.append(param['name'])

            # 生成 OpenAI 格式的 parameters 内容
            parameters = {
                "type": "object",  # 这个参数的类型是 object，表示它是一个复杂对象（包含多个属性）
                "properties": params_dict,  # 将生成的参数字典作为 properties
                "required": required_params,  # 添加必填字段
                "additionalProperties": False  # 不允许有额外的属性
            }
        elif isinstance(parameters, dict):
            pass

        # 创建工具的 OpenAI 格式描述
        tool_json = {
            "name": self.name,
            "description": self.description,
            "parameters": parameters,  # 此时 parameters 直接作为工具参数
        }

        return tool_json
        

    #############################################################################
    # self.parameters 定义了工具的输入参数格式，包括字段名、字段类型、必需字段等。
    # params 是 在call的时候，需要检查传入的参数，看符合不符合格式
    # 
    #############################################################################
    def _verify_json_format_args(self, params: Union[str, dict], strict_json: bool = False) -> dict:
        """Verify the parameters of the function call"""
        if isinstance(params, str):
            try:
                if strict_json:
                    params_json: dict = json.loads(params)
                else:
                    params_json: dict = json_loads(params)
            except json.decoder.JSONDecodeError:
                raise ValueError('Parameters must be formatted as a valid JSON!')
        else:
            params_json: dict = params
        if isinstance(self.parameters, list):
            for param in self.parameters:
                if 'required' in param and param['required']:
                    if param['name'] not in params_json:
                        raise ValueError('Parameters %s is required!' % param['name'])
        elif isinstance(self.parameters, dict):
            import jsonschema
            jsonschema.validate(instance=params_json, schema=self.parameters)
        else:
            raise ValueError
        return params_json

    # 下面是返回这个函数的信息
    #   name            函数名称
    #   description     函数说明
    #   parameters      函数的参数
    # 名称和说明都是标准的，参数的格式openai和qwen是不一样的
    @property
    def function_info(self) -> dict:
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters
        }

    @property
    def file_access(self) -> bool:
        return False


class NuwaBaseToolWithFileAccess(NuwaBaseTool, ABC):

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        assert self.name
        default_work_dir = os.path.join(DEFAULT_WORKSPACE, 'tools', self.name)
        self.work_dir: str = self.cfg.get('work_dir', default_work_dir)

    @property
    def file_access(self) -> bool:
        return True

    def call(self, params: Union[str, dict], files: List[str] = None, **kwargs) -> str:
        # Copy remote files to the working directory:
        if files:
            os.makedirs(self.work_dir, exist_ok=True)
            for file in files:
                try:
                    save_url_to_local_work_dir(file, self.work_dir)
                except Exception:
                    print_traceback()

        # Then do something with the files:
        # ...
