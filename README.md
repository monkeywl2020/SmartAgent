# SmartAgent
多agent协同工作的AI-agent框架，采用异步微服务框架

1:以数字人为应用层，可以配置多个agent协作。应用层关联和管理用户的历史记录这些内容。用户的历史记录信息存储在数据库中，mysql和redis数据库。

2：agent层面，2个agent，每个agent都是根据配置生成的，各自作用不一样。agent负责调用 大模型客户端与模型进行交互，获取模型返回的响应并解析其中的 function call或者tool call。

3：模型客户端层面，目前支持openAI格式，postApi格式（http），qwen-agent客户端格式。。。其他的后续增加

4：agent工作可以有2种形态，一种是多agent串行工作，一种是并行工作。

