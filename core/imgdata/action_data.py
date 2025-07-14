# core/imgdata/action_data.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class ImageRule:
    """
    ImageRule: 图像实体规则，支持 uid 判断和数量约束。
    字段说明：
        target_uid       : 相关实体 uid 列表。
        forbid_uid       : 禁止存在的实体 uid 列表。
        target_limit_min : 目标实体最小数量。
        target_limit_max : 目标实体最大数量。
    """
    target_uid: List[str] = field(default_factory=list)
    target_limit_min: Optional[int] = None
    target_limit_max: Optional[int] = None
    forbid_uid: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，便于序列化和存储。
        """
        return {
            "target_uid": self.target_uid,
            "target_limit_min": self.target_limit_min,
            "target_limit_max": self.target_limit_max,
            "forbid_uid": self.forbid_uid
        }

    def from_dict(self, data: Dict[str, Any]) -> 'ImageRule':
        """
        从字典格式加载数据，便于反序列化。
        """
        self.target_uid = data.get("target_uid", [])
        self.target_limit_min = data.get("target_limit_min")
        self.target_limit_max = data.get("target_limit_max")
        self.forbid_uid = data.get("forbid_uid", [])
        return self

@dataclass
class StateUnit:
    """
    StateUnit: 结构化描述图像和上下文环境的客观状态。
    用于动作与状态分离、追溯和流程控制。

    字段说明：
        state_id     : 状态唯一标识符。
        image_rule   : 图像实体规则（ImageRule 类实例）。
        context_rule : 上下文规则（如文本描述、环境约束）。
        describe     : 状态描述（如“主窗口可见”、“按钮A可点击”）。
    """
    state_id: str = None
    image_rule: Optional[ImageRule] = None
    context_rule: Optional[str] = None
    describe: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，便于序列化和存储。
        """
        return {
            "state_id": self.state_id,
            "image_rule": self.image_rule.to_dict() if self.image_rule else None,
            "context_rule": self.context_rule,
            "describe": self.describe
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StateUnit':
        """
        从字典格式加载数据，便于反序列化。
        """
        state = cls()
        state.state_id = data.get("state_id")
        if "image_rule" in data and data["image_rule"]:
            state.image_rule = ImageRule().from_dict(data["image_rule"])
        state.context_rule = data.get("context_rule")
        state.describe = data.get("describe")
        return state


@dataclass
class ActionUnit:
    """
    ActionUnit: 结构化描述具体动作（如点击、输入、拖动等），不关心结果。

    字段说明：
        action_id   : 动作唯一标识。
        subject     : 发起动作的实体（如“按钮A”、“图片A”）。
        subject_pos : 实体相对位置 (x, y)，如 (0.5, 0.5) 表示中心。
        action_type : 动作类型（如“click”、“input”、“drag”）。
        parameters  : 动作参数（如位置、文本等）。
        description : 动作描述。
        pre_state   : 执行前的状态列表。
    """
    action_id: str
    subject: str
    subject_pos: tuple[float, float] = (0.5, 0.5)
    action_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    description: Optional[str] = None
    pre_state: List[StateUnit] = field(default_factory=list)
    post_state: List[StateUnit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，便于序列化和存储。
        """
        return {
            "action_id": self.action_id,
            "subject": self.subject,
            "subject_pos": self.subject_pos,
            "action_type": self.action_type,
            "parameters": self.parameters,
            "description": self.description,
            "pre_state": [state.to_dict() for state in self.pre_state],
            "post_state": [state.to_dict() for state in self.post_state]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ActionUnit':
        """
        从字典格式加载数据，便于反序列化。
        """
        action = cls(
            action_id=data.get("action_id"),
            subject=data.get("subject"),
            subject_pos=tuple(data.get("subject_pos", (0.5, 0.5))),
            action_type=data.get("action_type"),
            parameters=data.get("parameters", {}),
            description=data.get("description")
        )
        action.pre_state = [StateUnit.from_dict(state) for state in data.get("pre_state", [])]
        action.post_state = [StateUnit.from_dict(state) for state in data.get("post_state", [])]
        return action


@dataclass
class WorkflowNode:
    """
    WorkflowNode: 流程节点，集成意图（目的）与动作。

    字段说明：
        node_id       : 节点唯一标识。
        description   : 节点描述。
        action        : 节点对应的具体动作。
        next_nodes    : 下一步节点 id 列表。
        timeout       : 动作执行的时间上限（秒）。
        poll_interval : 动作轮询的时间间隔（秒）。
        poll_limit    : 动作轮询的最大次数（防止死循环）。
    """
    node_id: str
    description: Optional[str] = None
    action: ActionUnit
    next_nodes: List[str] = field(default_factory=list)
    timeout: Optional[float] = None
    poll_interval: Optional[float] = None
    poll_limit: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，便于序列化和存储。
        """
        return {
            "node_id": self.node_id,
            "description": self.description,
            "action": self.action.to_dict(),
            "next_nodes": self.next_nodes,
            "timeout": self.timeout,
            "poll_interval": self.poll_interval,
            "poll_limit": self.poll_limit
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowNode':
        """
        从字典格式加载数据，便于反序列化。
        """
        action = ActionUnit.from_dict(data.get("action", {}))
        return cls(
            node_id=data.get("node_id"),
            description=data.get("description"),
            action=action,
            next_nodes=data.get("next_nodes", []),
            timeout=data.get("timeout"),
            poll_interval=data.get("poll_interval"),
            poll_limit=data.get("poll_limit")
        )


@dataclass
class WorkflowGraph:
    """
    WorkflowGraph: 任务/流程结构，支持分支和条件跳转。

    字段说明：
        workflow_id  : 任务唯一标识。
        name         : 任务名称。
        description  : 任务描述。
        nodes        : 所有节点（node_id -> WorkflowNode）。
        start_node   : 起始节点 id。
        final_states : 终止的状态。
    """
    workflow_id: str
    name: str
    description: Optional[str] = None
    nodes: Dict[str, WorkflowNode] = field(default_factory=dict)
    start_node: str = ""
    final_states: List[StateUnit] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式，便于序列化和存储。
        """
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "nodes": {node_id: node.to_dict() for node_id, node in self.nodes.items()},
            "start_node": self.start_node,
            "final_states": [state.to_dict() for state in self.final_states]
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkflowGraph':
        """
        从字典格式加载数据，便于反序列化。
        """
        workflow = cls(
            workflow_id=data.get("workflow_id"),
            name=data.get("name"),
            description=data.get("description"),
            start_node=data.get("start_node", "")
        )
        workflow.nodes = {node_id: WorkflowNode.from_dict(node) for node_id, node in data.get("nodes", {}).items()}
        workflow.final_states = [StateUnit.from_dict(state) for state in data.get("final_states", [])]
        return workflow
