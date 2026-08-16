"""
模块通信模块 (Message Bus)

发布-订阅模式的消息总线, 支持:
  - 发布/订阅(pub/sub)
  - 点对点消息
  - 广播
  - 消息优先级
  - 中间件拦截
  - 消息历史

每个模块通过BusModule基类挂载到总线, 自动获得通信能力。
"""
import time
import uuid
from typing import Dict, List, Callable, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum


class MessageType(Enum):
    """消息类型"""
    # 感知
    SENSORY_INPUT = "sensory.input"
    PERCEPTION_RESULT = "perception.result"
    # 记忆
    MEMORY_STORE = "memory.store"
    MEMORY_RECALL = "memory.recall"
    MEMORY_RESULT = "memory.result"
    # 情绪
    EMOTION_UPDATE = "emotion.update"
    DOPAMINE_RELEASE = "dopamine.release"
    # 认知
    ATTENTION_SHIFT = "attention.shift"
    THOUGHT = "thought"
    # 意识
    CONSCIOUS_BROADCAST = "conscious.broadcast"
    # 动作
    ACTION = "action"
    RESPONSE = "response"
    # 通用
    CUSTOM = "custom"
    QUERY = "query"
    BROADCAST = "broadcast"


@dataclass
class Message:
    """总线消息"""
    msg_type: MessageType = MessageType.CUSTOM
    sender: str = "unknown"
    receiver: Optional[str] = None  # None=广播
    content: Any = None
    topic: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    priority: int = 0  # 0=普通 1=高 2=紧急
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    reply_to: Optional[str] = None

    def reply(self, content: Any,
              msg_type: MessageType = MessageType.RESPONSE) -> 'Message':
        return Message(
            msg_type=msg_type, sender=self.receiver or "unknown",
            receiver=self.sender, content=content, reply_to=self.id
        )


Middleware = Callable[[Message], Optional[Message]]


class MessageBus:
    """消息总线"""

    def __init__(self, history_size: int = 1000):
        self.subscriptions: Dict[str, List[tuple]] = defaultdict(list)
        self.message_history: deque = deque(maxlen=history_size)
        self.modules: Dict[str, 'BusModule'] = {}
        self._middleware: List[Middleware] = []
        self.stats = {
            'total_messages': 0,
            'messages_by_type': defaultdict(int),
        }

    def register_module(self, module: 'BusModule'):
        """注册模块"""
        self.modules[module.module_name] = module

    def subscribe(self, module_name: str, msg_type: MessageType,
                  handler: Callable):
        """订阅消息"""
        self.subscriptions[msg_type.value].append((module_name, handler))

    def unsubscribe(self, module_name: str, msg_type: MessageType):
        """取消订阅"""
        self.subscriptions[msg_type.value] = [
            (n, h) for n, h in self.subscriptions[msg_type.value]
            if n != module_name
        ]

    def publish(self, msg: Message):
        """发布消息(广播或点对点)"""
        # 中间件处理
        for mw in self._middleware:
            msg = mw(msg)
            if msg is None:
                return

        self.message_history.append(msg)
        self.stats['total_messages'] += 1
        self.stats['messages_by_type'][msg.msg_type.value] += 1

        if msg.receiver:
            # 点对点
            target = self.modules.get(msg.receiver)
            if target:
                target.receive_message(msg)
        else:
            # 广播给订阅者
            for mod_name, handler in self.subscriptions.get(msg.msg_type.value, []):
                handler(msg)
            # 也广播给BROADCAST订阅者
            if msg.msg_type != MessageType.BROADCAST:
                for mod_name, handler in self.subscriptions.get(
                        MessageType.BROADCAST.value, []):
                    handler(msg)

    def send(self, sender: str, receiver: str, msg_type: MessageType,
             content: Any = None, **metadata) -> Optional[Message]:
        """发送点对点消息并等待回复"""
        msg = Message(
            msg_type=msg_type, sender=sender, receiver=receiver,
            content=content, metadata=metadata
        )
        target = self.modules.get(receiver)
        if target:
            return target.handle_request(msg)
        return None

    def add_middleware(self, mw: Middleware):
        """添加中间件"""
        self._middleware.append(mw)

    def get_history(self, msg_type: MessageType = None,
                    limit: int = 50) -> List[Message]:
        """获取消息历史"""
        msgs = list(self.message_history)
        if msg_type:
            msgs = [m for m in msgs if m.msg_type == msg_type]
        return msgs[-limit:]

    def get_stats(self) -> Dict:
        return {
            'total_messages': self.stats['total_messages'],
            'by_type': dict(self.stats['messages_by_type']),
            'modules': list(self.modules.keys()),
            'subscriptions': {
                k: len(v) for k, v in self.subscriptions.items() if v
            },
        }


class BusModule:
    """可挂载到总线的模块基类"""

    def __init__(self, name: str, bus: Optional[MessageBus] = None):
        self.module_name = name
        self.bus = bus
        self._message_queue: deque = deque(maxlen=100)
        if bus:
            self.attach_bus(bus)

    def attach_bus(self, bus: MessageBus):
        """挂载到总线"""
        self.bus = bus
        bus.register_module(self)

    def publish(self, msg_type: MessageType, content: Any = None,
                topic: str = "", **metadata):
        """发布消息"""
        if self.bus:
            self.bus.publish(Message(
                msg_type=msg_type, sender=self.module_name,
                content=content, topic=topic, metadata=metadata
            ))

    def send(self, receiver: str, msg_type: MessageType,
             content: Any = None, **metadata) -> Optional[Message]:
        """发送私信"""
        if self.bus:
            return self.bus.send(
                self.module_name, receiver, msg_type, content, **metadata)
        return None

    def subscribe(self, msg_type: MessageType, handler: Callable):
        """订阅消息"""
        if self.bus:
            self.bus.subscribe(self.module_name, msg_type, handler)

    def receive_message(self, msg: Message):
        """接收消息(子类可覆盖)"""
        self._message_queue.append(msg)

    def handle_request(self, msg: Message) -> Optional[Message]:
        """处理请求(子类可覆盖)"""
        self.receive_message(msg)
        return None

    def get_pending_messages(self) -> List[Message]:
        """获取待处理消息"""
        msgs = list(self._message_queue)
        self._message_queue.clear()
        return msgs
