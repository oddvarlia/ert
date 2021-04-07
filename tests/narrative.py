from typing import Any, Dict, List, Optional
import uuid

try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7

import websockets
from cloudevents.http import CloudEvent, to_json, from_json


class EventDescription(TypedDict):  # type: ignore
    id_: str
    source: str
    type_: str
    datacontenttype: Optional[str]
    subject: Optional[str]
    data: Optional[Any]


class _Event:
    def __repr__(self) -> str:
        return str(self.__dict__)

    def __init__(self, description: EventDescription) -> None:
        self._id = description.get("id_", uuid.uuid4())
        self.source = description["source"]
        self.type_ = description["type_"]
        self.datacontenttype = description.get("datacontenttype")
        self.subject = description.get("subject")
        self.data = description.get("data")

    def assert_matches(self, other: CloudEvent):
        msg = f"{self} did not match {other}"
        if self.source:
            assert self.source == other["source"], msg
        if self.type_:
            assert self.type_ == other["type"], msg
        if self.subject:
            assert self.subject == other["subject"], msg
        if self.data:
            assert self.data == other.data, msg
        if self.datacontenttype:
            assert self.datacontenttype == other["datacontenttype"], msg

    def to_cloudevent(self) -> CloudEvent:
        attrs = {}
        if self.source:
            attrs["source"] = self.source
        if self.type_:
            attrs["type"] = self.type_
        if self.subject:
            attrs["subject"] = self.subject
        if self.datacontenttype:
            attrs["datacontenttype"] = self.datacontenttype
        return CloudEvent(attrs, self.data)


class _InteractionDefinition:
    def __init__(self, provider_states: Optional[List[Dict[str, Any]]]) -> None:
        self.provider_states: Optional[List[Dict[str, Any]]] = provider_states
        self.scenario: str = ""
        self.events: List[_Event] = []


class _ReceiveDefinition(_InteractionDefinition):
    pass


class _ResponseDefinition(_InteractionDefinition):
    pass


class _Narrative:
    def __init__(self, consumer: "Consumer", provider: "Provider") -> None:
        self.consumer = consumer
        self.provider = provider
        self._interactions: List[_InteractionDefinition] = []
        self.uri: str = ""

    def given(self, provider_state: Optional[str], **params) -> "_Narrative":
        state = None
        if provider_state:
            state = [{"name": provider_state, "params": params}]
        self._interactions.insert(0, _InteractionDefinition(state))
        return self

    def and_given(self, provider_state: str, **params) -> "_Narrative":
        # self._interactions[-1]["providerStates"].append({"name": provider_state, "params": params})
        raise NotImplementedError("not yet implemented")
        # return self

    def receives(self, scenario: str) -> "_Narrative":
        def_ = self._interactions[-1]
        if type(def_) == _InteractionDefinition:
            def_.__class__ = _ReceiveDefinition
        elif isinstance(def_, _ResponseDefinition) and not def_.events:
            raise ValueError("receive followed an empty response scenario")
        else:
            def_ = _ReceiveDefinition(self._interactions[-1].provider_states)
            self._interactions.insert(0, def_)
        def_.scenario = scenario
        return self

    def responds_with(self, scenario: str) -> "_Narrative":
        def_ = self._interactions[-1]
        if type(def_) == _InteractionDefinition:
            def_.__class__ = _ResponseDefinition
        elif isinstance(def_, _ReceiveDefinition) and not def_.events:
            raise ValueError("response followed an empty receive scenario")
        else:
            def_ = _ResponseDefinition(self._interactions[-1].provider_states)
            self._interactions.insert(0, def_)
        def_.scenario = scenario
        return self

    def cloudevents_in_order(self, events: List[EventDescription]) -> "_Narrative":
        cloudevents = []
        for event in events:
            cloudevents.append(_Event(event))
        self._interactions[0].events = cloudevents
        return self

    async def __aenter__(self):
        async def _handler(websocket, path):
            for interaction in reversed(self._interactions):
                if type(interaction) == _InteractionDefinition:
                    raise TypeError(
                        "the first interaction needs to be promoted to either response or receive"
                    )
                elif isinstance(interaction, _ReceiveDefinition):
                    for event in interaction.events:
                        received_event = await websocket.recv()
                        event.assert_matches(from_json(received_event))
                    print("OK", interaction.scenario)
                elif isinstance(interaction, _ResponseDefinition):
                    for event in interaction.events:
                        await websocket.send(to_json(event.to_cloudevent()))
                    print("OK", interaction.scenario)
                else:
                    raise TypeError(
                        f"expected either receive or response, got {interaction}"
                    )

        proto, hostname, port = self.uri.split(":")
        if proto == "wss":
            raise ValueError("cannot mock secure socket")
        hostname = hostname[2:]
        self._ws = await websockets.serve(_handler, hostname, int(port))

    async def __aexit__(self, *args):
        self._ws.close()
        await self._ws.wait_closed()


class _Actor:
    def __init__(self, name: str) -> None:
        self.name = name


class Provider(_Actor):
    pass


class Consumer(_Actor):
    def forms_narrative_with(self, provider: Provider) -> _Narrative:
        return _Narrative(self, provider)
