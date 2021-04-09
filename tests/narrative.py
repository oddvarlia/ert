import asyncio
from asyncio.events import AbstractEventLoop
from asyncio.queues import QueueEmpty
from typing import Any, Dict, List, Optional
import uuid
import threading
import functools
from ert_shared.ensemble_evaluator.ws_util import wait
from websockets.server import WebSocketServer

from fnmatch import fnmatchcase

try:
    from typing import TypedDict  # >=3.8
except ImportError:
    from mypy_extensions import TypedDict  # <=3.7

import websockets
from cloudevents.http import CloudEvent, to_json, from_json


class _ConnectionInformation(TypedDict):  # type: ignore
    uri: str
    proto: str
    hostname: str
    port: int
    path: str
    base_uri: str


class EventDescription(TypedDict):  # type: ignore
    id_: str
    source: str
    type_: str
    datacontenttype: Optional[str]
    subject: Optional[str]
    data: Optional[Any]


class _Event:
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
            if "*" in self.source:
                assert fnmatchcase(other["source"], self.source), msg
            elif "*" in other["source"]:
                assert fnmatchcase(self.source, other["source"]), msg
            else:
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


class _NarrativeMock:
    def __init__(
        self,
        interactions: List[_InteractionDefinition],
        conn_info: _ConnectionInformation,
    ) -> None:
        self._interactions: List[_InteractionDefinition] = interactions
        self._loop: Optional[AbstractEventLoop] = None
        self._ws: Optional[WebSocketServer] = None
        self.uri: str = conn_info["uri"]
        self._conn_info = conn_info

        # A queue on which errors will be put
        self._errors: asyncio.Queue = asyncio.Queue()

    async def _ws_handler(self, websocket, path):
        expected_path = self._conn_info["path"]
        if path != expected_path:
            print(f"not handling {path} as it is not the expected path {expected_path}")
            return
        for interaction in reversed(self._interactions):
            if type(interaction) == _InteractionDefinition:
                e = TypeError(
                    "the first interaction needs to be promoted to either response or receive"
                )
                self._errors.put_nowait(e)
            elif isinstance(interaction, _ReceiveDefinition):
                for event in interaction.events:
                    received_event = await websocket.recv()
                    try:
                        event.assert_matches(from_json(received_event))
                    except AssertionError as e:
                        self._errors.put_nowait(e)
                print("OK", interaction.scenario)
            elif isinstance(interaction, _ResponseDefinition):
                for event in interaction.events:
                    await websocket.send(to_json(event.to_cloudevent()))
                print("OK", interaction.scenario)
            else:
                e = TypeError(f"expected either receive or response, got {interaction}")
                self._errors.put_nowait(e)

    def _sync_ws(self, delay_startup=0):
        self._loop = asyncio.new_event_loop()
        self._done = self._loop.create_future()

        async def _serve():
            await asyncio.sleep(delay_startup)
            ws = await websockets.serve(
                self._ws_handler, self._conn_info["hostname"], self._conn_info["port"]
            )
            await self._done
            ws.close()
            await ws.wait_closed()

        self._loop.run_until_complete(_serve())
        self._loop.close()

    async def _verify(self):
        errors = []
        while True:
            try:
                errors.append(self._errors.get_nowait())
            except QueueEmpty:
                break
        return errors

    def __enter__(self):
        self._ws_thread = threading.Thread(target=self._sync_ws)
        self._ws_thread.start()
        if asyncio.get_event_loop().is_running():
            raise RuntimeError(
                "sync narrative should control the loop, maybe you called it from within an async test?"
            )
        asyncio.get_event_loop().run_until_complete(
            wait(self._conn_info["base_uri"], 2)
        )

    def __exit__(self, *args, **kwargs):
        self._loop.call_soon_threadsafe(self._done.set_result, None)
        self._ws_thread.join()
        errors = asyncio.get_event_loop().run_until_complete(self._verify())
        if errors:
            raise AssertionError(errors)

    async def __aenter__(self):
        self._ws = await websockets.serve(
            self._ws_handler, self._conn_info["hostname"], self._conn_info["port"]
        )
        return self

    async def __aexit__(self, *args):
        self._ws.close()
        await self._ws.wait_closed()
        errors = await self._verify()
        if errors:
            raise AssertionError(errors)


class _Narrative:
    def __init__(self, consumer: "Consumer", provider: "Provider", uri: str) -> None:
        self.consumer = consumer
        self.provider = provider
        self.interactions: List[_InteractionDefinition] = []
        self._mock: Optional[_NarrativeMock] = None
        self.uri: str = uri
        proto, hostname, port = uri.split(":")
        path = ""
        if "/" in port:
            port, path = port.split("/")
        if proto == "wss":
            raise ValueError("cannot mock secure socket")
        hostname = hostname[2:]
        self.path = "/" + path
        self.port = int(port)
        self.hostname = hostname
        self.base_uri = f"{proto}://{hostname}:{port}"
        self._conn_info = _ConnectionInformation(
            uri=self.uri,
            proto=proto,
            hostname=self.hostname,
            port=self.port,
            path=self.path,
            base_uri=self.base_uri,
        )

    def given(self, provider_state: Optional[str], **params) -> "_Narrative":
        state = None
        if provider_state:
            state = [{"name": provider_state, "params": params}]
        self.interactions.insert(0, _InteractionDefinition(state))
        return self

    def and_given(self, provider_state: str, **params) -> "_Narrative":
        raise NotImplementedError("not yet implemented")

    def receives(self, scenario: str) -> "_Narrative":
        def_ = self.interactions[-1]
        if type(def_) == _InteractionDefinition:
            def_.__class__ = _ReceiveDefinition
        elif isinstance(def_, _ResponseDefinition) and not def_.events:
            raise ValueError("receive followed an empty response scenario")
        else:
            def_ = _ReceiveDefinition(self.interactions[-1].provider_states)
            self.interactions.insert(0, def_)
        def_.scenario = scenario
        return self

    def responds_with(self, scenario: str) -> "_Narrative":
        def_ = self.interactions[-1]
        if type(def_) == _InteractionDefinition:
            def_.__class__ = _ResponseDefinition
        elif isinstance(def_, _ReceiveDefinition) and not def_.events:
            raise ValueError("response followed an empty receive scenario")
        else:
            def_ = _ResponseDefinition(self.interactions[-1].provider_states)
            self.interactions.insert(0, def_)
        def_.scenario = scenario
        return self

    def cloudevents_in_order(self, events: List[EventDescription]) -> "_Narrative":
        cloudevents = []
        for event in events:
            cloudevents.append(_Event(event))
        self.interactions[0].events = cloudevents
        return self

    def __enter__(self):
        self._mock = _NarrativeMock(self.interactions, self._conn_info)
        return self._mock.__enter__()

    def __exit__(self, *args, **kwargs):
        self._mock.__exit__(*args, **kwargs)

    async def __aenter__(self):
        self._mock = _NarrativeMock(self.interactions, self._conn_info)
        return await self._mock.__aenter__()

    async def __aexit__(self, *args):
        await self._mock.__aexit__(*args)


class _Actor:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name


class Provider(_Actor):
    pass


class Consumer(_Actor):
    def forms_narrative_with(self, provider: Provider, uri, **kwargs) -> _Narrative:
        return _Narrative(self, provider, uri, **kwargs)
