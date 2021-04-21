from contextlib import contextmanager

import websockets
import pytest
import asyncio
import threading
import queue
import time

from ert_shared.ensemble_evaluator.client import Client
from ert_shared.ensemble_evaluator.entity import serialization
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_STARTED,
    JOB_STATE_FAILURE,
    JOB_STATE_FINISHED,
    JOB_STATE_RUNNING,
    ENSEMBLE_STATE_STOPPED,
)
from cloudevents.http import from_json
from ert_shared.ensemble_evaluator.evaluator import (
    EnsembleEvaluator,
    ee_monitor,
)
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.ensemble_evaluator.entity.snapshot import Snapshot
from tests.ensemble_evaluator.ensemble_test import TestEnsemble, send_dispatch_event
from tests.narrative import EventDescription
from tests.narratives import monitor_happy_path_narrative


@pytest.fixture
def ee_config(unused_tcp_port):
    return EvaluatorServerConfig(unused_tcp_port)


@pytest.fixture
def evaluator(ee_config):
    ensemble = TestEnsemble(0, 2, 1, 2)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    yield ee
    ee.stop()


def test_dispatchers_can_connect_and_monitor_can_shut_down_evaluator(evaluator):
    with evaluator.run() as monitor:
        events = monitor.track()
        host = evaluator._config.host
        port = evaluator._config.port
        url = evaluator._config.url
        # first snapshot before any event occurs
        snapshot_event = next(events)
        snapshot = Snapshot(snapshot_event.data)
        assert snapshot.get_status() == ENSEMBLE_STATE_STARTED
        # two dispatchers connect

        with Client(
            url + "/dispatch", max_retries=1, timeout_multiplier=1
        ) as dispatch1, Client(
            url + "/dispatch", max_retries=1, timeout_multiplier=1
        ) as dispatch2:

            # first dispatcher informs that job 0 is running
            send_dispatch_event(
                dispatch1,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                "/ert/ee/0/real/0/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )
            snapshot = Snapshot(next(events).data)
            assert snapshot.get_job("0", "0", "0").status == JOB_STATE_RUNNING

            # second dispatcher informs that job 0 is running
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_RUNNING,
                "/ert/ee/0/real/1/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )
            snapshot = Snapshot(next(events).data)
            assert snapshot.get_job("1", "0", "0").status == JOB_STATE_RUNNING

            # second dispatcher informs that job 0 is done
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_SUCCESS,
                "/ert/ee/0/real/1/step/0/job/0",
                "event1",
                {"current_memory_usage": 1000},
            )
            snapshot = Snapshot(next(events).data)
            assert snapshot.get_job("1", "0", "0").status == JOB_STATE_FINISHED

            # second dispatcher informs that job 1 is failed
            send_dispatch_event(
                dispatch2,
                identifiers.EVTYPE_FM_JOB_FAILURE,
                "/ert/ee/0/real/1/step/0/job/1",
                "event_job_1_fail",
                {identifiers.ERROR_MSG: "error"},
            )
            snapshot = Snapshot(next(events).data)
            assert snapshot.get_job("1", "0", "1").status == JOB_STATE_FAILURE

            # a second monitor connects
            with ee_monitor.create(host, port) as monitor2:
                events2 = monitor2.track()
                snapshot = Snapshot(next(events2).data)
                assert snapshot.get_status() == ENSEMBLE_STATE_STARTED
                assert snapshot.get_job("0", "0", "0").status == JOB_STATE_RUNNING
                assert snapshot.get_job("1", "0", "0").status == JOB_STATE_FINISHED

                # one monitor requests that server exit
                monitor.signal_cancel()

                # both monitors should get a terminated event
                terminated = next(events)
                terminated2 = next(events2)
                assert terminated["type"] == identifiers.EVTYPE_EE_TERMINATED
                assert terminated2["type"] == identifiers.EVTYPE_EE_TERMINATED

                for e in [events, events2]:
                    for _ in e:
                        assert False, "got unexpected event from monitor"


class Dialogue(object):
    CLIENT2SERVER = 0
    SERVER2CLIENT = 1

    def __init__(self, client_name, server_name, state_name):
        self.client_name = client_name
        self.server_name = server_name
        self.state_name = state_name
        self.events = []

    def client2server(self, description, event):
        self.events.append((Dialogue.CLIENT2SERVER, description, event))

    def server2client(self, description, event, n=1):
        self.events.extend(
            [(Dialogue.SERVER2CLIENT, description, event) for _i in range(0, n)]
        )

    async def _async_proxy(self, url, q):
        done = asyncio.get_event_loop().create_future()

        async def handle_server(server, client, events):
            async for msg in server:
                print("FROM SERVER:")
                print(msg)
                print()
                ce = from_json(
                    msg, data_unmarshaller=serialization.evaluator_unmarshaller
                )
                event = events.pop(0)
                assert event[0] == Dialogue.SERVER2CLIENT
                assert event[2]["type_"] == ce["type"]
                await client.send(msg)

        async def handle_client(client, _path):
            async with websockets.connect(url + _path) as server:
                events = self.events.copy()
                server_task = asyncio.create_task(handle_server(server, client, events))
                try:
                    async for msg in client:
                        print("FROM CLIENT:")
                        print(msg)
                        print()
                        ce = from_json(
                            msg, data_unmarshaller=serialization.evaluator_unmarshaller
                        )
                        event = events.pop(0)
                        assert event[0] == Dialogue.CLIENT2SERVER
                        assert event[2]["type_"] == ce["type"]
                        await server.send(msg)
                except Exception as e:
                    print(e)
                    raise
                finally:
                    server_task.cancel()

        async with websockets.serve(handle_client, host="localhost", port=0) as s:
            port = s.sockets[0].getsockname()[1]
            asyncio.get_event_loop().run_in_executor(None, lambda: q.put(port))
            asyncio.get_event_loop().run_in_executor(None, lambda: q.put(done))
            await done

    def _proxy(self, url, q):
        asyncio.set_event_loop(asyncio.new_event_loop())
        q.put(asyncio.get_event_loop())
        asyncio.get_event_loop().run_until_complete(self._async_proxy(url, q))

    @contextmanager
    def proxy(self, url):
        q = queue.Queue()
        t = threading.Thread(target=self._proxy, args=(url, q))
        t.start()
        loop = q.get()
        port = q.get()
        done = q.get()
        yield port
        loop.call_soon_threadsafe(done.set_result, None)
        t.join()


def test_ensemble_monitor_communication_given_success(ee_config):
    ensemble = TestEnsemble(iter=1, reals=2, steps=2, jobs=2)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    dialogue = Dialogue(
        client_name="monitor",
        server_name="evaluator",
        state_name="a successfull ensemble",
    )
    dialogue.server2client(
        "initial snapshot", EventDescription(type_=identifiers.EVTYPE_EE_SNAPSHOT)
    )
    dialogue.server2client(
        "snapshot updates",
        EventDescription(type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE),
        n=25,
    )
    dialogue.server2client(
        "snapshot update stopped",
        EventDescription(
            type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
            data={identifiers.STATUS: ENSEMBLE_STATE_STOPPED},
        ),
    )
    dialogue.client2server(
        "monitor done event", EventDescription(type_=identifiers.EVTYPE_EE_USER_DONE)
    )
    dialogue.server2client(
        "evaluator termination event",
        EventDescription(type_=identifiers.EVTYPE_EE_TERMINATED),
    )

    ee.run()
    with dialogue.proxy(ee_config.url) as port:
        with ee_monitor.create("localhost", port) as monitor:
            for event in monitor.track():
                if event["type"] == identifiers.EVTYPE_EE_SNAPSHOT:
                    ensemble.start()
                elif (
                    event.data
                    and event.data.get(identifiers.STATUS) == ENSEMBLE_STATE_STOPPED
                ):
                    monitor.signal_done()

    ensemble.join()


def test_ensemble_monitor_communication_given_failing_job(ee_config):
    ensemble = TestEnsemble(iter=1, reals=2, steps=2, jobs=2)
    ensemble.addFailJob(real=1, step=0, job=1)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    with ee.run() as monitor:
        for event in monitor.track():
            print(event)
            if (
                event.data
                and event.data.get(identifiers.STATUS) == ENSEMBLE_STATE_STOPPED
            ):
                monitor.signal_done()

    ensemble.join()


def test_verify_narratives(ee_config, caplog):
    ensemble = TestEnsemble(iter=1, reals=2, steps=2, jobs=2)
    ensemble.addFailJob(real=1, step=0, job=1)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    ee.run()

    def start_ensemble():
        time.sleep(0.5)  # FIXME
        ensemble.start()

    threading.Thread(target=start_ensemble).run()
    monitor_happy_path_narrative.with_marshaller(
        serialization.evaluator_marshaller
    ).with_unmarshaller(serialization.evaluator_unmarshaller).verify(
        ee_config.client_uri
    )
    ensemble.join()


def test_monitor_stop(evaluator):
    with evaluator.run() as monitor:
        for event in monitor.track():
            snapshot = Snapshot(event.data)
            break
    assert snapshot.get_status() == ENSEMBLE_STATE_STARTED
