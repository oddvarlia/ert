from contextlib import contextmanager

import websockets
import pytest
import asyncio
import threading
import queue
from unittest.mock import Mock

from pytest_asyncio.plugin import unused_tcp_port

from ert_shared.ensemble_evaluator.entity.ensemble_base import _Ensemble
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_STARTED,
    JOB_STATE_FAILURE,
    JOB_STATE_FINISHED,
    JOB_STATE_RUNNING,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_FAILED,
)
from cloudevents.http import to_json
from cloudevents.http.event import CloudEvent
from ert_shared.ensemble_evaluator.evaluator import (
    EnsembleEvaluator,
    ee_monitor,
)
from ert_shared.ensemble_evaluator.config import EvaluatorServerConfig
from ert_shared.ensemble_evaluator.entity.ensemble import (
    create_ensemble_builder,
    create_realization_builder,
    create_step_builder,
    create_legacy_job_builder,
    _Realization,
    _Stage,
    _Step,
    _BaseJob,
)
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers
from ert_shared.ensemble_evaluator.entity.snapshot import Snapshot
from tests.narrative import Consumer, Provider, EventDescription


@pytest.fixture
def ee_config(unused_tcp_port):
    return EvaluatorServerConfig(unused_tcp_port)


@pytest.fixture
def evaluator(ee_config):
    ensemble = (
        create_ensemble_builder()
        .add_realization(
            real=create_realization_builder()
            .active(True)
            .set_iens(0)
            .add_step(
                step=create_step_builder()
                .set_id("0")
                .set_name("cats")
                .add_job(
                    job=create_legacy_job_builder()
                    .set_id(0)
                    .set_name("cat")
                    .set_ext_job(Mock())
                )
                .add_job(
                    job=create_legacy_job_builder()
                    .set_id(1)
                    .set_name("cat2")
                    .set_ext_job(Mock())
                )
                .set_dummy_io()
            )
        )
        .set_ensemble_size(2)
        .build()
    )
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    yield ee
    ee.stop()


class Client:
    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.stop()

    def __init__(self, host, port, path):
        self.host = host
        self.port = port
        self.path = path
        self.loop = asyncio.new_event_loop()
        self.q = asyncio.Queue(loop=self.loop)
        self.thread = threading.Thread(
            name="test_websocket_client", target=self._run, args=(self.loop,)
        )

    def _run(self, loop):
        asyncio.set_event_loop(loop)
        uri = f"ws://{self.host}:{self.port}{self.path}"

        async def send_loop(q):
            async with websockets.connect(uri) as websocket:
                while True:
                    msg = await q.get()
                    if msg == "stop":
                        return
                    await websocket.send(msg)

        loop.run_until_complete(send_loop(self.q))

    def send(self, msg):
        self.loop.call_soon_threadsafe(self.q.put_nowait, msg)

    def stop(self):
        self.loop.call_soon_threadsafe(self.q.put_nowait, "stop")
        self.thread.join()
        self.loop.close()


def send_dispatch_event(client, event_type, source, event_id, data):
    event1 = CloudEvent({"type": event_type, "source": source, "id": event_id}, data)
    client.send(to_json(event1))


def test_dispatchers_can_connect_and_monitor_can_shut_down_evaluator(evaluator):
    with evaluator.run() as monitor:
        events = monitor.track()

        host = evaluator._config.host
        port = evaluator._config.port

        # first snapshot before any event occurs
        snapshot_event = next(events)
        snapshot = Snapshot(snapshot_event.data)
        assert snapshot.get_status() == ENSEMBLE_STATE_STARTED
        # two dispatchers connect
        with Client(host, port, "/dispatch") as dispatch1, Client(
            host, port, "/dispatch"
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


class TestEnsemble(_Ensemble):
    def __init__(self, iter, reals, stages, steps, jobs):
        self.iter = iter
        self.reals = reals
        self.stages = stages
        self.steps = steps
        self.jobs = jobs
        self.fail_jobs = []

        the_reals = [
            _Realization(
                real_no,
                stages=[
                    _Stage(
                        id_=stage_no,
                        steps=[
                            _Step(
                                id_=step_no,
                                inputs=[],
                                outputs=[],
                                jobs=[
                                    _BaseJob(
                                        id_=job_no, name=f"job-{job_no}", step_source=""
                                    )
                                    for job_no in range(0, jobs)
                                ],
                                name=f"step-{step_no}",
                                ee_url="",
                                source="",
                            )
                            for step_no in range(0, steps)
                        ],
                        status="unknown",
                        name=f"stage-{stage_no}",
                    )
                    for stage_no in range(0, stages)
                ],
                active=True,
            )
            for real_no in range(0, reals)
        ]
        super().__init__(the_reals, {})

    def _evaluate(self, host, port, ee_id):
        event_id = 0
        with Client(host, port, "/dispatch") as dispatch:
            send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_ENSEMBLE_STARTED,
                f"/ert/ee/{ee_id}",
                f"event-{event_id}",
                None,
            )
            event_id = event_id + 1
            for real in range(0, self.reals):
                for stage in range(0, self.stages):
                    for step in range(0, self.stages):
                        job_failed = False
                        send_dispatch_event(
                            dispatch,
                            identifiers.EVTYPE_FM_STEP_START,
                            f"/ert/ee/{ee_id}/real/{real}/stage/{stage}/step/{step}",
                            f"event-{event_id}",
                            None,
                        )
                        event_id = event_id + 1
                        for job in range(0, self.jobs):
                            send_dispatch_event(
                                dispatch,
                                identifiers.EVTYPE_FM_JOB_RUNNING,
                                f"/ert/ee/{ee_id}/real/{real}/stage/{stage}/step/{step}/job/{job}",
                                f"event-{event_id}",
                                {"current_memory_usage": 1000},
                            )
                            event_id = event_id + 1
                            if self._shouldFailJob(real, stage, step, job):
                                send_dispatch_event(
                                    dispatch,
                                    identifiers.EVTYPE_FM_JOB_FAILURE,
                                    f"/ert/ee/{ee_id}/real/{real}/stage/{stage}/step/{step}/job/{job}",
                                    f"event-{event_id}",
                                    {},
                                )
                                event_id = event_id + 1
                                job_failed = True
                                break
                            else:
                                send_dispatch_event(
                                    dispatch,
                                    identifiers.EVTYPE_FM_JOB_SUCCESS,
                                    f"/ert/ee/{ee_id}/real/{real}/stage/{stage}/step/{step}/job/{job}",
                                    f"event-{event_id}",
                                    {"current_memory_usage": 1000},
                                )
                                event_id = event_id + 1
                        if job_failed:
                            send_dispatch_event(
                                dispatch,
                                identifiers.EVTYPE_FM_STEP_FAILURE,
                                f"/ert/ee/{ee_id}/real/{real}/stage/{stage}/step/{step}/job/{job}",
                                f"event-{event_id}",
                                {},
                            )
                            event_id = event_id + 1
                        else:
                            send_dispatch_event(
                                dispatch,
                                identifiers.EVTYPE_FM_STEP_SUCCESS,
                                f"/ert/ee/{ee_id}/real/{real}/stage/{stage}/step/{step}/job/{job}",
                                f"event-{event_id}",
                                {},
                            )
                            event_id = event_id + 1

            send_dispatch_event(
                dispatch,
                identifiers.EVTYPE_ENSEMBLE_STOPPED,
                f"/ert/ee/{ee_id}",
                f"event-{event_id}",
                None,
            )

    def join(self):
        self._eval_thread.join()

    def evaluate(self, config, ee_id):
        self._eval_thread = threading.Thread(
            target=self._evaluate,
            args=(config.host, config.port, ee_id),
        )
        self._eval_thread.start()

    def _shouldFailJob(self, real, stage, step, job):
        return (real, stage, step, job) in self.fail_jobs

    def addFailJob(self, real, stage, step, job):
        self.fail_jobs.append((real, stage, step, job))


class Dialogue(object):
    def __init__(self, client_name, server_name, state_name):
        self.client_name = client_name
        self.server_name = server_name
        self.state_name = state_name
        self.events = []

    def client2server(self, description, event):
        self.events.append((description, event))

    def server2client(self, description, event, n=1):
        self.events.append((description, event))

    async def _async_proxy(self, url, q):
        done = asyncio.get_event_loop().create_future()

        async def handle_server(server, client):
            async for msg in server:
                print("FROM SERVER:")
                print(msg)
                print()
                await client.send(msg)

        async def handle_client(client, _path):
            async with websockets.connect(url + _path) as server:
                server_task = asyncio.create_task(handle_server(server, client))
                try:
                    async for msg in client:
                        print("FROM CLIENT:")
                        print(msg)
                        print()
                        await server.send(msg)
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
    ensemble = TestEnsemble(iter=1, reals=2, stages=2, steps=2, jobs=2)
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
        n=26,
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
                if (
                    event.data
                    and event.data.get(identifiers.STATUS) == ENSEMBLE_STATE_STOPPED
                ):
                    monitor.signal_done()

    ensemble.join()


def test_ensemble_monitor_communication_given_failing_job(ee_config):
    ensemble = TestEnsemble(iter=1, reals=2, stages=2, steps=2, jobs=2)
    ensemble.addFailJob(real=1, stage=1, step=0, job=1)
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


def test_monitor_stop(evaluator):
    with evaluator.run() as monitor:
        for event in monitor.track():
            snapshot = Snapshot(event.data)
            break
    assert snapshot.get_status() == ENSEMBLE_STATE_STARTED
