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
from tests.narrative import EventDescription, Consumer, Provider, _Narrative, _Response, _Request
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

    def __init__(self, narrative: _Narrative):
        self.narrative = narrative
        self.currentInteraction = narrative.interactions.pop()
        self.error = None

    async def verify_event(self, ce, type):
        if not self.currentInteraction.events:
            self.currentInteraction = self.narrative.interactions.pop()
        event = self.currentInteraction.events[0]
        if not event.repeating:
            assert isinstance(self.currentInteraction, type)
            assert event.matches(ce)
            self.currentInteraction.events.pop(0)
        else:
            # repeating

            if isinstance(self.narrative.interactions[-1], type) and self.narrative.interactions[-1].events[0].matches(ce):
                self.currentInteraction = self.narrative.interactions.pop()

            if isinstance(self.currentInteraction, type):
                if not event.matches(ce):
                    self.currentInteraction = self.narrative.interactions.pop()
                    event = self.currentInteraction.events[0]
                    assert event.matches(ce)
            else:
                self.currentInteraction = self.narrative.interactions.pop()
                event = self.currentInteraction.events[0]
                assert event.matches(ce)
                if not event.repeating:
                    self.currentInteraction.events.pop(0)


    async def _async_proxy(self, url, q):
        self.done = asyncio.get_event_loop().create_future()

        async def handle_server(server, client):
            try:
                async for msg in server:
                    print("FROM SERVER:")
                    print(msg)
                    print()
                    ce = from_json(
                        msg, data_unmarshaller=serialization.evaluator_unmarshaller
                    )
                    try:
                        await self.verify_event(ce, _Response)
                    except AssertionError as e:
                        self.done.set_result(e)
                        await client.close()
                        raise e

                    await client.send(msg)
            except Exception as e:
                raise e

        async def handle_client(client, _path):
            try:
                if _path == "/client":
                    async with websockets.connect(url + _path) as server:
                        server_task = asyncio.create_task(handle_server(server, client))

                        async for msg in client:
                            print("FROM CLIENT:")
                            print(msg)
                            print()
                            ce = from_json(
                                msg, data_unmarshaller=serialization.evaluator_unmarshaller
                            )
                            try:
                                await self.verify_event(ce, _Request)
                            except Exception as e:
                                self.done.set_result(e)
                                await server.close()
                                raise e

                            await server.send(msg)
            except Exception as e:
                print(e)
                raise
            finally:
                print("finally")

        async with websockets.serve(handle_client, host="localhost", port=0) as s:
            port = s.sockets[0].getsockname()[1]
            asyncio.get_event_loop().run_in_executor(None, lambda: q.put(port))
            asyncio.get_event_loop().run_in_executor(None, lambda: q.put(self.done))
            error = await self.done
            q.put(error)

    def _proxy(self, url, q):
        asyncio.set_event_loop(asyncio.new_event_loop())
        q.put(asyncio.get_event_loop())
        asyncio.get_event_loop().run_until_complete(self._async_proxy(url, q))
        print("her")


    @contextmanager
    def proxy(self, url):
        q = queue.Queue()
        t = threading.Thread(target=self._proxy, args=(url, q))
        t.start()
        loop = q.get()
        port = q.get()
        done = q.get()
        yield port
        if not done.done():
            loop.call_soon_threadsafe(done.set_result, None)
        t.join()
        error = q.get()
        if error:
            raise AssertionError('Sum thing wong') from error


def test_ensemble_monitor_communication_given_success(ee_config, unused_tcp_port):
    ensemble = TestEnsemble(iter=1, reals=2, steps=2, jobs=2)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    narrative = (
        Consumer("Monitor")
            .forms_narrative_with(Provider("Evaluator"))
            .given("Successful Ensemble with 2 reals, with 2 steps each, with 2 jobs each")
            .responds_with("Snapshot")
            .cloudevents_in_order([EventDescription(type_=identifiers.EVTYPE_EE_SNAPSHOT, source="*")])
            .responds_with("Some amount of Snapshot updates")
            .cloudevents_repeating(EventDescription(type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE, source="*"))
            .receives("Monitor done")
            .cloudevents_in_order([EventDescription(type_=identifiers.EVTYPE_EE_USER_DONE, source="*")])
            # .receives("Kødd")
            # .cloudevents_in_order([EventDescription(type_=identifiers.EVTYPE_EE_USER_DONE, source="Frode")])
            .responds_with("Termination")
            .cloudevents_in_order([EventDescription(type_=identifiers.EVTYPE_EE_TERMINATED, source="*")])
            .on_uri(f"ws://localhost:{unused_tcp_port}")
    )

    ee.run()
    with Dialogue(narrative).proxy(ee_config.url) as port:
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


def test_ensemble_monitor_communication_given_failing_job(ee_config, unused_tcp_port):
    narrative = (
        Consumer("Monitor")
            .forms_narrative_with(Provider("Evaluator"))
            .given("Ensemble with 2 reals, with 2 steps each, with 2 jobs each, job 1 in real 1 fails")
            .responds_with("Snapshot")
            .cloudevents_in_order([EventDescription(type_=identifiers.EVTYPE_EE_SNAPSHOT, source="*")])
            .responds_with("Some amount of Snapshot updates")
            .cloudevents_repeating(EventDescription(type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE, source="*"))
            # .responds_with("One update of failing job")
            # .cloudevents_in_order([EventDescription(type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE, source="*")])
            # .responds_with("Some amount of Snapshot updates")
            # .cloudevents_repeating(EventDescription(type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE, source="*"))
            .receives("Monitor done")
            .cloudevents_in_order([EventDescription(type_=identifiers.EVTYPE_EE_USER_DONE, source="*")])
            # .receives("Kødd")
            # .cloudevents_in_order([EventDescription(type_=identifiers.EVTYPE_EE_USER_DONE, source="Frode")])
            .responds_with("Termination")
            .cloudevents_in_order([EventDescription(type_=identifiers.EVTYPE_EE_TERMINATED, source="*")])
            .on_uri(f"ws://localhost:{unused_tcp_port}")
    )
    ensemble = TestEnsemble(iter=1, reals=2, steps=2, jobs=2)
    ensemble.addFailJob(real=1, step=0, job=1)
    ee = EnsembleEvaluator(
        ensemble,
        ee_config,
        0,
        ee_id="ee-0",
    )
    with ee.run() as m:
        pass
    with Dialogue(narrative).proxy(ee_config.url) as port:
        with ee_monitor.create("localhost", port) as monitor:
            for event in monitor.track():
                print(f"monitor received: {event}")
                if event["type"] == identifiers.EVTYPE_EE_SNAPSHOT:
                    ensemble.start()
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
