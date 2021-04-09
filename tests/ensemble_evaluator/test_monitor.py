from ert_shared.ensemble_evaluator.monitor import _Monitor
from tests.narrative import Consumer, EventDescription, Provider
import ert_shared.ensemble_evaluator.entity.identifiers as ids
from ert_shared.status.entity.state import (
    ENSEMBLE_STATE_CANCELLED,
    ENSEMBLE_STATE_STOPPED,
    ENSEMBLE_STATE_FAILED,
)

import websockets
from cloudevents.http import CloudEvent, to_json, from_json
import pytest


def test_consume(unused_tcp_port, caplog):
    import logging

    caplog.set_level(logging.DEBUG, logger="ert_shared.ensemble_evaluator")
    narrative = (
        Consumer("Monitor")
        .forms_narrative_with(
            Provider("Ensemble Evaluator"),
            uri=f"ws://localhost:{unused_tcp_port}/client",
        )
        .given("a successful one-member one-step one-job ensemble")
        .responds_with("nominal event flow")
        .cloudevents_in_order(
            [
                EventDescription(type_=ids.EVTYPE_EE_SNAPSHOT, source="/provider"),
                EventDescription(
                    type_=ids.EVTYPE_EE_SNAPSHOT_UPDATE,
                    source="/provider",
                    data={ids.STATUS: ENSEMBLE_STATE_STOPPED},
                ),
            ]
        )
        .receives("done")
        .cloudevents_in_order(
            [
                EventDescription(
                    type_=ids.EVTYPE_EE_USER_DONE, source="/ert/monitor/*"
                ),
            ]
        )
        .responds_with("termination")
        .cloudevents_in_order(
            [
                EventDescription(type_=ids.EVTYPE_EE_TERMINATED, source="/provider"),
            ]
        )
    )

    expected_events_types = iter(
        [
            ids.EVTYPE_EE_SNAPSHOT,
            ids.EVTYPE_EE_SNAPSHOT_UPDATE,
            ids.EVTYPE_EE_TERMINATED,
        ]
    )
    with narrative:
        with _Monitor(narrative.hostname, narrative.port) as monitor:
            for event in monitor.track():
                assert event["type"] == next(expected_events_types)
                if event.data and event.data.get(ids.STATUS) == ENSEMBLE_STATE_STOPPED:
                    monitor.signal_done()
