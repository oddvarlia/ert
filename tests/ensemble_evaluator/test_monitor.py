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


@pytest.mark.asyncio
async def test_consume(unused_tcp_port):
    narrative = Consumer("Monitor").forms_narrative_with(Provider("Ensemble Evaluator"))
    narrative.uri = f"ws://localhost:{unused_tcp_port}/client"
    narrative.given("a successful one-member one-step one-job ensemble").responds_with(
        "nominal event flow"
    ).cloudevents_in_order(
        [
            EventDescription(type_=ids.EVTYPE_EE_SNAPSHOT, source="/provider"),
            EventDescription(
                type_=ids.EVTYPE_EE_SNAPSHOT_UPDATE,
                source="/provider",
                data={ids.STATUS: ENSEMBLE_STATE_STOPPED},
            ),
        ]
    ).receives(
        "done"
    ).cloudevents_in_order(
        [
            EventDescription(type_=ids.EVTYPE_EE_USER_DONE, source="/consumer"),
        ]
    ).responds_with(
        "termination"
    ).cloudevents_in_order(
        [
            EventDescription(type_=ids.EVTYPE_EE_TERMINATED, source="/provider"),
        ]
    )
    # needs sync version of narrative
    async with narrative:
        print("narrative", narrative)
        return
        with _Monitor(narrative.hostname, narrative.port) as monitor:
            terminated = True
            for event in monitor.track(conn_check=False):
                if event.data.get(ids.STATUS) == ENSEMBLE_STATE_STOPPED:
                    monitor.signal_done()
                    terminated = True
                if terminated:
                    assert event["type"] == ids.EVTYPE_EE_TERMINATED
