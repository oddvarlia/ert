from ert_shared.status.entity.state import ENSEMBLE_STATE_STOPPED
from tests.narrative import Consumer, EventDescription, Provider
import ert_shared.ensemble_evaluator.entity.identifiers as identifiers


dispatchers_failing_job_narrative = (
    Consumer("Dispatch")
    .forms_narrative_with(Provider("Ensemble Evaluator"))
    .given("small ensemble")
    .receives("a job eventually fails")
    .cloudevents_in_order(
        [
            EventDescription(
                type_=identifiers.EVTYPE_FM_JOB_RUNNING,
                source="/ert/ee/0/real/0/stage/0/step/0/job/0",
                data={identifiers.CURRENT_MEMORY_USAGE: 1000},
            ),
            EventDescription(
                type_=identifiers.EVTYPE_FM_JOB_RUNNING,
                source="/ert/ee/0/real/1/stage/0/step/0/job/0",
                data={identifiers.CURRENT_MEMORY_USAGE: 2000},
            ),
            EventDescription(
                type_=identifiers.EVTYPE_FM_JOB_SUCCESS,
                source="/ert/ee/0/real/0/stage/0/step/0/job/0",
                data={identifiers.CURRENT_MEMORY_USAGE: 2000},
            ),
            EventDescription(
                type_=identifiers.EVTYPE_FM_JOB_FAILURE,
                source="/ert/ee/0/real/1/stage/0/step/0/job/0",
                data={identifiers.ERROR_MSG: "error"},
            ),
        ]
    )
)

monitor_happy_path_narrative = (
    Consumer("Monitor")
    .forms_narrative_with(
        Provider("Ensemble Evaluator"),
    )
    .given("a successful one-member one-step one-job ensemble")
    .responds_with("nominal event flow")
    .cloudevents_in_order(
        [
            EventDescription(type_=identifiers.EVTYPE_EE_SNAPSHOT, source="/provider"),
            EventDescription(
                type_=identifiers.EVTYPE_EE_SNAPSHOT_UPDATE,
                source="/provider",
                data={identifiers.STATUS: ENSEMBLE_STATE_STOPPED},
            ),
        ]
    )
    .receives("done")
    .cloudevents_in_order(
        [
            EventDescription(
                type_=identifiers.EVTYPE_EE_USER_DONE, source="/ert/monitor/*"
            ),
        ]
    )
    .responds_with("termination")
    .cloudevents_in_order(
        [
            EventDescription(
                type_=identifiers.EVTYPE_EE_TERMINATED, source="/provider"
            ),
        ]
    )
)
