from typing import Any, Dict, Optional

from llama_stack.apis.post_training import (
    AlgorithmConfig,
    DPOAlignmentConfig,
    ListPostTrainingJobsResponse,
    PostTrainingJob,
    PostTrainingJobArtifactsResponse,
    PostTrainingJobStatusResponse,
    LoraFinetuningConfig,
    TrainingConfig,
    JobStatus,
)
from .config import (
    InstructLabTrainPostTrainingConfig,
)
from llama_stack.log import get_logger
from llama_stack.providers.utils.scheduler import Scheduler
from llama_stack.providers.utils.scheduler import JobStatus as SchedulerJobStatus

_JOB_TYPE_SUPERVISED_FINE_TUNE = "supervised-fine-tune"

logger = get_logger(name=__name__, category="post_training")


class InstructLabTrainPostTrainingImpl:
    def __init__(
        self,
        config: InstructLabTrainPostTrainingConfig,
        #  datasetio_api: DatasetIO,
        #  datasets: Datasets,
    ) -> None:
        self.config = config
        # self.datasetio_api = datasetio_api
        # self.datasets_api = datasets
        self._scheduler = Scheduler()

        self.checkpoints_dict = {}

    async def shutdown(self):
        pass

    async def supervised_fine_tune(
        self,
        job_uuid: str,
        training_config: TrainingConfig,  # this is basically unused
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
        model: str,
        checkpoint_dir: Optional[str],
        algorithm_config: Optional[AlgorithmConfig],
    ) -> PostTrainingJob:
        if isinstance(algorithm_config, LoraFinetuningConfig):
            raise NotImplementedError()

        post_training_job = PostTrainingJob(job_uuid=job_uuid)

        async def handler(
            on_log_message_cb,
            on_status_change_cb,
            on_artifact_collected_cb,
            # gpu_identifier: str,
            # cpu_per_worker: str,
            # memory_per_worker: str,
            # tolerations: list,
            # node_selectors: dict,
            # pytorchjob_output_yaml: dsl.Output[dsl.Artifact],
            # model_pvc_name: str,
            # input_pvc_name: str,
            # output_pvc_name: str,
            # name_suffix: str,
            # phase_num: int,
            # base_image: str,
            # nproc_per_node: int = self.config.nproc_per_node,
            # nnodes: int = self.config.nnodes,
            # num_epochs: int = training_config.n_epochs,
            # effective_batch_size: int = training_config.effective_batch_size,
            # learning_rate: float = training_config.learning_rate,
            # num_warmup_steps: int = self.config.num_warmup_steps,
            # save_samples: int = self.config.save_samples,
            # max_batch_len: int = self.config.max_batch_len,
            # seed: int = self.config.seed,
            # job_timeout: int = 86400,
            # delete_after_done: bool = False,
            # keep_last_checkpoint_only: bool = False,
        ):
            # TODO: Implement the training job

            on_log_message_cb("InstructLab finetuning completed")

        self._scheduler.schedule(_JOB_TYPE_SUPERVISED_FINE_TUNE, job_uuid, handler)

        return post_training_job

    async def preference_optimize(
        self,
        job_uuid: str,
        finetuned_model: str,
        algorithm_config: DPOAlignmentConfig,
        training_config: TrainingConfig,
        hyperparam_search_config: Dict[str, Any],
        logger_config: Dict[str, Any],
    ) -> PostTrainingJob:
        pass

    async def get_training_jobs(self) -> ListPostTrainingJobsResponse:
        return ListPostTrainingJobsResponse(
            data=map(
                lambda job: PostTrainingJob(job_uuid=job.id), self._scheduler.get_jobs()
            )
        )

    async def get_training_job_status(
        self, job_uuid: str
    ) -> Optional[PostTrainingJobStatusResponse]:
        job = self._scheduler.get_job(job_uuid)

        match job.status:
            # TODO: Add support for other statuses to API
            case SchedulerJobStatus.new | SchedulerJobStatus.scheduled:
                status = JobStatus.scheduled
            case SchedulerJobStatus.running:
                status = JobStatus.in_progress
            case SchedulerJobStatus.completed:
                status = JobStatus.completed
            case SchedulerJobStatus.failed:
                status = JobStatus.failed
            case _:
                raise NotImplementedError()

        return PostTrainingJobStatusResponse(
            job_uuid=job_uuid,
            status=status,
            scheduled_at=job.scheduled_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            checkpoints=[],
        )

    async def cancel_training_job(self, job_uuid: str) -> None:
        raise NotImplementedError("Job cancel is not implemented yet")

    async def get_training_job_artifacts(
        self, job_uuid: str
    ) -> Optional[PostTrainingJobArtifactsResponse]:
        if job_uuid in self.checkpoints_dict:
            checkpoints = self.checkpoints_dict.get(job_uuid, [])
            return PostTrainingJobArtifactsResponse(
                job_uuid=job_uuid, checkpoints=checkpoints
            )
        return None


async def get_adapter_impl(config: InstructLabTrainPostTrainingConfig, _deps):
    impl = InstructLabTrainPostTrainingImpl(
        config,
        #  _deps[Api.datasetio],
        #  _deps[Api.datasets]
    )
    return impl
