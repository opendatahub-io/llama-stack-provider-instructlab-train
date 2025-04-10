from llama_stack.providers.datatypes import (
    ProviderSpec,
    Api,
    AdapterSpec,
    remote_provider_spec,
)


def get_provider_spec() -> ProviderSpec:
    return remote_provider_spec(
        api=Api.post_training,
        adapter=AdapterSpec(
            adapter_type="instructlab_train",
            pip_packages=["instructlab-training"],
            config_class="config.InstructLabTrainPostTrainingConfig",
            module="kft_adapter",
        ),
    )
