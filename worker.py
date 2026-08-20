import random
import os

from vastai import Worker, WorkerConfig, HandlerConfig, LogActionConfig, BenchmarkConfig

# Ollama OpenAI-compatible server configuration
MODEL_SERVER_URL           = 'http://127.0.0.1'
MODEL_SERVER_PORT          = int(os.environ.get("OLLAMA_PORT", 11434))
MODEL_LOG_FILE             = os.environ.get("MODEL_LOG", "/var/log/ollama.log")
MODEL_HEALTHCHECK_ENDPOINT = "/"

# Ollama-specific log messages
MODEL_LOAD_LOG_MSG = [
    "Listening on",
]

MODEL_ERROR_LOG_MSGS = [
    "panic:",
    "level=FATAL",
    "Traceback (most recent call last):",
]

MODEL_INFO_LOG_MSGS = [
    "downloading",
]

WORD_LIST = (
    "the quick brown fox jumps over lazy dog time work code data model "
    "system process function value error handle request response server "
    "client user file text word list array object number string boolean "
    "network memory thread queue stack loop condition variable module "
    "package import export return call define class method field type"
).split()


def request_parser(request):
    return request


def completions_benchmark_generator() -> dict:
    prompt = " ".join(random.choices(WORD_LIST, k=int(250)))
    model = os.environ.get("MODEL_NAME")
    if not model:
        raise ValueError("MODEL_NAME environment variable not set")

    benchmark_data = {
        "model": model,
        "prompt": prompt,
        "temperature": 0.7,
        "max_tokens": 500,
    }

    return benchmark_data


worker_config = WorkerConfig(
    model_server_url=MODEL_SERVER_URL,
    model_server_port=MODEL_SERVER_PORT,
    model_log_file=MODEL_LOG_FILE,
    model_healthcheck_url=MODEL_HEALTHCHECK_ENDPOINT,
    handlers=[
        HandlerConfig(
            route="/v1/completions",
            workload_calculator=lambda data: data.get("max_tokens", 0),
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
            benchmark_config=BenchmarkConfig(
                generator=completions_benchmark_generator,
                concurrency=4,
                runs=3
            )
        ),
        HandlerConfig(
            route="/v1/chat/completions",
            workload_calculator=lambda data: data.get("max_tokens", 0),
            allow_parallel_requests=True,
            request_parser=request_parser,
            max_queue_time=600.0,
        )
    ],
    log_action_config=LogActionConfig(
        on_load=MODEL_LOAD_LOG_MSG,
        on_error=MODEL_ERROR_LOG_MSGS,
        on_info=MODEL_INFO_LOG_MSGS
    )
)

Worker(worker_config).run()
