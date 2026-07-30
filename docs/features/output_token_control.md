(output-token-control)=

# Output Token Control

vLLM provides several SamplingParams fields that give you fine-grained control
over token generation — from restricting vocabulary to detecting repetitive
output patterns. These fields are available in both the offline Python API and
the online OpenAI-compatible API (via ``extra_body``).

---

## `bad_words`

Prevents specific words or phrases from appearing in the generated text. When
the next generated token would complete a forbidden sequence, that token is
blocked.

### Offline API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B")
sampling_params = SamplingParams(
    temperature=0.8,
    bad_words=["bad phrase", "unwanted"],
)
outputs = llm.generate(["Write a short story."], sampling_params)
```

### Online API

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B",
        "messages": [{"role": "user", "content": "Write a short story."}],
        "extra_body": {
            "bad_words": ["bad phrase", "unwanted"]
        }
    }'
```

!!! note
    ``bad_words`` is applied as a logits processor. Only the final token of
    each forbidden sequence is blocked, which prevents the sequence from
    completing.

---

## `allowed_token_ids`

Restricts generation to a whitelist of token IDs. Only tokens in the
provided list can be generated; all other token logits are masked to
``-inf``.

Useful for:
- Constraining generation to a specific vocabulary (e.g., classification labels)
- Implementing custom grammar constraints without a full structured output backend
- Forcing the model to choose from a predefined set of options

### Offline API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B")

# Allow only tokens for "positive" (token ID 1234) and "negative" (token ID 5678)
sampling_params = SamplingParams(
    temperature=0.0,
    allowed_token_ids=[1234, 5678],
)
outputs = llm.generate(["Classify: great movie!"], sampling_params)
```

### Online API

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B",
        "messages": [{"role": "user", "content": "Classify: great movie!"}],
        "extra_body": {
            "allowed_token_ids": [1234, 5678]
        }
    }'
```

---

## `logprob_token_ids`

Returns log probabilities for a specific set of token IDs without fetching
the full vocabulary. This is more efficient than setting ``logprobs=-1`` when
you only need logprobs for a small number of tokens — for example, when scoring
classification labels or comparing alternatives.

The logprobs for the requested token IDs are returned alongside the sampled
token's logprob.

### Offline API

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen2.5-0.5B")
sampling_params = SamplingParams(
    temperature=0.0,
    logprobs=1,
    logprob_token_ids=[1234, 5678],  # Only request logprobs for these token IDs
)
outputs = llm.generate(["Hello, world!"], sampling_params)
```

### Online API

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B",
        "messages": [{"role": "user", "content": "Hello, world!"}],
        "logprobs": true,
        "extra_body": {
            "logprob_token_ids": [1234, 5678]
        }
    }'
```

---

## `repetition_detection`

Detects and terminates repetitive N-gram output early. LLMs can sometimes
generate repetitive, unhelpful token patterns (e.g., ``abcdabcdabcd...`` or
repeated emoji), stopping only when they hit the maximum output length. This
feature detects such behavior and ends generation at the repetition boundary.

Repetition detection is configured via ``RepetitionDetectionParams``:

| Parameter | Type | Default | Description |
|---|---|---|---|
| ``max_pattern_size`` | ``int`` | ``0`` | Maximum N-gram size to check. Set to ``0`` to disable. |
| ``min_count`` | ``int`` | ``0`` | Minimum number of repetitions of the same pattern before triggering termination. |

### Offline API

```python
from vllm import LLM, SamplingParams, RepetitionDetectionParams

llm = LLM(model="Qwen/Qwen2.5-0.5B")
sampling_params = SamplingParams(
    temperature=0.8,
    max_tokens=512,
    repetition_detection=RepetitionDetectionParams(
        max_pattern_size=5,   # Check for N-grams up to size 5
        min_count=3,          # Terminate after 3 repetitions of the same pattern
    ),
)
outputs = llm.generate(["Tell me a story."], sampling_params)
```

### Online API

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-0.5B",
        "messages": [{"role": "user", "content": "Tell me a story."}],
        "max_tokens": 512,
        "extra_body": {
            "repetition_detection": {
                "max_pattern_size": 5,
                "min_count": 3
            }
        }
    }'
```

!!! note
    ``repetition_detection`` requires both ``max_pattern_size > 0`` and
    ``min_count > 0`` to activate. The feature only terminates generation
    when the same N-gram pattern repeats the configured number of times.
