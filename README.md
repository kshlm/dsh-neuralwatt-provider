# dsh-llm-neuralwatt

Neuralwatt LLM provider plugin for [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness).

It dynamically populates models from Neuralwatt's `GET /v1/models` endpoint and sets
each model's capabilities from the endpoint metadata: input modalities (text or image),
context window, max output tokens, and reasoning efforts. It speaks Neuralwatt's
OpenAI-compatible `/chat/completions` API.

## Install

    dsh plugin --profile <profile> add dsh-llm-neuralwatt

The package declares a `dsh.bundle` layer (`cordis.patch.yml`) that inserts the
`llm-neuralwatt` plugin into the profile automatically, so no manual patch row is
needed. To pass configuration, restate the row with a `config` block in your
profile's `cordis.patch.yml`:

    - id: llm-neuralwatt
      name: dsh-llm-neuralwatt
      config:
        baseURL: https://api.neuralwatt.com/v1

Then remove any hand-written `neuralwatt` block under `llm-pi-ai.providers` in your
profile's `cordis.patch.yml` (the two routes would otherwise collide).

## Configuration

Configure under the plugin (`llm-neuralwatt`) in `cordis.patch.yml`:

    - id: llm-neuralwatt
      name: dsh-llm-neuralwatt
      config:
        baseURL: https://api.neuralwatt.com/v1   # optional; override here
        apiKeyEnv: NEURALWATT_API_KEY            # optional; env var *name*
        route: neuralwatt                        # optional
        displayName: Neuralwatt                  # optional

| Field | Default | Env fallback | Meaning |
|---|---|---|---|
| `route` | `neuralwatt` | None | Provider route key / selector id. |
| `displayName` | `Neuralwatt` | None | Provider display name. |
| `baseURL` | `https://api.neuralwatt.com/v1` | `NEURALWATT_BASE_URL` | Endpoint root (overridable). |
| `apiKeyEnv` | `NEURALWATT_API_KEY` | None | Env-var *name* holding the API key. |

Precedence: `config` > env > built-in default. Set the API key by exporting
`NEURALWATT_API_KEY` (or whatever `apiKeyEnv` names) or storing it via the credentials
service.

## Behavior

- Models are fetched once at load (boot snapshot) from `GET {baseURL}/models`.
- All models are exposed; only entries marked `deprecated` are skipped.
- Vision models advertise `image` input; text-only models advertise `text`.
- Reasoning efforts are mapped from Neuralwatt's `accepted_efforts`, folding
  `effort_aliases` onto canonical ids and normalizing `none` to `off`.
- Context and max-output capacities come from the endpoint; when it omits them the
  defaults are `262144` context and `32768` output tokens.
- Every request carries `attributionHeaders()` and forwards the caller's abort signal.

## License

[MIT](LICENSE)
