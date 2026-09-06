import re
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from .subject_agents import SubjectAgent


DEFAULT_AGENTS_DIR = Path(__file__).resolve().parent.parent / "agents"
TOPIC_RE = re.compile(r"^[a-z][a-z0-9_]*$")
REASONING_EFFORTS = {"none", "low", "medium", "high", "xhigh", "max"}
ANSWER_LANGUAGES = {"auto", "he", "en"}
OUTPUT_STYLES = {"standard", "ipad_clean"}
PAPER_STYLES = {"plain", "grid", "ruled"}


class AgentConfigError(ValueError):
    pass


def normalize_alias(value: str) -> str:
    return re.sub(r"[^\w]+", "_", (value or "").strip().lower(), flags=re.UNICODE).strip("_")


@dataclass(frozen=True)
class AgentRegistry:
    agents: dict[str, SubjectAgent]
    aliases: dict[str, str]

    @property
    def topics(self) -> tuple[str, ...]:
        return tuple(self.agents)

    def normalize_topic(self, value: str) -> str:
        normalized = normalize_alias(value)
        normalized = self.aliases.get(normalized, normalized)
        return normalized if normalized in self.agents else "general"

    def get(self, topic: str) -> SubjectAgent:
        return self.agents.get(self.normalize_topic(topic), self.agents["general"])


def _string_list(value, field_name: str, source: Path) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise AgentConfigError(f"{source}: {field_name} must be a list of non-empty strings")
    return tuple(item.strip().lower() for item in value)


def load_agent_definition(path: Path) -> tuple[SubjectAgent, bool]:
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise AgentConfigError(f"Invalid TOML in {path}: {exc}") from exc

    topic = data.get("topic")
    display_name = data.get("display_name")
    instructions = data.get("instructions")
    enabled = data.get("enabled", True)
    provider = data.get("provider", "openai")
    model = data.get("model")
    temperature = data.get("temperature", 0.2)
    reasoning_effort = data.get("reasoning_effort", "medium")
    answer_language = data.get("answer_language", "auto")
    output_style = data.get("output_style", "standard")
    paper_style = data.get("paper_style", "plain")
    if not isinstance(topic, str) or not TOPIC_RE.fullmatch(topic):
        raise AgentConfigError(f"{path}: topic must use lowercase snake_case")
    if not isinstance(display_name, str) or not display_name.strip():
        raise AgentConfigError(f"{path}: display_name must be a non-empty string")
    if not isinstance(instructions, str) or not instructions.strip():
        raise AgentConfigError(f"{path}: instructions must be a non-empty string")
    if not isinstance(enabled, bool):
        raise AgentConfigError(f"{path}: enabled must be true or false")
    if not isinstance(provider, str) or not TOPIC_RE.fullmatch(provider):
        raise AgentConfigError(f"{path}: provider must use lowercase snake_case")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise AgentConfigError(f"{path}: model must be a non-empty string when provided")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        raise AgentConfigError(f"{path}: temperature must be a number between 0 and 2")
    if not 0 <= float(temperature) <= 2:
        raise AgentConfigError(f"{path}: temperature must be between 0 and 2")
    if not isinstance(reasoning_effort, str) or reasoning_effort not in REASONING_EFFORTS:
        raise AgentConfigError(
            f"{path}: reasoning_effort must be one of {sorted(REASONING_EFFORTS)}"
        )
    if not isinstance(answer_language, str) or answer_language not in ANSWER_LANGUAGES:
        raise AgentConfigError(
            f"{path}: answer_language must be one of {sorted(ANSWER_LANGUAGES)}"
        )
    if not isinstance(output_style, str) or output_style not in OUTPUT_STYLES:
        raise AgentConfigError(f"{path}: output_style must be one of {sorted(OUTPUT_STYLES)}")
    if not isinstance(paper_style, str) or paper_style not in PAPER_STYLES:
        raise AgentConfigError(f"{path}: paper_style must be one of {sorted(PAPER_STYLES)}")

    agent = SubjectAgent(
        topic=topic,
        display_name=display_name.strip(),
        instructions=instructions.strip(),
        keywords=_string_list(data.get("keywords"), "keywords", path),
        aliases=_string_list(data.get("aliases"), "aliases", path),
        provider=provider,
        model=model.strip() if model is not None else None,
        temperature=float(temperature),
        reasoning_effort=reasoning_effort,
        answer_language=answer_language,
        output_style=output_style,
        paper_style=paper_style,
    )
    return agent, enabled


def load_agent_registry(agents_dir: Path = DEFAULT_AGENTS_DIR) -> AgentRegistry:
    if not agents_dir.is_dir():
        raise AgentConfigError(f"Agent definitions directory does not exist: {agents_dir}")

    agents: dict[str, SubjectAgent] = {}
    aliases: dict[str, str] = {}
    paths = sorted(agents_dir.glob("*.toml"))
    if not paths:
        raise AgentConfigError(f"No agent TOML files found in {agents_dir}")

    for path in paths:
        agent, enabled = load_agent_definition(path)
        if not enabled:
            continue
        if agent.topic in agents:
            raise AgentConfigError(f"Duplicate agent topic: {agent.topic}")
        agents[agent.topic] = agent
        for alias in agent.aliases:
            normalized_alias = normalize_alias(alias)
            if not normalized_alias:
                raise AgentConfigError(f"Agent alias cannot normalize to an empty value: {alias!r}")
            if normalized_alias in aliases and aliases[normalized_alias] != agent.topic:
                raise AgentConfigError(f"Duplicate agent alias: {alias}")
            aliases[normalized_alias] = agent.topic

    if "general" not in agents:
        raise AgentConfigError("An enabled general.toml fallback agent is required")
    return AgentRegistry(agents=agents, aliases=aliases)


@lru_cache(maxsize=1)
def get_agent_registry() -> AgentRegistry:
    return load_agent_registry(DEFAULT_AGENTS_DIR)
