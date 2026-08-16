"""Seed knowledge base for grounding recommendations.

READ THIS BEFORE USING THE CITATIONS.

These passages are paraphrases written from memory of well-known work in music
psychology and EEG. They are a *development seed* -- enough to exercise the
retrieval path -- not a verified bibliography. Every entry carries
`verified=False`, and `assert_verified_corpus()` will refuse to let an
unverified corpus be used in any clinical or published context.

Before this is used for anything that reaches a clinician or a reviewer:
resolve each `citation` to a DOI, check that the passage reflects what the
paper actually found, and flip `verified` to True. A recommender that cites
literature it has not checked is worse than one that cites nothing, because
the citation is what buys the trust.

Target corpus size for the full system is 500-1000 passages spanning
literature plus therapist guidance; this seed has far fewer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Passage:
    """One retrievable unit of evidence or guidance."""

    passage_id: str
    text: str
    citation: str
    source_type: str  # "literature" | "clinical_guidance"
    tags: tuple[str, ...]
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.text.strip():
            raise ValueError(f"{self.passage_id}: text must not be empty")
        if self.source_type not in {"literature", "clinical_guidance"}:
            raise ValueError(f"{self.passage_id}: unknown source_type {self.source_type!r}")


SEED_PASSAGES: tuple[Passage, ...] = (
    Passage(
        "lit_tempo_arousal",
        "Musical tempo tracks physiological arousal: slower tempi are associated with "
        "reduced heart rate and respiratory rate, while faster tempi raise both. Tempo "
        "is among the most reliable single predictors of an autonomic response to music.",
        "Bernardi, Porta & Sleight (1996/2006), cardiovascular and respiratory responses to music",
        "literature",
        ("tempo", "arousal", "autonomic"),
    ),
    Passage(
        "lit_music_stress_response",
        "Listening to music before or during a stressor is associated with faster recovery "
        "of the autonomic nervous system and lower subjective stress, though effects vary "
        "substantially with listener choice and musical style.",
        "Thoma et al. (2013), PLoS ONE, the effect of music on the human stress response",
        "literature",
        ("stress", "recovery", "intervention"),
    ),
    Passage(
        "lit_neurochemistry",
        "Music engages reward, arousal and stress-regulation systems; effects on cortisol "
        "and on subjective calm are documented across relaxation and clinical contexts.",
        "Chanda & Levitin (2013), Trends in Cognitive Sciences, the neurochemistry of music",
        "literature",
        ("stress", "reward", "cortisol"),
    ),
    Passage(
        "lit_alpha_relaxation",
        "Alpha-band power (8-13 Hz) increases in relaxed, low-arousal wakefulness and "
        "decreases with alertness, attention and stress. Frontal alpha is therefore a "
        "common index of relaxation in music-listening studies.",
        "Klimesch (1999), Brain Research Reviews, alpha and theta oscillations",
        "literature",
        ("alpha", "relaxation", "eeg"),
    ),
    Passage(
        "lit_beta_alpha_ratio",
        "The ratio of beta to alpha power is used as a cortical arousal index: it rises "
        "under stress and cognitive load and falls during relaxation, making it a more "
        "stable marker than either band alone.",
        "Pfurtscheller & Lopes da Silva (1999), Clinical Neurophysiology, ERD/ERS",
        "literature",
        ("beta", "alpha", "ratio", "arousal"),
    ),
    Passage(
        "lit_frontal_theta",
        "Frontal-midline theta at Fz is associated with cognitive control and sustained "
        "mental effort; it typically elevates under task-induced stress.",
        "Klimesch (1999), Brain Research Reviews, alpha and theta oscillations",
        "literature",
        ("theta", "fz", "effort"),
    ),
    Passage(
        "lit_raga_structure",
        "Emotional responses to Hindustani raga music relate systematically to musical "
        "structure, including scale (thaat) and tempo, with listener familiarity "
        "moderating the response.",
        "Mathur et al. (2015), Frontiers in Psychology, emotional responses to Hindustani raga music",
        "literature",
        ("raga", "scale", "emotion", "tempo"),
    ),
    Passage(
        "lit_drone_continuity",
        "A sustained tonic drone provides continuous, low-variability harmonic context. "
        "Reduced acoustic surprise is associated with lower orienting response and is a "
        "plausible mechanism for the calming character attributed to drone-based music.",
        "Synthesised from music-perception literature on predictability and arousal -- NEEDS A PRIMARY SOURCE",
        "literature",
        ("drone", "predictability", "arousal"),
    ),
    Passage(
        "lit_music_anxiety_review",
        "Systematic reviews of music interventions for anxiety report small-to-moderate "
        "reductions in subjective anxiety, with heterogeneity across populations and a "
        "recurring recommendation that music be matched to listener preference.",
        "Bradt & Dileo, Cochrane systematic reviews of music interventions",
        "literature",
        ("anxiety", "review", "preference"),
    ),
    Passage(
        "guid_preference_first",
        "Listener preference is a strong moderator of outcome. A theoretically calming "
        "track that the listener dislikes can raise rather than lower arousal, so a "
        "recommender should treat measured individual response as outranking generic "
        "acoustic rules.",
        "Clinical guidance summary -- ILLUSTRATIVE, replace with your therapist protocol",
        "clinical_guidance",
        ("preference", "personalisation"),
    ),
    Passage(
        "guid_rhythmic_intensity",
        "Dense percussive activity carries entrainment pressure that can sustain arousal "
        "even at low tempo. For a downregulation goal, prefer sparse rhythmic texture "
        "rather than relying on tempo alone.",
        "Clinical guidance summary -- ILLUSTRATIVE, replace with your therapist protocol",
        "clinical_guidance",
        ("rhythm", "entrainment", "downregulation"),
    ),
    Passage(
        "guid_session_length",
        "Relaxation responses to music typically need several minutes to establish. "
        "Very short excerpts may not produce a measurable physiological change even when "
        "the selection is appropriate.",
        "Clinical guidance summary -- ILLUSTRATIVE, replace with your therapist protocol",
        "clinical_guidance",
        ("duration", "protocol"),
    ),
    Passage(
        "guid_no_diagnosis",
        "EEG-derived stress indices are research measures, not diagnostic instruments. "
        "Recommendations should be framed as supportive, and any clinical decision must "
        "rest on assessment by a qualified practitioner.",
        "Clinical guidance summary -- ILLUSTRATIVE, replace with your governance policy",
        "clinical_guidance",
        ("safety", "scope", "governance"),
    ),
)


def assert_verified_corpus(passages: tuple[Passage, ...]) -> None:
    """Refuse to proceed if any citation is still unchecked."""
    unverified = [p.passage_id for p in passages if not p.verified]
    if unverified:
        raise ValueError(
            f"{len(unverified)} passage(s) have unverified citations: {unverified}. "
            "Resolve each to a DOI and confirm the paraphrase before using this "
            "corpus outside development."
        )


def load_corpus(path: str | None = None) -> tuple[Passage, ...]:
    """Load a JSON corpus, or fall back to the development seed.

    JSON schema: a list of objects with keys passage_id, text, citation,
    source_type, tags, verified.
    """
    if path is None:
        return SEED_PASSAGES

    import json
    from pathlib import Path

    file = Path(path)
    if not file.is_file():
        raise FileNotFoundError(f"corpus file not found: {file}")
    raw = json.loads(file.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{file} must contain a non-empty JSON list of passages")
    return tuple(
        Passage(
            passage_id=str(item["passage_id"]),
            text=str(item["text"]),
            citation=str(item["citation"]),
            source_type=str(item["source_type"]),
            tags=tuple(item.get("tags", ())),
            verified=bool(item.get("verified", False)),
        )
        for item in raw
    )
