from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


from jev_rag_research.shared.config import (
    GENERATION_MODEL,
    TOP_K,
    EMBEDDING_MODEL,
)
from jev_rag_research.shared.embeddings import embed_text
from jev_rag_research.shared.retrieval import retrieve
from jev_rag_research.normal_rag.rag import run_rag
from jev_rag_research.jev_rag.oracle import KeywordOracle
from jev_rag_research.jev_rag.rag import run_filtered_rag
from jev_rag_research.jev_rag.early_exit import run_early_exit_rag


DATA = ROOT / "data"

# The repository supplied with the project stores these under data/.
# The app also accepts the trimmed project's top-level data layout.
if not (DATA / "processed").exists() and (ROOT / "processed").exists():
    DATA = ROOT


QUESTIONS_FILE = DATA / "processed" / "chunked_questions.json"
CHUNKS_FILE = DATA / "processed" / "chunks.json"
EMBEDDINGS_FILE = DATA / "embeddings" / "chunk_embeddings.json"
RESULTS_DIR = ROOT / "results"


st.set_page_config(
    page_title="JEV-RAG | Research Demo",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>

    .stApp {
        background: #0a0d12;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stSidebar"] {
        background: #0d1118;
        border-right: 1px solid #202938;
    }

    .hero {
        padding: 1.4rem 1.6rem;
        border: 1px solid #263244;
        border-radius: 18px;
        background: linear-gradient(135deg,#111722,#0b1018);
        margin-bottom: 1.2rem;
    }

    .eyebrow {
        color:#8ea2c5;
        font-size:.78rem;
        letter-spacing:.12em;
        text-transform:uppercase;
        font-weight:700;
    }

    .hero h1 {
        margin:.25rem 0 .35rem;
        font-size:2.25rem;
        letter-spacing:-.04em;
    }

    .hero p {
        color:#aebbd0;
        margin:0;
        max-width:850px;
        font-size:1rem;
    }

    .pill {
        display:inline-block;
        padding:.28rem .65rem;
        border-radius:999px;
        background:#182334;
        border:1px solid #2a3950;
        color:#a9bddb;
        font-size:.72rem;
        margin:.6rem .35rem 0 0;
    }

    .card {
        border:1px solid #252f3e;
        border-radius:14px;
        background:#0f141d;
        padding:1rem;
    }

    .chunk {
        border:1px solid #293445;
        border-radius:14px;
        padding:1rem;
        margin:.6rem 0;
        background:#0d121a;
    }

    .chunk.keep {
        border-color:#315e4a;
    }

    .chunk.drop {
        border-color:#51363d;
        opacity:.82;
    }

    .chunk-title {
        display:flex;
        justify-content:space-between;
        align-items:center;
        gap:1rem;
    }

    .badge {
        padding:.2rem .5rem;
        border-radius:999px;
        font-size:.7rem;
        font-weight:700;
    }

    .keep-badge {
        color:#9ce2b8;
        background:#12271d;
        border:1px solid #28583b;
    }

    .drop-badge {
        color:#e9a7ad;
        background:#29161a;
        border:1px solid #5a3037;
    }

    .muted {
        color:#8795aa;
        font-size:.82rem;
    }

    .answer {
        border:1px solid #32435b;
        border-radius:16px;
        padding:1.2rem;
        background:#101722;
    }

    .answer-label {
        color:#91a9ca;
        text-transform:uppercase;
        letter-spacing:.1em;
        font-size:.7rem;
        font-weight:700;
    }

    .answer-text {
        color:#edf3fc;
        font-size:1.12rem;
        line-height:1.65;
        margin-top:.45rem;
    }

    .section-title {
        font-size:1.1rem;
        font-weight:700;
        margin:1.2rem 0 .5rem;
    }

    .warning {
        padding:.75rem .9rem;
        border-radius:10px;
        background:#241f12;
        border:1px solid #55451f;
        color:#d9c58e;
        font-size:.82rem;
    }

    .stMetric {
        background:#101620;
        border:1px solid #252f3e;
        padding:.65rem;
        border-radius:12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def load_resources():

    questions = load_json(str(QUESTIONS_FILE))
    chunks = load_json(str(CHUNKS_FILE))
    embedding_data = load_json(str(EMBEDDINGS_FILE))

    embedding_map = {
        x["chunk_id"]: x["embedding"]
        for x in embedding_data
    }

    chunk_embeddings = [
        embedding_map.get(x["id"])
        for x in chunks
    ]

    chunk_index = {
        x["id"]: i
        for i, x in enumerate(chunks)
    }

    return questions, chunks, chunk_embeddings, chunk_index


def pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def fmt_time(x: float) -> str:
    return f"{x * 1000:.0f} ms" if x < 1 else f"{x:.2f} s"


def render_chunk(
    chunk_id,
    score,
    text,
    status=None,
    confidence=None,
    rank=None,
):

    cls = (
        "keep"
        if status is True
        else "drop"
        if status is False
        else ""
    )

    if status is True:
        badge = '<span class="badge keep-badge">KEEP</span>'
    elif status is False:
        badge = '<span class="badge drop-badge">FILTERED</span>'
    else:
        badge = (
            '<span class="badge" '
            'style="color:#9db0c9;'
            'background:#17202d;'
            'border:1px solid #2b394c">'
            'RETRIEVED'
            '</span>'
        )

    rank_text = f" · Rank {rank}" if rank else ""

    conf_text = (
        f" · Decision confidence {confidence:.2f}"
        if confidence is not None
        else ""
    )

    safe_chunk_id = html.escape(str(chunk_id))
    safe_text = html.escape(str(text))

    # Built as a single line, with no embedded newlines or leading
    # whitespace, so Streamlit's Markdown parser can never mistake
    # any part of it (e.g. after a blank line) for an indented code
    # block instead of an HTML block.
    html_block = (
        f'<div class="chunk {cls}">'
        f'<div class="chunk-title">'
        f'<div><strong>{safe_chunk_id}</strong>'
        f'<span class="muted">'
        f'{rank_text}'
        f' · similarity {score:.3f}'
        f'{conf_text}'
        f'</span></div>'
        f'<div>{badge}</div>'
        f'</div>'
        f'<div style="margin-top:.65rem;color:#c3cede;'
        f'line-height:1.55;white-space:pre-wrap;">'
        f'{safe_text}'
        f'</div>'
        f'</div>'
    )

    st.markdown(
        html_block,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------

# Single-line HTML for the same reason as render_chunk above: no
# blank lines or indented continuation lines for Markdown to
# misinterpret as a code block.
hero_html = (
    '<div class="hero">'
    '<div class="eyebrow">Research prototype · RAG efficiency</div>'
    '<h1>JEV-RAG</h1>'
    '<p>Decision-guided context filtering and early exit '
    'for Retrieval-Augmented Generation.</p>'
    '<span class="pill">nomic-embed-text</span>'
    '<span class="pill">NumPy cosine retrieval</span>'
    '<span class="pill">Gemma 3 4B</span>'
    '<span class="pill">Top-5 retrieval</span>'
    '</div>'
)

st.markdown(
    hero_html,
    unsafe_allow_html=True,
)


try:

    questions, chunks, chunk_embeddings, chunk_index = (
        load_resources()
    )

except Exception as exc:

    st.error(
        f"Could not load the project data: {exc}"
    )

    st.stop()


# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------

with st.sidebar:

    st.markdown("### Demo controls")

    mode = st.radio(
        "Pipeline",
        [
            "Normal RAG",
            "Decision-Guided Filter",
            "Early Exit",
        ],
        index=1,
    )

    threshold = st.slider(
        "Decision threshold",
        0.00,
        0.50,
        0.15,
        0.01,
    )

    if mode == "Normal RAG":
        threshold = None

    st.caption(
        "The current decision backend is the deterministic "
        "KeywordOracle prototype — not JEV."
    )

    st.divider()

    st.markdown("### Local stack")

    st.write(
        f"Embedding: `{EMBEDDING_MODEL}`"
    )

    st.write(
        f"Generation: `{GENERATION_MODEL}`"
    )

    st.write(
        f"Candidate pool: `Top-{TOP_K}`"
    )

    st.write(
        f"Questions: `{len(questions)}`"
    )

    st.write(
        f"Chunks: `{len(chunks)}`"
    )

    st.divider()

    st.markdown("### Research snapshot")

    paired_summary_path = (
        RESULTS_DIR / "paired_comparison.json"
    )

    if paired_summary_path.exists():

        paired_summary = load_json(
            str(paired_summary_path)
        )

        st.metric(
            "Retrieval Recall@5",
            pct(
                paired_summary["gold_chunk"][
                    "survival_rate"
                ]
            ),
        )

        st.metric(
            "Normal RAG F1",
            pct(
                paired_summary["normal"]["f1"]
            ),
        )

        st.metric(
            "Filter @0.15 context reduction",
            pct(
                paired_summary["efficiency"][
                    "average_context_reduction"
                ]
            ),
        )

    else:

        st.caption(
            "Run the paired comparison experiment to "
            "populate this snapshot."
        )


# -----------------------------------------------------------------------------
# Question input
# -----------------------------------------------------------------------------

question_options = [
    q["question"]
    for q in questions[:30]
]

selected_example = st.selectbox(
    "Try a dataset question",
    ["Custom question"] + question_options,
)

if selected_example != "Custom question":

    default_question = selected_example

else:

    default_question = (
        "Who led the Mongol invasion of Europe?"
    )


question = st.text_area(
    "Question",
    value=default_question,
    height=90,
    placeholder=(
        "Ask a question that can be answered "
        "from the indexed SQuAD passages..."
    ),
)


run = st.button(
    "Run pipeline",
    type="primary",
    use_container_width=True,
)


if run:

    if not question.strip():

        st.warning("Enter a question first.")
        st.stop()

    with st.spinner(
        "Running local RAG pipeline..."
    ):

        try:

            if mode == "Normal RAG":

                result = run_rag(
                    question.strip(),
                    chunks,
                    chunk_embeddings,
                )

                result["mode"] = mode
                result["decisions"] = []

                result["selected_chunks"] = [
                    x["chunk_id"]
                    for x in result["retrieved"]
                ]

                # Normalize Normal RAG to the schema expected
                # by the dashboard.

                result["num_retrieved"] = (
                    result["num_chunks"]
                )

                result["num_selected"] = (
                    result["num_chunks"]
                )

                result["context_characters_before"] = (
                    result["context_characters"]
                )

                result["context_characters_after"] = (
                    result["context_characters"]
                )

                result["context_reduction"] = 0.0

            elif mode == "Decision-Guided Filter":

                backend = KeywordOracle(
                    threshold=float(threshold)
                )

                result = run_filtered_rag(
                    question.strip(),
                    chunks,
                    chunk_embeddings,
                    backend,
                )

                result["mode"] = mode

            else:

                backend = KeywordOracle(
                    threshold=float(threshold)
                )

                result = run_early_exit_rag(
                    question.strip(),
                    chunks,
                    chunk_embeddings,
                    backend,
                )

                result["mode"] = mode

        except Exception as exc:

            st.error(
                f"Pipeline failed: {exc}"
            )

            st.info(
                "Make sure Ollama is running and "
                "the configured models are available."
            )

            st.stop()


    # -------------------------------------------------------------------------
    # Summary metrics
    # -------------------------------------------------------------------------

    st.markdown(
        "<div class='section-title'>Run summary</div>",
        unsafe_allow_html=True,
    )

    cols = st.columns(5)

    if mode == "Early Exit":

        cols[0].metric(
            "Retrieval depth",
            f"{result['retrieval_depth']}/{TOP_K}",
        )

        cols[1].metric(
            "Chunks used",
            f"{result['num_selected']}/"
            f"{result['num_candidates']}",
        )

        cols[2].metric(
            "Context reduction",
            pct(result['context_reduction']),
        )

        cols[3].metric(
            "Decision time",
            fmt_time(result['decision_time']),
        )

        cols[4].metric(
            "Total time",
            fmt_time(result['total_time']),
        )

    else:

        num_selected = result.get(
            "num_selected",
            result.get("num_chunks", 0),
        )

        num_retrieved = result.get(
            "num_retrieved",
            result.get("num_chunks", 0),
        )

        context_reduction = result.get(
            "context_reduction",
            0.0,
        )

        context_after = result.get(
            "context_characters_after",
            result.get("context_characters", 0),
        )

        cols[0].metric(
            "Chunks used",
            f"{num_selected}/{num_retrieved}",
        )

        cols[1].metric(
            "Context reduction",
            pct(context_reduction),
        )

        cols[2].metric(
            "Context",
            f"{context_after:,} chars",
        )

        cols[3].metric(
            "LLM time",
            fmt_time(
                result.get("llm_time", 0.0)
            ),
        )

        cols[4].metric(
            "Total time",
            fmt_time(
                result.get("total_time", 0.0)
            ),
        )


    if mode != "Normal RAG":

        decisions = (
            result.get("decisions") or []
        )

        backend_name = (
            decisions[0].get(
                "backend",
                "keyword_oracle",
            )
            if decisions
            else "keyword_oracle"
        )

        # Single-line HTML — see the note in render_chunk above.
        warning_html = (
            '<div class="warning">Decision backend: '
            f'<strong>{html.escape(str(backend_name))}</strong>. '
            'This demo visualizes the current prototype decision '
            'mechanism; it is not an evaluation of JEV itself.'
            '</div>'
        )

        st.markdown(
            warning_html,
            unsafe_allow_html=True,
        )


    st.markdown(
        "<div class='section-title'>Generated answer</div>",
        unsafe_allow_html=True,
    )

    # Single-line HTML — see the note in render_chunk above.
    answer_html = (
        '<div class="answer">'
        f'<div class="answer-label">{html.escape(str(mode))}</div>'
        f'<div class="answer-text">'
        f'{html.escape(str(result["answer"]))}'
        f'</div>'
        '</div>'
    )

    st.markdown(
        answer_html,
        unsafe_allow_html=True,
    )


    # -------------------------------------------------------------------------
    # Pipeline visualization
    # -------------------------------------------------------------------------

    tab_chunks, tab_context, tab_timing = st.tabs(
        [
            "Retrieved evidence",
            "Context comparison",
            "Timing",
        ]
    )


    with tab_chunks:

        st.markdown(
            "#### Evidence decisions"
        )

        decision_map = {
            d["chunk_id"]: d
            for d in result.get(
                "decisions",
                [],
            )
        }


        if mode == "Early Exit":

            candidates = result[
                "retrieved_candidates"
            ]

            exited = False

            for item in candidates:

                d = decision_map.get(
                    item["chunk_id"],
                    {},
                )

                cid = item["chunk_id"]

                text = chunks[
                    chunk_index[cid]
                ]["text"]

                render_chunk(
                    cid,
                    item["similarity"],
                    text,
                    d.get("keep")
                    if d
                    else None,
                    d.get("confidence")
                    if d
                    else None,
                    item["rank"],
                )

                if d.get("keep"):

                    st.caption(
                        f"Early exit triggered at rank "
                        f"{d['rank']} — accumulated "
                        f"evidence accepted."
                    )

                    exited = True

                    break

            if not exited:

                st.caption(
                    "Early exit was not triggered — the "
                    "decision backend never accepted the "
                    "accumulated evidence within the "
                    "candidate pool."
                )

        else:

            for rank, item in enumerate(
                result["retrieved"],
                1,
            ):

                cid = item["chunk_id"]

                d = decision_map.get(cid)

                text = chunks[
                    chunk_index[cid]
                ]["text"]

                render_chunk(
                    cid,
                    item.get(
                        "similarity",
                        item.get("score", 0.0),
                    ),
                    text,
                    d.get("keep")
                    if d
                    else None,
                    d.get("confidence")
                    if d
                    else None,
                    rank,
                )


    with tab_context:

        before = result.get(
            "context_characters_before",
            result.get(
                "context_characters",
                0,
            ),
        )

        after = result.get(
            "context_characters_after",
            result.get(
                "context_characters",
                0,
            ),
        )

        reduction = result.get(
            "context_reduction",
            0.0,
        )

        a, b, c = st.columns(3)

        a.metric(
            "Before decision",
            f"{before:,} chars",
        )

        b.metric(
            "After decision",
            f"{after:,} chars",
        )

        c.metric(
            "Reduction",
            pct(reduction),
        )


        if mode == "Normal RAG":

            st.info(
                "Normal RAG sends the full Top-5 "
                "retrieved context to the generator."
            )

        elif mode == "Decision-Guided Filter":

            st.info(
                "The prototype evaluates each "
                "retrieved chunk independently and "
                "keeps the chunks above the decision "
                "threshold."
            )

        else:

            st.info(
                "The prototype accumulates retrieved "
                "evidence and exits when the decision "
                "backend accepts the accumulated context."
            )


    with tab_timing:

        timing_data = {
            "Query embedding": result.get(
                "embedding_time",
                0,
            ),

            "Retrieval": result.get(
                "retrieval_time",
                0,
            ),

            "Decision": result.get(
                "decision_time",
                0,
            ),

            "LLM generation": result.get(
                "llm_time",
                0,
            ),

            "Total": result.get(
                "total_time",
                0,
            ),
        }

        for label, value in timing_data.items():

            st.write(
                f"**{label}** — {fmt_time(value)}"
            )

            if label != "Total":

                st.progress(
                    min(
                        value
                        / max(
                            result.get(
                                "total_time",
                                1,
                            ),
                            1e-9,
                        ),
                        1.0,
                    )
                )


# -----------------------------------------------------------------------------
# Research results
# -----------------------------------------------------------------------------

st.divider()

st.markdown(
    "<div class='section-title'>Measured research results</div>",
    unsafe_allow_html=True,
)

st.caption(
    "These numbers are loaded from the existing experiment "
    "artifacts in the project; they are not produced by "
    "the single-question demo run."
)


paired_path = (
    RESULTS_DIR /
    "paired_comparison.json"
)

early_answer_path = (
    RESULTS_DIR /
    "oracle_early_exit_answer_evaluation.json"
)

early_result_path = (
    RESULTS_DIR /
    "oracle_early_exit_accumulated_results_200.json"
)


if paired_path.exists():

    paired = load_json(
        str(paired_path)
    )

    normal = paired["normal"]
    filtered = paired["filtered"]
    eff = paired["efficiency"]
    g = paired["gold_chunk"]

    rcols = st.columns(4)

    rcols[0].metric(
        "Normal RAG F1",
        f"{normal['f1'] * 100:.2f}%",
    )

    rcols[1].metric(
        "Filtered F1",
        f"{filtered['f1'] * 100:.2f}%",
    )

    rcols[2].metric(
        "Context reduction",
        f"{eff['average_context_reduction'] * 100:.2f}%",
    )

    rcols[3].metric(
        "Gold evidence retained",
        f"{g['survival_rate'] * 100:.2f}%",
    )


    with st.expander(
        "Filtering experiment details"
    ):

        st.write(
            {
                "Questions": paired[
                    "paired_questions"
                ],

                "Normal EM": normal[
                    "exact_match"
                ],

                "Filtered EM": filtered[
                    "exact_match"
                ],

                "Average chunks removed": eff[
                    "average_chunks_removed"
                ],

                "Normal average total time": eff[
                    "normal_average_total_time"
                ],

                "Filtered average total time": eff[
                    "filtered_average_total_time"
                ],

                "Gold chunks survived": g[
                    "survived"
                ],

                "Gold chunks lost": g[
                    "lost"
                ],
            }
        )


if early_answer_path.exists():

    early = load_json(
        str(early_answer_path)
    )

    # Answer-quality metrics come from the
    # answer-evaluation artifact.
    # Efficiency/context metrics come from
    # the early-exit run artifact, which stores
    # a list of per-question accumulated results
    # rather than a single aggregate dict.

    early_result = (
        load_json(
            str(early_result_path)
        )
        if early_result_path.exists()
        else []
    )

    context_reduction = 0.0

    if isinstance(early_result, list):

        reductions = []

        for item in early_result:

            if not isinstance(item, dict):
                continue

            value = item.get("context_reduction")

            if value is not None:
                reductions.append(float(value))

        if reductions:
            context_reduction = (
                sum(reductions) / len(reductions)
            )

    elif isinstance(early_result, dict):

        context_reduction = float(
            early_result.get(
                "context_reduction",
                early_result.get(
                    "average_context_reduction",
                    0.0,
                ),
            )
        )

    # Some artifacts may store percentages
    # rather than fractions.

    if context_reduction > 1:
        context_reduction /= 100.0

    st.markdown(
        "#### Early-exit experiment"
    )

    ec = st.columns(5)

    ec[0].metric(
        "F1",
        f"{early.get('token_f1', 0.0) * 100:.2f}%",
    )

    ec[1].metric(
        "Avg depth",
        f"{early.get('average_retrieval_depth', 0.0):.2f}",
    )

    ec[2].metric(
        "Context reduction",
        f"{context_reduction * 100:.2f}%",
    )

    ec[3].metric(
        "Gold survival",
        f"{early.get('gold_chunk_survival', 0.0) * 100:.2f}%",
    )

    ec[4].metric(
        "Early-exit rate",
        f"{early.get('early_exit_rate', 0.0) * 100:.2f}%",
    )


    with st.expander(
        "Early-exit experiment details"
    ):

        st.write(
            {
                "Questions": early.get(
                    "questions"
                ),

                "Exact Match": early.get(
                    "exact_match"
                ),

                "Token F1": early.get(
                    "token_f1"
                ),

                "Average retrieval depth": early.get(
                    "average_retrieval_depth"
                ),

                "Early-exit rate": early.get(
                    "early_exit_rate"
                ),

                "Gold chunk survival": early.get(
                    "gold_chunk_survival"
                ),

                "Context reduction": context_reduction,
            }
        )


st.caption(
    "Research status: the current UI demonstrates "
    "the implemented RAG pipelines. The KeywordOracle "
    "is a development proxy and should not be presented "
    "as JEV results."
)