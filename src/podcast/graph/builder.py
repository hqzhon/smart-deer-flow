# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from src.graph.common_builder import build_simple_graph
from src.podcast.graph.audio_mixer_node import audio_mixer_node
from src.podcast.graph.script_writer_node import script_writer_node
from src.podcast.graph.state import PodcastState
from src.podcast.graph.tts_node import tts_node


def build_graph():
    """Build and return the podcast workflow graph."""
    return build_simple_graph(
        state_class=PodcastState,
        nodes=[
            ("script_writer", script_writer_node),
            ("tts", tts_node),
            ("audio_mixer", audio_mixer_node),
        ],
        edges=[
            ("script_writer", "tts"),
            ("tts", "audio_mixer"),
        ],
    )


workflow = build_graph()

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    report_content = open("examples/nanjing_tangbao.md").read()
    final_state = workflow.invoke({"input": report_content})
    for line in final_state["script"].lines:
        print("<M>" if line.speaker == "male" else "<F>", line.text)

    with open("final.mp3", "wb") as f:
        f.write(final_state["output"])
