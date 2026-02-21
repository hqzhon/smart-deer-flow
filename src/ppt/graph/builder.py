# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT

from src.graph.common_builder import build_simple_graph
from src.ppt.graph.ppt_composer_node import ppt_composer_node
from src.ppt.graph.ppt_generator_node import ppt_generator_node
from src.ppt.graph.state import PPTState


def build_graph():
    """Build and return the ppt workflow graph."""
    return build_simple_graph(
        state_class=PPTState,
        nodes=[
            ("ppt_composer", ppt_composer_node),
            ("ppt_generator", ppt_generator_node),
        ],
        edges=[
            ("ppt_composer", "ppt_generator"),
        ],
    )


workflow = build_graph()

if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    report_content = open("examples/nanjing_tangbao.md").read()
    final_state = workflow.invoke({"input": report_content})
