from typing import Any, Dict, List, Optional

from AgentTailor.ATNetwork.Actor import Actor


def prepare_edge_inputs(
    summary: str,
    edge_records: List[Dict[str, Any]],
    actor: Actor,
    include_output: bool = True,
    max_output_len: Optional[int] = None,
) -> List[Dict[str, Any]]:
    edge_inputs: List[Dict[str, Any]] = []
    seen_edges = set()

    for record in edge_records:
        out_node_id = record.get("out_node_id", "")
        in_node_id = record.get("in_node_id", "")
        edge_type = record.get("type", "spatial")
        round_idx = record.get("round", 0)

        if edge_type == "spatial" and out_node_id == in_node_id:
            continue

        edge_key = (out_node_id, in_node_id, edge_type, round_idx)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        out_desc, in_desc = actor.get_edge_node_descriptions(
            out_node_id,
            in_node_id,
            include_output=include_output,
            max_output_len=max_output_len,
        )

        edge_inputs.append(
            {
                "out_node_id": out_node_id,
                "in_node_id": in_node_id,
                "node1_info": out_desc,
                "node2_info": in_desc,
                "edge_info": str(record.get("out_output", "")),
                "ans_info": summary,
                "type": edge_type,
                "selected": True,
                "round": round_idx,
            }
        )
    return edge_inputs


def get_unselected_edges(
    actor: Actor,
    edge_records: List[Dict[str, Any]],
    task_text: str,
    include_output: bool = False,
    max_output_len: Optional[int] = None,
) -> List[Dict[str, Any]]:

    selected_edges = set()
    for record in edge_records:
        edge_type = record.get("type", "spatial")
        edge_key = (record["out_node_id"], record["in_node_id"], edge_type)
        selected_edges.add(edge_key)

    unselected_edge_inputs: List[Dict[str, Any]] = []

    for edge in actor.potential_spatial_edges:
        if edge[0] == edge[1]:
            continue
        edge_key = (edge[0], edge[1], "spatial")
        if edge_key in selected_edges:
            continue
        out_desc, in_desc = actor.get_edge_node_descriptions(
            edge[0],
            edge[1],
            include_output=include_output,
            max_output_len=max_output_len,
        )
        unselected_edge_inputs.append(
            {
                "out_node_id": edge[0],
                "in_node_id": edge[1],
                "node1_info": out_desc,
                "node2_info": in_desc,
                "edge_info": "",
                "ans_info": task_text,
                "type": "spatial",
                "selected": False,
            }
        )

    for edge in actor.potential_temporal_edges:
        edge_key = (edge[0], edge[1], "temporal")
        if edge_key in selected_edges:
            continue
        out_desc, in_desc = actor.get_edge_node_descriptions(
            edge[0],
            edge[1],
            include_output=include_output,
            max_output_len=max_output_len,
        )
        unselected_edge_inputs.append(
            {
                "out_node_id": edge[0],
                "in_node_id": edge[1],
                "node1_info": out_desc,
                "node2_info": in_desc,
                "edge_info": "",
                "ans_info": task_text,
                "type": "temporal",
                "selected": False,
            }
        )

    return unselected_edge_inputs


__all__ = ["prepare_edge_inputs", "get_unselected_edges"]


