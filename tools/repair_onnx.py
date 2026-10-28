#!/usr/bin/env python3
"""
Repairs a malformed quantised ONNX graph.

szFangshi/tiny-sd-int8 was exported with a broken naming pass: for every quantised conv/matmul,
the producer node was given the *downstream* Cast node's output name, so

    QLinearConv  ->  /conv_in/Conv_output_0_output_quantized_cast_output   (wrong: should be
                                                                           ..._output_quantized)
    Cast         ->  /conv_in/Conv_output_0_output_quantized_cast_output   (correct)

leaves two nodes defining one tensor and nothing defining the tensor the Cast actually reads.
ONNX Runtime reports this as "Duplicate definition of name".

The repair restores the producer's original name. For each duplicated tensor N:

  * if one definition is a Cast whose input M is undefined, then the other definition is the
    producer that should have emitted M — rename it back to M;
  * otherwise fall back to renaming the later definition and rewiring its consumers (graphs are
    topologically ordered, so consumers always follow their producer).

    python tools/repair_onnx.py <model.onnx> [output.onnx]
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import onnx


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: repair_onnx.py <model.onnx> [output.onnx]")
    src = Path(sys.argv[1])
    dst = Path(sys.argv[2]) if len(sys.argv) > 2 else src.with_name(src.stem + ".fixed.onnx")

    print(f"parsing {src.name} ({src.stat().st_size / 1048576:.1f} MB)")
    model = onnx.ModelProto()
    model.ParseFromString(src.read_bytes())
    graph = model.graph

    defined: set[str] = {i.name for i in graph.initializer}
    defined |= {i.name for i in graph.input}
    for node in graph.node:
        defined |= {o for o in node.output if o}

    definitions: dict[str, list[int]] = defaultdict(list)
    for index, node in enumerate(graph.node):
        for out in node.output:
            if out:
                definitions[out].append(index)

    duplicated = {n: idxs for n, idxs in definitions.items() if len(idxs) > 1}
    print(f"nodes: {len(graph.node)}, tensors defined more than once: {len(duplicated)}")
    if not duplicated:
        print("nothing to repair")
        return

    restored = 0
    rewired = 0
    active: dict[str, str] = {}

    # Pass 1: restore producer names where a Cast consumes an undefined tensor.
    for name, idxs in duplicated.items():
        for def_index in idxs:
            node = graph.node[def_index]
            if node.op_type != "Cast" or not node.input:
                continue
            source = node.input[0]
            if source in defined:
                continue
            for other_index in idxs:
                if other_index == def_index:
                    continue
                other = graph.node[other_index]
                for pos, out in enumerate(other.output):
                    if out == name:
                        other.output[pos] = source
                        defined.add(source)
                        restored += 1
                        print(f"  restored {source[-56:]}")
                        break
                break

    # Pass 2: anything still duplicated gets a unique name and its consumers rewired.
    defined.clear()
    defined |= {i.name for i in graph.initializer}
    defined |= {i.name for i in graph.input}
    for node in graph.node:
        defined |= {o for o in node.output if o}

    definitions.clear()
    for index, node in enumerate(graph.node):
        for out in node.output:
            if out:
                definitions[out].append(index)

    still = {n: idxs for n, idxs in definitions.items() if len(idxs) > 1}
    counters: dict[str, int] = defaultdict(int)
    rename_by_node: dict[int, tuple[str, str]] = {}
    for name, idxs in still.items():
        for def_index in idxs[1:]:
            counters[name] += 1
            rename_by_node[def_index] = (name, f"{name}__dup{counters[name]}")

    for index, node in enumerate(graph.node):
        for pos, inp in enumerate(node.input):
            if inp in active:
                node.input[pos] = active[inp]
                rewired += 1
        if index in rename_by_node:
            old, new = rename_by_node[index]
            for pos, out in enumerate(node.output):
                if out == old:
                    node.output[pos] = new
            active[old] = new

    print(f"restored {restored} producer names, renamed {len(rename_by_node)} leftovers, "
          f"rewired {rewired} inputs")
    dst.write_bytes(model.SerializeToString())
    print(f"wrote {dst} ({dst.stat().st_size / 1048576:.1f} MB)")


if __name__ == "__main__":
    main()
