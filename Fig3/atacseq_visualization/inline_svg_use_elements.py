"""
Guaranteed fallback fix for the matplotlib-SVG-to-Illustrator <use>
rendering bug: rewrites every <use xlink:href="#id" x=".." y=".."/>
element by replacing it with the ACTUAL <path>/content it references
from <defs>, translated by the <use> element's x/y offset, so the
output SVG has no <use>/<defs> indirection left at all. This works
regardless of WHY matplotlib chose to emit <use> (path simplification
caching, marker reuse, etc.) -- it fixes the symptom directly.

Usage:
    python inline_svg_use_elements.py input.svg output.svg
"""

import sys
import re
import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
XLINK_NS = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG_NS)
ET.register_namespace("xlink", XLINK_NS)


def inline_use_elements(input_path, output_path):
    tree = ET.parse(input_path)
    root = tree.getroot()

    # index every element in <defs> by its id, searching the whole tree
    # (matplotlib sometimes nests defs, so don't assume a single top-level block)
    defs_by_id = {}
    for el in root.iter():
        el_id = el.get("id")
        if el_id:
            defs_by_id[el_id] = el

    use_tag = f"{{{SVG_NS}}}use"
    href_attr = f"{{{XLINK_NS}}}href"

    n_replaced = 0
    n_missing = 0

    # walk the tree looking for <use> elements and their parents, so we
    # can replace each <use> in-place with a copy of its referenced content
    for parent in root.iter():
        children = list(parent)
        for i, child in enumerate(children):
            if child.tag != use_tag:
                continue

            href = child.get(href_attr) or child.get("href")
            if href is None or not href.startswith("#"):
                continue
            ref_id = href[1:]

            referenced = defs_by_id.get(ref_id)
            if referenced is None:
                print(f"  WARNING: <use> references '#{ref_id}' but no matching element found -- leaving as-is")
                n_missing += 1
                continue

            x = child.get("x", "0")
            y = child.get("y", "0")
            style = child.get("style", "")

            # build a <g> wrapping a deep copy of the referenced content,
            # translated by the use element's x/y offset, carrying over
            # the use element's own style (fill/opacity etc. set on the
            # <use> itself, which is common in matplotlib's output)
            import copy
            new_el = copy.deepcopy(referenced)
            g_wrapper = ET.Element(f"{{{SVG_NS}}}g")
            g_wrapper.set("transform", f"translate({x} {y})")
            if style:
                g_wrapper.set("style", style)
            g_wrapper.append(new_el)

            parent.remove(child)
            parent.insert(i, g_wrapper)
            n_replaced += 1

    print(f"Inlined {n_replaced} <use> element(s), {n_missing} unresolved reference(s)")
    tree.write(output_path, xml_declaration=True, encoding="UTF-8")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python inline_svg_use_elements.py input.svg output.svg")
        sys.exit(1)
    inline_use_elements(sys.argv[1], sys.argv[2])
