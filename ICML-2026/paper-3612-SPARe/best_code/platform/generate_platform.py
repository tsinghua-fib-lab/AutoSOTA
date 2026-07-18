# Copyright (c) 2026, University of Florida. All rights reserved.

# This file is part of the simulation software package developed by
# the team members of Prof. Xiaoyi Lu's group at University of Florida.

# For detailed copyright and licensing information, please refer to the license
# file LICENSE in the top level directory.

#!/usr/bin/env python3
import argparse
import fnmatch
import itertools
import json
from xml.etree.ElementTree import Element, SubElement, ElementTree, tostring

def build_platform(cfg):
    platform = Element("platform", {"version": "4.1"})
    zone = SubElement(platform, "zone", {"id": "AS0", "routing": "Full"})

    # Hosts
    workers = expand_workers(cfg)
    for w in workers:
        host = SubElement(zone, "host", {"id": w["id"], "speed": w["speed"]})
        SubElement(
            host,
            "disk",
            {
                "id": "disk0",
                "read_bw": w["disk_read_bw"],
                "write_bw": w["disk_write_bw"],
            },
        )

    # Links
    for lname, lcfg in cfg["links"].items():
        SubElement(
            zone,
            "link",
            {"id": lname, "bandwidth": lcfg["bandwidth"], "latency": lcfg["latency"]},
        )

    by_id = {w["id"]: w for w in workers}
    routes = cfg.get("routes")
    if routes:
        add_routes(zone, routes)
    else:
        build_routes_from_routing(zone, by_id, cfg.get("routing", {}))

    return platform


def normalize_range(value):
    if isinstance(value, list):
        return value
    if isinstance(value, int):
        return [value]
    if isinstance(value, str) and ".." in value:
        start, end = value.split("..", 1)
        return list(range(int(start), int(end) + 1))
    return [value]


def format_value(val, values):
    if isinstance(val, str):
        return val.format(**values)
    return val


def expand_spec(spec):
    ranges = spec.get("ranges", {})
    keys = sorted(ranges.keys())
    values_list = [normalize_range(ranges[k]) for k in keys]
    for combo in itertools.product(*values_list):
        values = {k: combo[i] for i, k in enumerate(keys)}
        values_str = {k: str(v) for k, v in values.items()}
        worker = {
            "id": format_value(spec["id"], values_str),
            "speed": format_value(spec["speed"], values_str),
            "disk_read_bw": format_value(spec["disk_read_bw"], values_str),
            "disk_write_bw": format_value(spec["disk_write_bw"], values_str),
        }
        levels = spec.get("levels", {})
        if levels:
            worker["levels"] = {k: format_value(v, values_str) for k, v in levels.items()}
        yield worker


def expand_workers(cfg):
    workers = []
    if "workers" in cfg:
        workers.extend(cfg["workers"])
    if "workers_range" in cfg:
        for spec in cfg["workers_range"]:
            workers.extend(list(expand_spec(spec)))
    if "workers_template" in cfg:
        tmpl = cfg["workers_template"]
        tmpl_specs = tmpl if isinstance(tmpl, list) else [tmpl]
        for spec in tmpl_specs:
            workers.extend(list(expand_spec(spec)))
    overrides = cfg.get("overrides", [])
    if overrides:
        for w in workers:
            for o in overrides:
                if fnmatch.fnmatch(w["id"], o["id"]):
                    w.update({k: v for k, v in o.items() if k != "id"})
    return workers


def add_route(zone, src, dst, links):
    route = SubElement(zone, "route", {"src": src, "dst": dst})
    for link in links:
        SubElement(route, "link_ctn", {"id": link})


def add_routes(zone, routes):
    for r in routes:
        links = r.get("links") or [r["link"]]
        add_route(zone, r["src"], r["dst"], links)


def pick_link_by_level(by_id, routing, src, dst):
    if "levels" in routing:
        levels = routing["levels"]
        src_levels = by_id[src].get("levels", {})
        dst_levels = by_id[dst].get("levels", {})
        for level in levels:
            lname = level["name"]
            if src_levels.get(lname) == dst_levels.get(lname):
                return level["link"]
        return routing.get("default_link")
    if "intra_link" in routing and "inter_link" in routing:
        same_group = by_id[src].get("group") == by_id[dst].get("group")
        return routing["intra_link"] if same_group else routing["inter_link"]
    return None


def build_routes_from_routing(zone, by_id, routing):
    pairs = routing.get("pairs")
    if pairs:
        for p in pairs:
            src = p["src"]
            dst = p["dst"]
            links = p.get("links")
            if not links:
                link = p.get("link") or pick_link_by_level(by_id, routing, src, dst)
                if not link:
                    continue
                links = [link]
        add_route(zone, src, dst, links)
        return

    ids = sorted(by_id.keys())
    for i, src in enumerate(ids):
        for dst in ids[i + 1 :]:
            link = pick_link_by_level(by_id, routing, src, dst)
            if not link:
                continue
            add_route(zone, src, dst, [link])


def write_xml(root, path):
    indent_xml(root)
    with open(path, "w", encoding="utf-8") as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n")
        f.write("<!DOCTYPE platform SYSTEM \"https://simgrid.org/simgrid.dtd\">\n")
        f.write(tostring(root, encoding="unicode"))


def indent_xml(elem, level=0):
    indent = "\n" + "  " * level
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = indent + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = indent


def example_config():
    return {
        "workers": [
            {
                "id": "worker0",
                "speed": "200Gf",
                "disk_read_bw": "2GBps",
                "disk_write_bw": "1GBps",
                "levels": {"node": 0, "rack": 0, "cluster": 0},
            },
            {
                "id": "worker1",
                "speed": "180Gf",
                "disk_read_bw": "1.5GBps",
                "disk_write_bw": "1GBps",
                "levels": {"node": 0, "rack": 0, "cluster": 0},
            },
            {
                "id": "worker2",
                "speed": "220Gf",
                "disk_read_bw": "2.2GBps",
                "disk_write_bw": "1.2GBps",
                "levels": {"node": 1, "rack": 0, "cluster": 0},
            },
            {
                "id": "worker3",
                "speed": "210Gf",
                "disk_read_bw": "2GBps",
                "disk_write_bw": "1GBps",
                "levels": {"node": 1, "rack": 1, "cluster": 0},
            },
        ],
        "links": {
            "nvlink": {"bandwidth": "300GBps", "latency": "5us"},
            "infiniband": {"bandwidth": "200GBps", "latency": "10us"},
            "slingshot": {"bandwidth": "100GBps", "latency": "50us"},
        },
        "routing": {
            "levels": [
                {"name": "node", "link": "nvlink"},
                {"name": "rack", "link": "infiniband"},
                {"name": "cluster", "link": "slingshot"},
            ],
            "pairs": [
                {"src": "worker0", "dst": "worker1"},
                {"src": "worker0", "dst": "worker2"},
                {"src": "worker2", "dst": "worker3"},
            ],
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Generate SimGrid platform.xml")
    parser.add_argument("--config", required=False, help="Path to config JSON")
    parser.add_argument("--out", default="platform.xml", help="Output XML path")
    parser.add_argument("--example", action="store_true", help="Print example config JSON")
    args = parser.parse_args()

    if args.example:
        print(json.dumps(example_config(), indent=2))
        return

    if not args.config:
        raise SystemExit("Missing --config. Use --example to see the JSON format.")

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    root = build_platform(cfg)
    write_xml(root, args.out)


if __name__ == "__main__":
    main()
