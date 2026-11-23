"""Conditional routing, in parity with nanoodle-js src/branches.mjs."""

from .errors import NanoodleError

PORTS = ("yes", "no")


def potentially_skipped_nodes(graph, links):
    """Nodes that may skip through a branch or legacy gate, for static preflight."""
    fed = {(link.to_node, link.to_port) for link in links}
    gates = {
        node.id for node in graph.nodes.values()
        if node.type == "decide" and node.fields.get("branch") is not True
        and (node.fields.get("gate") in (True, "true") or (node.id, "gate") in fed)
        and (node.fields.get("mode") == "yesno" or (node.id, "mode") in fed)
    }
    downstream = {}
    skipped = set()
    for link in links:
        source = graph.node(link.from_node)
        downstream.setdefault(link.from_node, set()).add(link.to_node)
        if (link.from_node in gates
                or (source.type == "decide" and source.fields.get("branch") is True
                    and link.from_port in PORTS)):
            skipped.add(link.to_node)
    pending = list(skipped)
    while pending:
        for target in downstream.get(pending.pop(), ()):
            if target not in skipped:
                skipped.add(target)
                pending.append(target)
    return skipped


def plan_branches(graph, order):
    """Return a per-run function selecting today's inbound links, or None for legacy graphs."""
    direct = set()
    for link in graph.links:
        source = graph.node(link.from_node)
        if source.type != "decide" or link.from_port not in PORTS:
            continue
        if source.fields.get("branch") is not True:
            raise NanoodleError("node %s: enable branching on Decide to wire %s"
                                % (source.id, link.from_port))
        direct.add(link)
    if not direct:
        return None

    inbound = {nid: graph.inbound(nid) for nid in order}
    selectors = {}
    for nid in order:
        links = inbound[nid]
        controls = ([link for link in links if link in direct]
                    or [link for link in links if link.from_node in selectors])
        if controls:
            selectors[nid] = controls
    skipped = set()

    def active_links(nid, out_for):
        def closed(link):
            if link.from_node in skipped:
                return True
            if link not in direct:
                return False
            branch = ((out_for(link.from_node) or {}).get("decision") or {}).get("branch")
            return branch in PORTS and branch != link.from_port

        controls = selectors.get(nid)
        if controls and all(closed(link) for link in controls):
            skipped.add(nid)
            return None
        return [link for link in inbound[nid] if not closed(link)]

    return active_links
