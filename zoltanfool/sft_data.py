"""Fix for the zero-pass problem: teach the format 'signature + docstring, then body' from your corpus."""
import ast


def function_examples(source: str, max_chars: int = 400) -> list[tuple[str, str]]:
    """(prompt, body) pairs from top-level functions that have a docstring and a body after it."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    lines = source.splitlines(keepends=True)
    pairs = []
    for node in tree.body:                              # top level only: no methods that need self
        if not isinstance(node, ast.FunctionDef) or not ast.get_docstring(node):
            continue
        doc = node.body[0]
        if len(node.body) < 2 or node.end_lineno is None or doc.end_lineno is None:
            continue
        prompt = "".join(lines[node.lineno - 1:doc.end_lineno])
        body = "".join(lines[doc.end_lineno:node.end_lineno])
        if len(prompt) + len(body) <= max_chars:
            pairs.append((prompt, body))
    return pairs
