"""The first reward task. Prompt is signature plus docstring; tests are the judge."""
PROMPT = '''def rev_string(s):
    """Return s reversed."""
'''
TESTS = '''
assert rev_string("abc") == "cba"
assert rev_string("") == ""
assert rev_string("a") == "a"
'''
