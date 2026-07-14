"""
To debug regex: https://www.debuggex.com/
"""
import pytest 
from tempfile import TemporaryFile, NamedTemporaryFile
from RCAEval.logparser import (
    find_json_bounds,
    mask_dict_values, 
    mask_dict_values_in_log,
)


@pytest.mark.parametrize("text, expected", [
    (
        "This is a json: {'a': 1}",
        ["{'a': 1}"]
    ),
    (
        "This is a json: {'a': 1, 'b': 2}",
        ["{'a': 1, 'b': 2}"]
    ),
    (
        "This is a json: {'a': 1, 'b': 2, 'c': 3}",
        ["{'a': 1, 'b': 2, 'c': 3}"]
    ),
    (
        "This is two dicts: {'a': 1} and {'b': 2}",
        ["{'a': 1}", "{'b': 2}"]
    ),
    (
        "This is a nested dict: {'a': 1, 'b': {'c': 2}}",
        ["{'a': 1, 'b': {'c': 2}}"]
    ),
    (
        "This is a very very complex nested dict with multiple data types (e.g., int, str, list, dict): {'a': 1, 'b': {'c': 2, 'd': [3, 4]}, 'e': 'string', 'f': [5, 6], 'g': {'h': 7}}",
        ["{'a': 1, 'b': {'c': 2, 'd': [3, 4]}, 'e': 'string', 'f': [5, 6], 'g': {'h': 7}}"]
    ),
])
def test_find_json_bound(text, expected):
    text = "This is a json: {'a': 1}"
    expected = ["{'a': 1}"]
    bounds = find_json_bounds(text)
    for i, (start, end) in enumerate(bounds):
        assert text[start:end] == expected[i]


@pytest.mark.parametrize("data, expected", [
    (
        {'id': 100, 'data': {'log-1': 'aaa', 10: [1, 2, 'a']}},
        {'id': '<*>', 'data': {'log-1': '<*>', 10: ['<*>', '<*>', '<*>']}}
    ),
    (
        {'id': 100, 'data': {'log-1': 'aaa', 10: [1, 2, 'a'], 'log-2': {'a': 1}}},
        {'id': '<*>', 'data': {'log-1': '<*>', 10: ['<*>', '<*>', '<*>'], 'log-2': {'a': '<*>'}}}
    ),
    (
        "abc",
        "<*>"
    )
])
def test_mask_dict_values(data, expected):
    assert mask_dict_values(data) == expected


@pytest.mark.parametrize("log, expected", [
    (
        "This is a log: {'id': 100, 'data': {'log-1': 'aaa', 'key-2': [1,2,3, 'a']}} with some values.",
        'This is a log: {"id": "<*>", "data": {"log-1": "<*>", "key-2": ["<*>", "<*>", "<*>", "<*>"]}} with some values.'
    ),
    (
        'POST to carts: items body: {"itemId":"819e1fbf-8b7e-4f6d-811f-693534916a8b","unitPrice":14}"',
        'POST to carts: items body: {"itemId": "<*>", "unitPrice": "<*>"}"',
    ),
    (   
        '{"id":"819e1fbf-8b7e-4f6d-811f-693534916a8b","name":"Figueroa","description":"enim officia aliqua excepteur esse deserunt quis aliquip nostrud anim","imageUrl":["/catalogue/images/WAT.jpg"],"price":14,"count":808,"tag":["formal","green","blue"]}',
        '{"id": "<*>", "name": "<*>", "description": "<*>", "imageUrl": ["<*>"], "price": "<*>", "count": "<*>", "tag": ["<*>", "<*>", "<*>"]}',
    )
])
def test_mask_dict_values_in_log(log, expected):
    assert mask_dict_values_in_log(log) == expected
