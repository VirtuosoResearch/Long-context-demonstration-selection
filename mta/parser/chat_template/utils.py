PARSER_TEST_MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Search for information about Python."},
    {
        "role": "assistant",
        "content": "I'll search for that.",
        "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Python programming"}'}}],
    },
    {"role": "user", "content": "What about Java?"},
    {
        "role": "assistant",
        "content": "Let me search for Java information.",
        "tool_calls": [{"function": {"name": "search", "arguments": '{"query": "Java programming"}'}}],
    },
]
