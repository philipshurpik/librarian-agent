import inspect
import json
from types import SimpleNamespace

from librarian.agent import loop, tools


def message(content=None, tool_calls=None):
    return SimpleNamespace(content=content, tool_calls=tool_calls)


def tool_call(call_id, name, **args):
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=json.dumps(args)))


class FakeOpenAI:
    def __init__(self, script):
        self.script, self.requests = list(script), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        self.requests.append({**kwargs, 'messages': [*kwargs['messages']]})  # snapshot: the loop mutates its list
        return SimpleNamespace(choices=[SimpleNamespace(message=self.script.pop(0))])


async def run_agent(monkeypatch, script, user='find me a kafka book'):
    fake = FakeOpenAI(script)
    monkeypatch.setattr(loop, '_client', lambda: fake)
    produced = await loop.run([{'role': 'user', 'content': user}])
    return fake, produced


async def test_plain_answer_needs_no_tools(monkeypatch):
    fake, produced = await run_agent(monkeypatch, [message(content='Hello! How can I help?')])

    assert produced == [{'role': 'assistant', 'content': 'Hello! How can I help?'}]
    assert fake.requests[0]['messages'][0]['role'] == 'system'
    assert fake.requests[0]['tools'] == tools.TOOL_SCHEMAS


async def test_tool_round_trip(monkeypatch):
    async def fake_search(query):
        return [{'book_id': 'bk-1', 'title': 'Kafka Guide'}]

    monkeypatch.setattr(tools, 'search_catalog', fake_search)
    script = [
        message(tool_calls=[tool_call('call-1', 'search_catalog', query='kafka')]),
        message(content='We have the Kafka Guide.'),
    ]
    fake, produced = await run_agent(monkeypatch, script)

    assert [m['role'] for m in produced] == ['assistant', 'tool', 'assistant']
    assert produced[1]['tool_call_id'] == 'call-1'
    assert json.loads(produced[1]['content']) == [{'book_id': 'bk-1', 'title': 'Kafka Guide'}]
    assert fake.requests[1]['messages'][-1] == produced[1]  # tool result went back to the model
    assert produced[2]['content'] == 'We have the Kafka Guide.'


async def test_tool_errors_feed_back_to_model(monkeypatch):
    async def boom(query):
        raise RuntimeError('qdrant down')

    monkeypatch.setattr(tools, 'search_catalog', boom)
    script = [
        message(tool_calls=[tool_call('c-1', 'search_catalog', query='kafka'), tool_call('c-2', 'no_such_tool')]),
        message(content='Sorry, having trouble searching right now.'),
    ]
    _, produced = await run_agent(monkeypatch, script)

    assert json.loads(produced[1]['content']) == {'error': 'RuntimeError: qdrant down'}
    assert json.loads(produced[2]['content']) == {'error': 'unknown tool: no_such_tool'}
    assert produced[3]['content'] == 'Sorry, having trouble searching right now.'


async def test_max_rounds_forces_text_answer(monkeypatch):
    async def fake_search(query):
        return []

    monkeypatch.setattr(tools, 'search_catalog', fake_search)
    searching = message(tool_calls=[tool_call('c', 'search_catalog', query='x')])
    script = [searching] * loop._MAX_TOOL_ROUNDS + [message(content='Best I could find.')]
    fake, produced = await run_agent(monkeypatch, script)

    assert produced[-1] == {'role': 'assistant', 'content': 'Best I could find.'}
    assert 'tools' not in fake.requests[-1]  # final call forbids further tool use


def test_schemas_match_tool_signatures():
    for schema in tools.TOOL_SCHEMAS:
        fn = schema['function']
        params = inspect.signature(getattr(tools, fn['name'])).parameters
        assert set(fn['parameters']['properties']) <= set(params)
        required = {name for name, p in params.items() if p.default is inspect.Parameter.empty}
        assert set(fn['parameters']['required']) == required
