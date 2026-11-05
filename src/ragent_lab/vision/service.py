"""Vision tasks: understand, generate (redraw), score against a rubric, cross-model consensus."""
from __future__ import annotations

import base64
import json
import string

from pydantic import BaseModel, ValidationError

from . import prompts
from .images import PreparedImage
from .openrouter import CallResult, OpenRouterError, chat_json
from .registry import JUDGE_MODEL, get_model
from .schema import (ConsensusResult, Geometry, Judgement, JudgeResult, Rubric,
                     Understanding, model_json_schema_for)

_POINTS = {'correct': 1.0, 'uncertain': 0.5, 'wrong': 0.0}


def _meta(call: CallResult, model: dict) -> dict:
    return {
        'model': call.model, 'provider': call.provider, 'finish_reason': call.finish_reason,
        'usage': call.usage, 'cost': call.cost, 'elapsed_ms': call.elapsed_ms,
        'params': {'max_tokens': model.get('max_tokens'),
                   'temperature': model.get('temperature'),
                   'reasoning': model.get('reasoning')},
    }


def _validate(cls: type[BaseModel], content: dict) -> BaseModel:
    try:
        return cls.model_validate(content)
    except ValidationError:
        raise OpenRouterError('invalid_output', '输出不符合约定结构。') from None


def _image_part(image: PreparedImage) -> dict:
    b64 = base64.b64encode(image.png).decode('ascii')
    return {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}}


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


async def understand(api_key: str, model_id: str, image: PreparedImage) -> dict:
    model = get_model(model_id)
    messages = [{'role': 'user', 'content': [
        _image_part(image), {'type': 'text', 'text': prompts.UNDERSTAND}]}]
    call = await chat_json(api_key, model, messages, 'understanding',
                           model_json_schema_for(Understanding))
    parsed = _validate(Understanding, call.content)
    return {'model': model['id'], 'model_name': model['name'],
            'understanding': parsed.model_dump(), 'meta': _meta(call, model)}


async def generate(api_key: str, model_id: str, image: PreparedImage,
                   understanding: dict) -> dict:
    model = get_model(model_id)
    text = prompts.GENERATE + '\n\n理解 JSON：\n' + _dumps(understanding)
    messages = [{'role': 'user', 'content': [
        _image_part(image), {'type': 'text', 'text': text}]}]
    call = await chat_json(api_key, model, messages, 'geometry',
                           model_json_schema_for(Geometry))
    parsed = _validate(Geometry, call.content)
    return {'model': model['id'], 'model_name': model['name'],
            'geometry': parsed.model_dump(), 'meta': _meta(call, model)}


def compute_score(rubric: Rubric, judgements: list[Judgement]) -> tuple[float, dict]:
    counts = {'correct': 0, 'uncertain': 0, 'wrong': 0}
    by_id = {j.item_id: j.verdict for j in judgements}
    total = 0.0
    for item in rubric.items:
        verdict = by_id.get(item.id, 'wrong')  # not mentioned counts as wrong
        counts[verdict] += 1
        total += _POINTS[verdict]
    n = len(rubric.items)
    return (round(total / n * 100, 1) if n else 0.0), counts


async def score(api_key: str, rubric: Rubric, understanding: dict) -> dict:
    items = [i.model_dump() for i in rubric.items]
    text = (prompts.JUDGE + '\n\n评分细则：\n' + _dumps(items)
            + '\n\n被评理解 JSON：\n' + _dumps(understanding))
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]
    call = await chat_json(api_key, JUDGE_MODEL, messages, 'judge_result',
                           model_json_schema_for(JudgeResult))
    parsed: JudgeResult = _validate(JudgeResult, call.content)
    expected = [i.id for i in rubric.items]
    got = [j.item_id for j in parsed.judgements]
    if len(got) != len(set(got)) or set(got) != set(expected):
        raise OpenRouterError('invalid_output', '判定结果不完整，请重试评分。')
    order = {item_id: n for n, item_id in enumerate(expected)}
    judgements = sorted(parsed.judgements, key=lambda j: order[j.item_id])
    value, counts = compute_score(rubric, judgements)
    return {'score': value, 'counts': counts,
            'judgements': [j.model_dump() for j in judgements],
            'meta': _meta(call, JUDGE_MODEL)}


def _letter(raw: str, mapping: dict[str, str]) -> str:
    key = raw.strip().removeprefix('模型').strip().upper()
    if key not in mapping:
        raise OpenRouterError('invalid_output', '一致性结果引用了不存在的模型编号。')
    return mapping[key]


async def consensus(api_key: str, results: list[dict]) -> dict:
    if len(results) > len(string.ascii_uppercase):
        raise ValueError('too many models for consensus')
    letters = string.ascii_uppercase[:len(results)]
    mapping = {letter: r['model_name'] for letter, r in zip(letters, results)}
    blocks = []
    for letter, r in zip(letters, results):
        u = r['understanding']
        payload = {'views': u.get('views', []), 'dimensions': u.get('dimensions', [])}
        blocks.append(f'模型 {letter}：\n' + _dumps(payload))
    text = prompts.CONSENSUS + '\n\n' + '\n\n'.join(blocks)
    messages = [{'role': 'user', 'content': [{'type': 'text', 'text': text}]}]
    call = await chat_json(api_key, JUDGE_MODEL, messages, 'consensus_result',
                           model_json_schema_for(ConsensusResult))
    parsed: ConsensusResult = _validate(ConsensusResult, call.content)
    for group in parsed.groups:
        for interp in group.interpretations:
            interp.models = [_letter(m, mapping) for m in interp.models]
    for single in parsed.singletons:
        single.models = [_letter(m, mapping) for m in single.models]
    return {**parsed.model_dump(), 'meta': _meta(call, JUDGE_MODEL)}
