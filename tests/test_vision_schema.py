"""Vision schema tests: understanding, geometry, rubric validation."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pydantic import ValidationError
from ragent_lab.vision.schema import (
    Geometry, Understanding, model_json_schema_for,
)


def geom(entities, **kw):
    data = {'units': 'mm', 'sheet': {'width': 420, 'height': 297},
            'entities': entities}
    data.update(kw)
    return data


def line(**kw):
    e = {'type': 'line', 'layer': 'OUTLINE', 'start': [0, 0], 'end': [10, 0]}
    e.update(kw)
    return e


class UnderstandingTests(unittest.TestCase):
    def test_understanding_accepts_minimal_and_ignores_extra(self):
        u = Understanding.model_validate({
            'part': {'name': '支架', 'bogus': 1},
            'views': [{'id': 'v1', 'type': 'main', 'name': '主视图'}],
            'dimensions': [], 'extra_top': 'x'})
        self.assertEqual(u.features, [])
        self.assertEqual(u.relations, [])
        self.assertEqual(u.derived, [])
        self.assertEqual(u.unresolved, [])
        self.assertFalse(hasattr(u, 'extra_top'))

    def test_understanding_rejects_unknown_view_type(self):
        with self.assertRaises(ValidationError):
            Understanding.model_validate({
                'part': {'name': 'p'},
                'views': [{'id': 'v1', 'type': 'isometric', 'name': 'x'}],
                'dimensions': []})


class GeometryTests(unittest.TestCase):
    def test_geometry_rejects_degenerate(self):
        cases = {
            'radius 0': [{'type': 'circle', 'layer': 'OUTLINE', 'center': [0, 0], 'radius': 0}],
            'radius -1': [{'type': 'circle', 'layer': 'OUTLINE', 'center': [0, 0], 'radius': -1}],
            'radius nan': [{'type': 'circle', 'layer': 'OUTLINE', 'center': [0, 0], 'radius': float('nan')}],
            'inf coord': [line(end=[float('inf'), 0])],
            'coord 10001': [line(end=[10001, 0])],
            'one-point polyline': [{'type': 'polyline', 'layer': 'OUTLINE', 'points': [[0, 0]], 'closed': False}],
            'zero-sweep arc': [{'type': 'arc', 'layer': 'OUTLINE', 'center': [0, 0], 'radius': 5,
                                'start_angle': 30, 'end_angle': 30}],
            '5001 entities': [line() for _ in range(5001)],
        }
        for name, ents in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValidationError):
                    Geometry.model_validate(geom(ents))
        with self.assertRaises(ValidationError):
            Geometry.model_validate(geom([line()], units='inch'))

    def test_geometry_accepts_all_entity_types(self):
        ents = [
            line(dimension_ref='d1'),
            {'type': 'circle', 'layer': 'CENTER', 'center': [5, 5], 'radius': 2},
            {'type': 'arc', 'layer': 'HIDDEN', 'center': [0, 0], 'radius': 5,
             'start_angle': 0, 'end_angle': 90},
            {'type': 'polyline', 'layer': 'OUTLINE', 'points': [[0, 0], [5, 0, 0.5], [5, 5]], 'closed': True},
            {'type': 'text', 'layer': 'TEXT', 'insert': [1, 1], 'height': 3.5, 'content': 'A'},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear',
             'p1': [0, 0], 'p2': [0, 0], 'location': [0, 5]},
        ]
        g = Geometry.model_validate(geom(ents))
        self.assertEqual(len(g.entities), 6)
        self.assertEqual(g.entities[0].dimension_ref, 'd1')
        self.assertEqual(g.entities[4].rotation, 0)
        self.assertEqual(g.entities[5].angle, 0)
        self.assertIsNone(g.entities[5].text)

    def test_json_schema_has_no_refs(self):
        for cls in (Understanding, Geometry):
            with self.subTest(cls.__name__):
                dumped = json.dumps(model_json_schema_for(cls))
                self.assertNotIn('$ref', dumped)
                self.assertNotIn('#/$defs', dumped)


if __name__ == '__main__':
    unittest.main()
