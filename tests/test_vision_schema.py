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

    def test_geometry_cost_budgets(self):
        def text(n, h=5):
            return {'type': 'text', 'layer': 'TEXT', 'insert': [0, 0], 'height': h, 'content': 'x' * n}
        poly = lambda n: {'type': 'polyline', 'layer': 'OUTLINE', 'points': [[i % 100, 0] for i in range(n)]}
        cases = {
            'text 201 chars': [text(201)],
            'text height 1001': [text(5, 1001)],
            'dim text 101': [{'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear', 'p1': [0, 0],
                              'p2': [1, 0], 'location': [0, 1], 'text': 'x' * 101}],
            'polyline 2001 points': [poly(2001)],
            'vertices 50001': [poly(2000) for _ in range(25)] + [line()],
            'text chars 20001': [text(200) for _ in range(100)] + [text(1)],
        }
        for name, ents in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValidationError):
                    Geometry.model_validate(geom(ents))
        Geometry.model_validate(geom([text(200, 1000), poly(2000)]))
        Geometry.model_validate(geom([poly(2000) for _ in range(25)]))

    def test_polyline_bulge_bounded(self):
        def pl(b):
            return {'type': 'polyline', 'layer': 'OUTLINE', 'points': [[0, 0, b], [10, 0]]}
        for ok in (10, -10, 0.5):
            Geometry.model_validate(geom([pl(ok)]))
        for bad in (10.01, -11, float('inf'), float('nan')):
            with self.subTest(bad), self.assertRaises(ValidationError):
                Geometry.model_validate(geom([pl(bad)]))

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
