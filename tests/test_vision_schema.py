"""Vision schema tests: understanding, geometry, rubric validation."""
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pydantic import ValidationError
from ragent_lab.vision.schema import (
    ConsensusResult, Geometry, JudgeResult, Understanding, model_json_schema_for,
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

    def test_understanding_maps_unknown_view_type_to_other(self):
        u = Understanding.model_validate({
            'part': {'name': 'p'},
            'views': [{'id': 'v1', 'type': 'isometric', 'name': 'x'}],
            'dimensions': []})
        self.assertEqual(u.views[0].type, 'other')


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

    def test_json_schema_has_no_optional_properties(self):
        # Anthropic rejects response schemas with more than 24 optional properties.
        def optional(node):
            if isinstance(node, dict):
                n = 0
                if isinstance(node.get('properties'), dict):
                    n = len(set(node['properties']) - set(node.get('required', [])))
                return n + sum(optional(v) for v in node.values())
            if isinstance(node, list):
                return sum(optional(v) for v in node)
            return 0

        for cls in (Understanding, Geometry, JudgeResult, ConsensusResult):
            with self.subTest(cls.__name__):
                self.assertEqual(optional(model_json_schema_for(cls)), 0)
        # Defaults still apply when a model leaves a field out.
        self.assertEqual(ConsensusResult.model_validate({'groups': []}).singletons, [])


if __name__ == '__main__':
    unittest.main()


class TolerantUnderstandingTests(unittest.TestCase):
    """Small slips in a model's reply are repaired; only the overall shape is strict."""

    def test_slips_are_repaired(self):
        from ragent_lab.vision.schema import Understanding
        u = Understanding.model_validate({
            'part': None,
            'views': [{'id': 'v1', 'type': 'Isometric', 'name': None, 'bbox': [0, 0, 1]},
                      {'id': 'v2', 'type': 'Section', 'bbox': ['0', 0.1, 0.5, '0.6']}, 'junk'],
            'features': [{'id': 'f1', 'type': 'rectangle', 'center': ['10', 20], 'radius': 'R9.5',
                          'points': [[0, 0], 'x', [1]], 'visible': 'false'}],
            'dimensions': [{'id': 'd1', 'label': 'Ø20', 'value': 'Ø20', 'kind': 'Diameter',
                            'feature_ids': 'f1', 'confidence': '90', 'uncertainty': None},
                           {'id': 'd2', 'value': None, 'kind': 'chamfer', 'confidence': None}],
            'relations': '同心', 'unresolved': None, 'derived': [{'value': 30}],
        })
        self.assertEqual(u.part.name, '')
        self.assertEqual([(v.type, v.bbox) for v in u.views],
                         [('other', None), ('section', [0.0, 0.1, 0.5, 0.6])])
        f = u.features[0]
        self.assertEqual((f.type, f.center, f.radius, f.points, f.visible),
                         ('other', [10.0, 20.0], 9.5, [[0.0, 0.0]], False))
        d1, d2 = u.dimensions
        self.assertEqual((d1.value, d1.kind, d1.feature_ids, d1.confidence, d1.uncertainty),
                         (20.0, 'diameter', ['f1'], 0.9, ''))
        self.assertEqual((d2.value, d2.kind, d2.confidence), (None, 'other', 0.5))
        self.assertEqual((u.relations, u.unresolved, u.derived[0].value), (['同心'], [], '30'))

    def test_shape_stays_strict(self):
        from pydantic import ValidationError
        from ragent_lab.vision.schema import GeneralUnderstanding, Understanding
        with self.assertRaises(ValidationError):
            Understanding.model_validate({'summary': '一张图'})  # general reply, not a drawing
        with self.assertRaises(ValidationError):
            GeneralUnderstanding.model_validate({'views': [], 'dimensions': []})
        g = GeneralUnderstanding.model_validate({'summary': ['a', 'b'], 'texts': 'x',
                                                 'data': [{'name': 'Q3', 'value': 120}]})
        self.assertEqual((g.summary, g.texts, g.data[0].value), ('a；b', ['x'], '120'))

    def test_schema_sent_to_models_keeps_the_categories(self):
        import json
        from ragent_lab.vision.schema import Understanding, model_json_schema_for
        text = json.dumps(model_json_schema_for(Understanding))
        for word in ('"main"', '"hidden_line"', '"diameter"', '"dimensions"'):
            self.assertIn(word, text)
