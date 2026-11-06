"""Geometry rendering (DXF / SVG / PDF) and dimension cross-check tests."""
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import ezdxf
from ragent_lab.vision import render
from ragent_lab.vision.schema import Geometry, Understanding


def _geometry(radius=50.0, extra=()):
    return Geometry.model_validate({
        'units': 'mm',
        'sheet': {'width': 297, 'height': 210},
        'entities': [
            {'type': 'circle', 'layer': 'OUTLINE', 'center': [150, 100], 'radius': radius,
             'dimension_ref': 'd1'},
            {'type': 'line', 'layer': 'CENTER', 'start': [90, 100], 'end': [210, 100]},
            {'type': 'text', 'layer': 'TEXT', 'insert': [20, 20], 'height': 5,
             'content': '层压方向'},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'diameter',
             'p1': [150 - radius, 100], 'p2': [150 + radius, 100],
             'location': [150 + radius + 20, 130]},
            *extra,
        ],
    })


def _understanding(value=300.0):
    return Understanding.model_validate({
        'part': {'name': '法兰'},
        'views': [{'id': 'v1', 'type': 'main', 'name': '主视图'}],
        'dimensions': [{'id': 'd1', 'label': 'Φ300', 'value': value, 'kind': 'diameter'}],
    })


class DxfTest(unittest.TestCase):
    def test_dxf_roundtrip_layers_and_text(self):
        doc = render.build_doc(_geometry())
        data = render.to_dxf(doc)
        self.assertIsInstance(data, bytes)
        back = ezdxf.read(io.StringIO(data.decode()))
        msp = back.modelspace()
        circles = msp.query('CIRCLE')
        self.assertEqual(len(circles), 1)
        self.assertAlmostEqual(circles[0].dxf.radius, 50.0)
        texts = [e.dxf.text for e in msp.query('TEXT')]
        self.assertIn('层压方向', texts)
        self.assertEqual(back.layers.get('CENTER').dxf.linetype, 'CENTER')
        self.assertEqual(back.layers.get('HIDDEN').dxf.linetype, 'HIDDEN')
        self.assertGreaterEqual(len(msp.query('DIMENSION')), 1)
        self.assertIn('CJK', back.styles)


class SvgPdfTest(unittest.TestCase):
    def test_svg_and_pdf_render(self):
        geometry = _geometry()
        doc = render.build_doc(geometry)
        svg = render.to_svg(doc)
        self.assertTrue(svg.lstrip().startswith('<svg'), svg[:80])
        self.assertGreater(len(svg), 1000)
        self.assertTrue('层压方向' in svg or '<path' in svg)
        pdf = render.to_pdf(doc, geometry.sheet)
        self.assertTrue(pdf.startswith(b'%PDF'))


class CheckDimensionsTest(unittest.TestCase):
    def test_check_dimensions(self):
        self.assertEqual(render.check_dimensions(_geometry(150), _understanding(300)), [])
        warnings = render.check_dimensions(_geometry(149), _understanding(300))
        # circle (2r = 298) and the diameter DimensionE have no ref; only the circle is checked
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].expected, 300)
        self.assertAlmostEqual(warnings[0].actual, 298)
        self.assertEqual(warnings[0].dimension_id, 'd1')
        self.assertEqual(warnings[0].entity_index, 0)

        bad = _geometry(150, extra=[{'type': 'line', 'layer': 'OUTLINE', 'start': [0, 0],
                                     'end': [10, 0], 'dimension_ref': 'nope'}])
        warnings = render.check_dimensions(bad, _understanding(300))
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].dimension_id, 'nope')

    def test_check_dimension_entities_and_tolerance(self):
        u = Understanding.model_validate({
            'part': {'name': 'x'}, 'views': [],
            'dimensions': [
                {'id': 'L', 'value': 100, 'kind': 'linear'},
                {'id': 'R', 'value': 20, 'kind': 'radius'},
                {'id': 'N', 'value': None, 'kind': 'linear'},
            ],
        })
        g = Geometry.model_validate({'units': 'mm', 'sheet': {'width': 300, 'height': 200},
                                     'entities': [
            # linear, horizontal projection of (0,0)-(100.4,50) is 100.4 -> within 0.5
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear', 'p1': [0, 0],
             'p2': [100.4, 50], 'location': [50, -10], 'dimension_ref': 'L'},
            # aligned: straight distance 111.8 -> warning
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'aligned', 'p1': [0, 0],
             'p2': [100, 50], 'location': [50, 40], 'dimension_ref': 'L'},
            {'type': 'arc', 'layer': 'OUTLINE', 'center': [0, 0], 'radius': 20,
             'start_angle': 0, 'end_angle': 90, 'dimension_ref': 'R'},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'radius', 'p1': [0, 0],
             'p2': [0, 21], 'location': [10, 30], 'dimension_ref': 'R'},
            {'type': 'line', 'layer': 'OUTLINE', 'start': [0, 0], 'end': [5, 0],
             'dimension_ref': 'N'},
        ]})
        warnings = render.check_dimensions(g, u)
        self.assertEqual([w.entity_index for w in warnings], [1, 3])


class RobustnessTest(unittest.TestCase):
    def test_bad_entity_skipped_not_raised(self):
        g = _geometry(extra=[{'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear',
                              'p1': [10, 10], 'p2': [10, 10], 'location': [10, 20]}])
        result = render.render_all(g, _understanding(300))
        self.assertTrue(result['skipped'])
        self.assertTrue(result['svg'].lstrip().startswith('<svg'))
        self.assertIsInstance(result['warnings'], list)
        self.assertIsInstance(result['warnings'][0], dict)

    def test_all_entity_kinds_build(self):
        g = Geometry.model_validate({'units': 'mm', 'sheet': {'width': 420, 'height': 297},
                                     'entities': [
            {'type': 'arc', 'layer': 'OUTLINE', 'center': [100, 100], 'radius': 30,
             'start_angle': 0, 'end_angle': 135},
            {'type': 'polyline', 'layer': 'HIDDEN', 'points': [[0, 0], [50, 0, 0.5], [50, 50]],
             'closed': True},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear', 'p1': [0, 0],
             'p2': [50, 0], 'location': [25, -15]},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'aligned', 'p1': [0, 0],
             'p2': [50, 50], 'location': [10, 40], 'text': '约71'},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'radius', 'p1': [100, 100],
             'p2': [130, 100], 'location': [100, 100]},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'diameter', 'p1': [70, 100],
             'p2': [130, 100], 'location': [150, 150]},
        ]})
        doc = render.build_doc(g)
        self.assertEqual(render.skipped_entities(doc), [])
        back = ezdxf.read(io.StringIO(render.to_dxf(doc).decode()))
        msp = back.modelspace()
        self.assertEqual(len(msp.query('DIMENSION')), 4)
        self.assertEqual(msp.query('LWPOLYLINE')[0].dxf.layer, 'HIDDEN')
        self.assertIn('约71', [d.dxf.text for d in msp.query('DIMENSION')])

    def test_failed_entity_leaves_no_partial_entity(self):
        target = 'ezdxf.entities.dimstyleoverride.DimStyleOverride.render'
        with patch(target, side_effect=RuntimeError('boom')):
            doc = render.build_doc(_geometry())
        self.assertEqual(len(doc.modelspace().query('DIMENSION')), 0)
        self.assertEqual(len(doc.modelspace().query('CIRCLE')), 1)
        self.assertIn('boom', render.skipped_entities(doc)[0])

    def test_render_all_without_understanding(self):
        result = render.render_all(_geometry(), None)
        self.assertEqual(result['warnings'], [])
        self.assertEqual(result['skipped'], [])


if __name__ == '__main__':
    unittest.main()
