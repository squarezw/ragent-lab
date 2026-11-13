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
    def test_dxf_dimensions_use_mm_style(self):
        """R128 must read 128 in CAD: no style may scale lengths (ezdxf's own use x100)."""
        g = Geometry.model_validate({'units': 'mm', 'sheet': {'width': 400, 'height': 300},
                                     'entities': [
            {'type': 'circle', 'layer': 'OUTLINE', 'center': [150, 150], 'radius': 128},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'radius', 'p1': [150, 150],
             'p2': [278, 150], 'location': [300, 200]},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear', 'p1': [0, 0],
             'p2': [390, 0], 'location': [195, -20]},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'diameter', 'p1': [131, 150],
             'p2': [169, 150], 'location': [150, 150]}]})
        doc = ezdxf.read(io.StringIO(render.to_dxf(render.build_doc(g)).decode('utf-8')))
        self.assertEqual(doc.header['$INSUNITS'], 4)  # millimetres
        self.assertEqual(doc.header['$DIMSTYLE'], render.DIM_STYLE)
        self.assertEqual({ds.dxf.get('dimlfac', 1.0) for ds in doc.dimstyles}, {1.0})
        dims = list(doc.modelspace().query('DIMENSION'))
        self.assertEqual(len(dims), 3)
        texts = []
        for d in dims:
            self.assertEqual(d.dxf.dimstyle, render.DIM_STYLE)
            block = doc.blocks.get(d.dxf.geometry)
            texts += [t.plain_text() for t in block.query('MTEXT')]
        self.assertEqual(sorted(texts), ['390', 'R128', 'Ø38'])

    def test_preview_is_dark_and_white_paper_is_available(self):
        g = Geometry.model_validate({'units': 'mm', 'sheet': {'width': 100, 'height': 80},
                                     'entities': [{'type': 'line', 'layer': 'OUTLINE',
                                                   'start': [10, 10], 'end': [90, 10]}]})
        doc = render.build_doc(g)
        dark = render.to_svg(doc, g.sheet)
        light = render.to_svg(doc, g.sheet, dark=False)
        self.assertRegex(dark, r'<rect fill="#000000"')
        self.assertRegex(light, r'<rect fill="#ffffff"')
        # OUTLINE (ACI 7) flips to stay visible: white on black, black on white.
        self.assertIn('stroke: #ffffff', dark)
        self.assertIn('stroke: #000000', light)

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


class DimensionLabelTests(unittest.TestCase):
    """Every dimension must measure the number in its label, with or without an understanding."""

    def test_label_values(self):
        cases = {'390': (390.0, 'linear'), '20.5': (20.5, 'linear'), 'R108': (108.0, 'radius'),
                 'Ø20': (20.0, 'diameter'), '2-Ø19': (19.0, 'diameter'), '4×Ø8': (8.0, 'diameter'),
                 '60±0.1': (60.0, 'linear'), 'Ø20H7': (20.0, 'diameter'), '%%c12': (12.0, 'diameter'),
                 '2-R5': (5.0, 'radius')}
        for text, want in cases.items():
            with self.subTest(text):
                self.assertEqual(render.label_value(text), want)
        for text in ('', None, '<>', '45°', 'C2', '20+20', 'M10', '待确认'):
            with self.subTest(text):
                self.assertIsNone(render.label_value(text))

    def _dims(self, *dims):
        return Geometry.model_validate({'units': 'mm', 'sheet': {'width': 500, 'height': 400},
                                        'entities': list(dims)})

    def test_mismatch_is_flagged_whatever_the_process(self):
        g = self._dims(
            # labelled 30 but spans 15 (the bracket's misplaced hole row)
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear', 'p1': [420, 295],
             'p2': [420, 280], 'location': [445, 287], 'angle': 90, 'text': '30'},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear', 'p1': [0, 0],
             'p2': [390, 0], 'location': [195, -20], 'text': '390'},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'radius', 'p1': [100, 100],
             'p2': [208, 100], 'location': [150, 150], 'text': 'R108'},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'diameter', 'p1': [0, 50],
             'p2': [19.2, 50], 'location': [9, 60], 'text': '2-Ø19'},
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'radius', 'p1': [0, 0],
             'p2': [10, 0], 'location': [5, 5], 'text': 'Ø20'},  # radius 10 = diameter 20
            {'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear', 'p1': [0, 0],
             'p2': [40, 0], 'location': [20, -5], 'text': '20+20'})
        ws = render.check_dimension_labels(g)
        self.assertEqual([(w.entity_index, w.dimension_id, w.expected, w.actual) for w in ws],
                         [(0, '30', 30.0, 15.0)])
        self.assertIn('标注「30」', ws[0].message)
        out = render.render_all(g, None)  # no understanding: still checked
        self.assertEqual([w['dimension_id'] for w in out['warnings']], ['30'])

    def test_lines_are_not_measured_against_dimensions(self):
        u = Understanding.model_validate({'part': {'name': 'x'}, 'views': [], 'dimensions': [
            {'id': 'd195', 'label': '195', 'value': 195, 'kind': 'linear'}]})
        g = self._dims({'type': 'line', 'layer': 'OUTLINE', 'start': [195, 0], 'end': [195, 60],
                        'dimension_ref': 'd195'})
        self.assertEqual(render.check_dimensions(g, u), [])

    def test_one_warning_per_dimension(self):
        u = Understanding.model_validate({'part': {'name': 'x'}, 'views': [], 'dimensions': [
            {'id': 'd30', 'label': '30', 'value': 30, 'kind': 'linear'}]})
        g = self._dims({'type': 'dimension', 'layer': 'DIM', 'dimtype': 'linear', 'p1': [0, 0],
                        'p2': [15, 0], 'location': [5, -5], 'text': '30', 'dimension_ref': 'd30'})
        out = render.render_all(g, u)
        self.assertEqual(len(out['warnings']), 1)
        self.assertEqual(out['warnings'][0]['message'], '尺寸 30 与图形不符')
