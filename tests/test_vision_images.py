"""Image preparation, in-memory store and sample loader tests."""
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import pypdfium2 as pdfium
from PIL import Image
from ragent_lab.vision import images as vi


def _enc(img, fmt):
    buf = io.BytesIO()
    img.save(buf, fmt)
    return buf.getvalue()


def _pdf(pages):
    doc = pdfium.PdfDocument.new()
    for _ in range(pages):
        doc.new_page(200, 100)
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def _open(png):
    return Image.open(io.BytesIO(png))


class ImagePrepTests(unittest.TestCase):
    def test_png_jpg_pdf_and_resize(self):
        p = vi.prepare_image(_enc(Image.new('RGB', (4000, 1000), 'red'), 'PNG'), 'a.png')
        self.assertEqual((p.width, p.height), (2048, 512))
        self.assertEqual(_open(p.png).size, (2048, 512))
        self.assertEqual(len(p.sha256), 64)
        self.assertEqual(p.source_name, 'a.png')

        j = vi.prepare_image(_enc(Image.new('CMYK', (50, 50), (0, 0, 0, 0)), 'JPEG'), 'c.jpg')
        self.assertEqual(_open(j.png).mode, 'RGB')

        rgba = Image.new('RGBA', (10, 10), (255, 0, 0, 0))
        r = vi.prepare_image(_enc(rgba, 'PNG'), 't.png')
        self.assertEqual(_open(r.png).convert('RGB').getpixel((5, 5)), (255, 255, 255))

        d = vi.prepare_image(_pdf(1), 'd.pdf')
        self.assertEqual(_open(d.png).format, 'PNG')
        self.assertGreater(d.width, 200)

    def test_rejections(self):
        with self.assertRaises(vi.ImageRejected) as cm:
            vi.prepare_image(_pdf(2), 'two.pdf')
        self.assertIn('单页', str(cm.exception))
        with self.assertRaises(vi.ImageRejected):
            vi.prepare_image(b'\x00\x01random bytes', 'x.png')
        with self.assertRaises(vi.ImageRejected):
            vi.prepare_image(b'0' * (15 * 1024 * 1024 + 1), 'big.png')
        ok = vi.prepare_image(_enc(Image.new('RGB', (20, 20), 'blue'), 'JPEG'), 'fake.png')
        self.assertEqual((ok.width, ok.height), (20, 20))


class StoreTests(unittest.TestCase):
    def _img(self, n):
        return vi.PreparedImage(id=f'i{n}', png=b'x', width=1, height=1, sha256='s', source_name='n')

    def test_store_ttl_and_capacity(self):
        store = vi.ImageStore()
        with patch('time.monotonic', return_value=1000.0):
            iid = store.put(self._img(0))
        with patch('time.monotonic', return_value=1000.0 + 7199):
            self.assertIsNotNone(store.get(iid))
        with patch('time.monotonic', return_value=1000.0 + 7201):
            self.assertIsNone(store.get(iid))

        store = vi.ImageStore()
        ids = [store.put(self._img(n)) for n in range(101)]
        self.assertIsNone(store.get(ids[0]))
        self.assertIsNotNone(store.get(ids[1]))
        self.assertIsNotNone(store.get(ids[100]))


class SampleTests(unittest.TestCase):
    def test_load_samples_from_tmpdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            good = root / 'good'
            good.mkdir()
            (good / 'pic.png').write_bytes(_enc(Image.new('RGB', (30, 20), 'white'), 'PNG'))
            (good / 'rubric.json').write_text(json.dumps({
                'id': 'good', 'name': 'Good', 'image': 'pic.png', 'rubric_status': 'approved',
                'items': [{'id': 'a', 'target': 't', 'expected': 'e'}]}), 'utf-8')
            bad = root / 'bad'
            bad.mkdir()
            (bad / 'rubric.json').write_text('{not json', 'utf-8')
            with self.assertLogs('ragent_lab.vision.images', level='WARNING'):
                samples = vi.load_samples(root)
            self.assertEqual([s.id for s in samples], ['good'])
            self.assertEqual(samples[0].rubric.rubric_status, 'approved')
            self.assertEqual(samples[0].image.width, 30)

            vi.reset_samples_cache()
            with patch.dict('os.environ', {'VISION_SAMPLES_DIR': str(root)}), \
                    self.assertLogs('ragent_lab.vision.images', level='WARNING'):
                self.assertEqual(vi.get_sample('good').name, 'Good')
                self.assertIsNone(vi.get_sample('nope'))
            vi.reset_samples_cache()
        self.assertEqual(vi.load_samples(Path(tmp) / 'missing'), [])


if __name__ == '__main__':
    unittest.main()
