import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

import vectorize


class InferenceIntegrityTests(unittest.TestCase):
    def scan(self, response):
        session = MagicMock()
        session.post.return_value.status_code = 200
        session.post.return_value.json.return_value = response
        with patch.object(vectorize, '_get_http_session', return_value=session), patch.object(vectorize.logger, 'warning'):
            result = vectorize.scan_satellite_image_roboflow(Image.new('RGB', (32, 32)), api_key='test',
                tile_size=16, overlap=0, target_size=(32, 32))
        return result, session.post.call_count

    def test_http_200_error_payload_is_not_counted_as_a_successful_empty_tile(self):
        result, calls = self.scan({'error': 'model unavailable'})
        self.assertEqual(result['tiles'], 4)
        self.assertEqual(result['failed_tiles'], 4)
        self.assertEqual(calls, 8)

    def test_valid_empty_detection_list_is_a_successful_tile(self):
        result, calls = self.scan({'predictions': [], 'image': {'width': 16, 'height': 16}})
        self.assertEqual(result['failed_tiles'], 0)
        self.assertEqual(len(result['raw_tiles']), 4)
        self.assertEqual(calls, 4)


if __name__ == '__main__':
    unittest.main()
