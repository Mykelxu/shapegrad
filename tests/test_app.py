import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest


class AppTests(unittest.TestCase):
    def test_generate_and_compare(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / 'app.py'), default_timeout=60).run()
        self.assertFalse(app.exception)
        app.slider[0].set_value(8)
        app.slider[1].set_value(25)
        app.select_slider[0].set_value(48)
        app.slider[2].set_value(8)
        app.slider[3].set_value(8)
        app.number_input[0].set_value(1)
        app.selectbox[0].select("Residual-guided search")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        results = app.session_state['results']
        self.assertEqual(len(results), 1)
        for result in results:
            self.assertTrue(result['png'].startswith(b'\x89PNG'))
            self.assertIn('<svg', result['svg'])
            self.assertLessEqual(result['metrics']['best_loss'], result['metrics']['initial_loss'])
