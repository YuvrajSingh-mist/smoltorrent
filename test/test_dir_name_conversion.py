"""Unit tests for model_id_to_dir_name() in utils/common_utils.py.

No network, no SSH, always fast.
"""

from utils.common_utils import model_id_to_dir_name


class TestModelIdToDirName:
    def test_slash_replaced_by_double_dash(self):
        assert (
            model_id_to_dir_name("mlx-community/Qwen2.5-0.5B-Instruct-bf16")
            == "mlx-community--Qwen2.5-0.5B-Instruct-bf16"
        )

    def test_no_slash_unchanged(self):
        assert model_id_to_dir_name("SmolLM2-135M") == "SmolLM2-135M"

    def test_multiple_slashes(self):
        # org/namespace/model → org--namespace--model
        assert model_id_to_dir_name("a/b/c") == "a--b--c"

    def test_already_double_dash_style_unchanged(self):
        assert (
            model_id_to_dir_name("mlx-community--SmolLM2-135M-Instruct")
            == "mlx-community--SmolLM2-135M-Instruct"
        )

    def test_empty_string(self):
        assert model_id_to_dir_name("") == ""
