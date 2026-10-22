import unittest
from unittest.mock import patch

from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CallToolResult

from faceledger.comparison import ComparisonOutcome
from faceledger.mcp_server import mcp


class MCPModelTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_choice_and_reference_guidance_are_advertised(self) -> None:
        tool = (await mcp.list_tools())[0]
        schema = tool.input_schema
        self.assertEqual(schema["properties"]["model"]["default"], "facenet512")
        self.assertEqual(
            schema["properties"]["model"]["enum"], ["facenet512", "arcface"]
        )
        self.assertNotIn("model", schema.get("required", []))
        for text in ("arcface", "0.30", "0.68"):
            self.assertIn(text, tool.description or "")

    async def test_default_and_explicit_models_reach_comparison(self) -> None:
        for selection, expected in (
            ({}, "Facenet512"),
            ({"model": "facenet512"}, "Facenet512"),
            ({"model": "arcface"}, "ArcFace"),
        ):
            with self.subTest(selection=selection):
                with patch(
                    "faceledger.mcp_server.compare",
                    return_value=ComparisonOutcome(matches=()),
                ) as compare:
                    result = await mcp.call_tool(
                        "compare_source_to_identity_folder",
                        {
                            "source_image": "/source.jpg",
                            "target_folder": "/target",
                            **selection,
                        },
                    )
                self.assertIsInstance(result, CallToolResult)
                assert isinstance(result, CallToolResult)
                self.assertFalse(result.is_error)
                self.assertEqual(compare.call_args.args[0].model_name, expected)

    async def test_unsupported_model_is_rejected_before_comparison(self) -> None:
        with (
            patch("faceledger.mcp_server.compare") as compare,
            self.assertRaisesRegex(ToolError, "facenet512.*arcface"),
        ):
            await mcp.call_tool(
                "compare_source_to_identity_folder",
                {
                    "source_image": "/source.jpg",
                    "target_folder": "/target",
                    "model": "unknown",
                },
            )
        compare.assert_not_called()
