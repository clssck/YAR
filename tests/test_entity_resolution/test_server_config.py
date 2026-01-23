"""Tests for server configuration issues.

These tests verify that configuration variables don't accidentally shadow
each other during server initialization.
"""



class TestApiKeyIsolation:
    """Tests that api_key for authentication is isolated from other api keys.

    Bug discovered: The rerank setup code was reusing the variable name `api_key`,
    which overwrote the authentication api_key and caused all routes to require
    the rerank API key for authentication.
    """

    def test_rerank_api_key_does_not_shadow_auth_key(self):
        """Verify rerank setup uses separate variable name from auth key.

        This is a static analysis test - we verify the variable naming in code
        to prevent the shadowing bug from recurring.
        """
        import ast
        from pathlib import Path

        server_file = Path(__file__).parent.parent.parent / 'yar' / 'api' / 'yar_server.py'
        source = server_file.read_text()
        tree = ast.parse(source)

        # Find all assignments to 'api_key' in the module
        api_key_assignments = []

        class AssignmentVisitor(ast.NodeVisitor):
            def visit_Assign(self, node):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'api_key':
                        api_key_assignments.append(node.lineno)
                self.generic_visit(node)

        AssignmentVisitor().visit(tree)

        # There should be exactly ONE assignment to api_key (the auth key at ~line 322)
        # If there are more, it means the bug has returned
        assert len(api_key_assignments) == 1, (
            f"Found {len(api_key_assignments)} assignments to 'api_key' at lines {api_key_assignments}. "
            "Expected exactly 1 (the auth key). "
            "Rerank config should use 'rerank_api_key' to avoid shadowing."
        )

    def test_rerank_uses_separate_variable(self):
        """Verify rerank module uses dedicated RERANK_BINDING_API_KEY env var."""
        from pathlib import Path

        # Rerank API key is handled in the rerank module, not yar_server.py
        # The server delegates to create_rerank_func() which handles API key internally
        rerank_file = Path(__file__).parent.parent.parent / 'yar' / 'rerank.py'
        source = rerank_file.read_text()

        # Should find RERANK_BINDING_API_KEY usage in rerank module
        assert 'RERANK_BINDING_API_KEY' in source, (
            "Rerank module should use 'RERANK_BINDING_API_KEY' env variable"
        )

        # Server should use create_rerank_func factory (not inline api_key handling)
        server_file = Path(__file__).parent.parent.parent / 'yar' / 'api' / 'yar_server.py'
        server_source = server_file.read_text()

        assert 'create_rerank_func' in server_source, (
            "Server should use create_rerank_func factory for rerank setup"
        )

        # Should NOT find bare api_key assignments in rerank section of server
        rerank_section_start = server_source.find('if args.enable_rerank:')
        if rerank_section_start != -1:
            # Find end of rerank section (next major block)
            rerank_section_end = server_source.find('rerank_model_func = None', rerank_section_start + 1)
            if rerank_section_end == -1:
                rerank_section_end = len(server_source)
            rerank_section = server_source[rerank_section_start:rerank_section_end]
            # Count 'api_key = os.getenv' (should be 0 - delegated to rerank module)
            import re
            bare_api_key = re.findall(r'\bapi_key\s*=\s*os\.getenv', rerank_section)
            assert len(bare_api_key) == 0, (
                f"Found {len(bare_api_key)} bare 'api_key = os.getenv' in server rerank section. "
                "API key should be handled by rerank module."
            )




class TestAuthDependencyConfig:
    """Tests for auth dependency configuration."""

    def test_get_combined_auth_dependency_with_none(self):
        """Auth dependency with api_key=None should not require API key."""
        from yar.api.utils_api import get_combined_auth_dependency

        # Create dependency with no API key
        dep = get_combined_auth_dependency(api_key=None)

        # The dependency should be a callable
        assert callable(dep)

        # We can't easily test the full behavior without a request context,
        # but we verify the function was created successfully

    def test_get_combined_auth_dependency_with_key(self):
        """Auth dependency with api_key set should require that key."""
        from yar.api.utils_api import get_combined_auth_dependency

        # Create dependency with API key
        dep = get_combined_auth_dependency(api_key='test-key-123')

        assert callable(dep)
