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

        server_file = Path(__file__).parent.parent.parent / 'lightrag' / 'api' / 'lightrag_server.py'
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
        """Verify rerank config uses 'rerank_api_key' variable name."""
        from pathlib import Path

        server_file = Path(__file__).parent.parent.parent / 'lightrag' / 'api' / 'lightrag_server.py'
        source = server_file.read_text()

        # Should find rerank_api_key assignments for each rerank binding
        assert 'rerank_api_key = os.getenv' in source, (
            "Rerank setup should use 'rerank_api_key' variable name"
        )

        # Should NOT find api_key assignments in rerank section
        # Check that between 'if args.enable_rerank:' and the local reranker,
        # there are no bare 'api_key =' assignments
        rerank_section_start = source.find('if args.enable_rerank:')
        rerank_section_end = source.find('else:  # local', rerank_section_start)

        if rerank_section_start != -1 and rerank_section_end != -1:
            rerank_section = source[rerank_section_start:rerank_section_end]
            # Count 'api_key = ' (not 'rerank_api_key = ')
            import re
            bare_api_key = re.findall(r'\bapi_key\s*=\s*os\.getenv', rerank_section)
            assert len(bare_api_key) == 0, (
                f"Found {len(bare_api_key)} bare 'api_key = os.getenv' in rerank section. "
                "Should use 'rerank_api_key' instead."
            )




class TestAuthDependencyConfig:
    """Tests for auth dependency configuration."""

    def test_get_combined_auth_dependency_with_none(self):
        """Auth dependency with api_key=None should not require API key."""
        from lightrag.api.utils_api import get_combined_auth_dependency

        # Create dependency with no API key
        dep = get_combined_auth_dependency(api_key=None)

        # The dependency should be a callable
        assert callable(dep)

        # We can't easily test the full behavior without a request context,
        # but we verify the function was created successfully

    def test_get_combined_auth_dependency_with_key(self):
        """Auth dependency with api_key set should require that key."""
        from lightrag.api.utils_api import get_combined_auth_dependency

        # Create dependency with API key
        dep = get_combined_auth_dependency(api_key='test-key-123')

        assert callable(dep)
