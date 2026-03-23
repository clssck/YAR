"""Tests for graph_routes.py - Graph API endpoints."""

import asyncio
import importlib
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

import yar.api.routers.graph_routes as graph_routes_module
from yar.api.routers.graph_routes import (
    EntityCreateRequest,
    EntityMergeRequest,
    EntityUpdateRequest,
    OrphanConnectionRequest,
    OrphanConnectionStatusResponse,
    RelationCreateRequest,
    RelationUpdateRequest,
)


def _structured_orphan_request(value: object) -> dict[str, int | str]:
    assert isinstance(value, dict)
    return cast(dict[str, int | str], value)


# =============================================================================
# Request Model Validation Tests
# =============================================================================


class TestEntityUpdateRequestValidation:
    """Test EntityUpdateRequest model validation."""

    @pytest.mark.offline
    def test_valid_simple_update(self):
        """Test valid simple entity update request."""
        req = EntityUpdateRequest(
            entity_name='Tesla',
            updated_data={'description': 'Updated description'},
        )
        assert req.entity_name == 'Tesla'
        assert req.updated_data == {'description': 'Updated description'}
        assert req.allow_rename is False
        assert req.allow_merge is False

    @pytest.mark.offline
    def test_valid_rename_with_merge(self):
        """Test valid rename request with merge enabled."""
        req = EntityUpdateRequest(
            entity_name='Elon Msk',
            updated_data={'entity_name': 'Elon Musk', 'description': 'Corrected'},
            allow_rename=True,
            allow_merge=True,
        )
        assert req.entity_name == 'Elon Msk'
        assert req.allow_rename is True
        assert req.allow_merge is True

    @pytest.mark.offline
    def test_missing_required_fields(self):
        """Test that missing required fields raises validation error."""
        with pytest.raises(ValidationError):
            EntityUpdateRequest(entity_name='Tesla')


class TestRelationUpdateRequestValidation:
    """Test RelationUpdateRequest model validation."""

    @pytest.mark.offline
    def test_valid_relation_update(self):
        """Test valid relation update request."""
        req = RelationUpdateRequest(
            source_id='Elon Musk',
            target_id='Tesla',
            updated_data={'description': 'CEO relationship'},
        )
        assert req.source_id == 'Elon Musk'
        assert req.target_id == 'Tesla'
        assert req.updated_data == {'description': 'CEO relationship'}

    @pytest.mark.offline
    def test_missing_target_id(self):
        """Test that missing target_id raises validation error."""
        with pytest.raises(ValidationError):
            RelationUpdateRequest(
                source_id='Elon Musk',
                updated_data={'description': 'test'},
            )


class TestEntityMergeRequestValidation:
    """Test EntityMergeRequest model validation."""

    @pytest.mark.offline
    def test_valid_merge_request(self):
        """Test valid entity merge request."""
        req = EntityMergeRequest(
            entities_to_change=['Elon Msk', 'Ellon Musk'],
            entity_to_change_into='Elon Musk',
        )
        assert len(req.entities_to_change) == 2
        assert req.entity_to_change_into == 'Elon Musk'

    @pytest.mark.offline
    def test_empty_entities_list(self):
        """Test that empty entities list raises validation error."""
        with pytest.raises(ValidationError):
            EntityMergeRequest(
                entities_to_change=[],
                entity_to_change_into='Elon Musk',
            )

    @pytest.mark.offline
    def test_empty_target_entity(self):
        """Test that empty target entity raises validation error."""
        with pytest.raises(ValidationError):
            EntityMergeRequest(
                entities_to_change=['Elon Msk'],
                entity_to_change_into='',
            )


class TestOrphanConnectionRequestValidation:
    """Test OrphanConnectionRequest model validation."""

    @pytest.mark.offline
    def test_default_values(self):
        """Test default values for orphan connection request."""
        req = OrphanConnectionRequest()
        assert req.max_candidates == 3
        assert req.similarity_threshold is None
        assert req.confidence_threshold is None
        assert req.cross_connect is None

    @pytest.mark.offline
    def test_custom_values(self):
        """Test custom values for orphan connection request."""
        req = OrphanConnectionRequest(
            max_candidates=5,
            similarity_threshold=0.7,
            confidence_threshold=0.8,
            cross_connect=True,
        )
        assert req.max_candidates == 5
        assert req.similarity_threshold == 0.7
        assert req.confidence_threshold == 0.8
        assert req.cross_connect is True

    @pytest.mark.offline
    def test_invalid_max_candidates_too_low(self):
        """Test that max_candidates below 1 raises validation error."""
        with pytest.raises(ValidationError):
            OrphanConnectionRequest(max_candidates=0)

    @pytest.mark.offline
    def test_invalid_max_candidates_too_high(self):
        """Test that max_candidates above 10 raises validation error."""
        with pytest.raises(ValidationError):
            OrphanConnectionRequest(max_candidates=11)

    @pytest.mark.offline
    def test_invalid_similarity_threshold(self):
        """Test that similarity_threshold above 1.0 raises validation error."""
        with pytest.raises(ValidationError):
            OrphanConnectionRequest(similarity_threshold=1.5)

    @pytest.mark.offline
    def test_invalid_confidence_threshold_negative(self):
        """Test that negative confidence_threshold raises validation error."""
        with pytest.raises(ValidationError):
            OrphanConnectionRequest(confidence_threshold=-0.1)


class TestEntityCreateRequestValidation:
    """Test EntityCreateRequest model validation."""

    @pytest.mark.offline
    def test_valid_entity_create(self):
        """Test valid entity create request."""
        req = EntityCreateRequest(
            entity_name='Tesla',
            entity_data={
                'description': 'Electric vehicle manufacturer',
                'entity_type': 'ORGANIZATION',
            },
        )
        assert req.entity_name == 'Tesla'
        assert req.entity_data['entity_type'] == 'ORGANIZATION'

    @pytest.mark.offline
    def test_empty_entity_name(self):
        """Test that empty entity name raises validation error."""
        with pytest.raises(ValidationError):
            EntityCreateRequest(
                entity_name='',
                entity_data={'description': 'test'},
            )


class TestRelationCreateRequestValidation:
    """Test RelationCreateRequest model validation."""

    @pytest.mark.offline
    def test_valid_relation_create(self):
        """Test valid relation create request."""
        req = RelationCreateRequest(
            source_entity='Elon Musk',
            target_entity='Tesla',
            relation_data={
                'description': 'Elon Musk is the CEO of Tesla',
                'keywords': 'CEO, founder',
                'weight': 1.0,
            },
        )
        assert req.source_entity == 'Elon Musk'
        assert req.target_entity == 'Tesla'
        assert req.relation_data['weight'] == 1.0

    @pytest.mark.offline
    def test_empty_source_entity(self):
        """Test that empty source entity raises validation error."""
        with pytest.raises(ValidationError):
            RelationCreateRequest(
                source_entity='',
                target_entity='Tesla',
                relation_data={'description': 'test'},
            )

    @pytest.mark.offline
    def test_empty_target_entity(self):
        """Test that empty target entity raises validation error."""
        with pytest.raises(ValidationError):
            RelationCreateRequest(
                source_entity='Elon Musk',
                target_entity='',
                relation_data={'description': 'test'},
            )


class TestOrphanConnectionStatusResponseValidation:
    """Test OrphanConnectionStatusResponse model validation."""

    @pytest.mark.offline
    def test_valid_status_response(self):
        """Test valid orphan connection status response."""
        resp = OrphanConnectionStatusResponse(
            busy=True,
            job_name='Connecting orphan entities',
            job_start='2024-01-15T10:30:00',
            total_orphans=100,
            processed_orphans=45,
            connections_made=12,
            pending_request=None,
            active_request={'max_candidates': 3, 'max_degree': 0, 'requested_at': '2024-01-15T10:30:00'},
            cancellation_requested=False,
            latest_message='Processing orphan 46/100...',
            history_messages=['Starting...', 'Processing...'],
        )
        assert resp.busy is True
        assert resp.total_orphans == 100
        assert resp.processed_orphans == 45

    @pytest.mark.offline
    def test_nullable_job_start(self):
        """Test that job_start can be None."""
        resp = OrphanConnectionStatusResponse(
            busy=False,
            job_name='',
            job_start=None,
            total_orphans=0,
            processed_orphans=0,
            connections_made=0,
            pending_request=None,
            active_request=None,
            cancellation_requested=False,
            latest_message='',
            history_messages=[],
        )
        assert resp.job_start is None


# =============================================================================
# Endpoint Tests
# =============================================================================


@pytest.fixture
def mock_rag():
    """Create a mock RAG instance with required methods."""
    rag = MagicMock()

    # Mock chunk_entity_relation_graph methods
    rag.chunk_entity_relation_graph = MagicMock()
    rag.chunk_entity_relation_graph.get_popular_labels = AsyncMock(return_value=['Tesla', 'Elon Musk', 'SpaceX'])
    rag.chunk_entity_relation_graph.search_labels = AsyncMock(return_value=['Tesla', 'Texas'])
    rag.chunk_entity_relation_graph.has_node = AsyncMock(return_value=True)

    # Mock RAG methods
    rag.get_graph_labels = AsyncMock(return_value=['PERSON', 'ORGANIZATION', 'LOCATION'])
    rag.get_knowledge_graph = AsyncMock(
        return_value={
            'nodes': [
                {'id': 'Tesla', 'label': 'ORGANIZATION'},
                {'id': 'Elon Musk', 'label': 'PERSON'},
            ],
            'edges': [
                {'source': 'Elon Musk', 'target': 'Tesla', 'relationship': 'CEO_OF'},
            ],
        }
    )
    rag.aedit_entity = AsyncMock(
        return_value={
            'entity_name': 'Tesla',
            'description': 'Updated description',
            'operation_summary': {
                'merged': False,
                'merge_status': 'not_attempted',
                'operation_status': 'success',
                'final_entity': 'Tesla',
            },
        }
    )
    rag.aedit_relation = AsyncMock(
        return_value={
            'src_id': 'Elon Musk',
            'tgt_id': 'Tesla',
            'description': 'Updated relationship',
        }
    )
    rag.acreate_entity = AsyncMock(
        return_value={
            'entity_name': 'SpaceX',
            'description': 'Space company',
            'entity_type': 'ORGANIZATION',
        }
    )
    rag.acreate_relation = AsyncMock(
        return_value={
            'src_id': 'Elon Musk',
            'tgt_id': 'SpaceX',
            'description': 'Founder relationship',
        }
    )
    rag.amerge_entities = AsyncMock(
        return_value={
            'merged_entity': 'Elon Musk',
            'deleted_entities': ['Elon Msk'],
            'relationships_transferred': 5,
        }
    )
    rag.aconnect_orphan_entities = AsyncMock(
        return_value={
            'orphans_found': 10,
            'connections_made': 3,
            'connections': [],
        }
    )

    return rag


@pytest.fixture
def test_app(mock_rag):
    """Create FastAPI test app with graph routes."""
    from fastapi import FastAPI

    app = FastAPI()
    reloaded_graph_routes = importlib.reload(graph_routes_module)
    router = reloaded_graph_routes.create_graph_routes(mock_rag)
    app.include_router(router)
    return app


class TestGraphLabelEndpoints:
    """Test graph label-related endpoints."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_graph_labels(self, test_app, mock_rag):
        """Test GET /graph/label/list endpoint."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/label/list')

        assert response.status_code == 200
        assert response.json() == ['PERSON', 'ORGANIZATION', 'LOCATION']
        mock_rag.get_graph_labels.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_popular_labels_default_limit(self, test_app, mock_rag):
        """Test GET /graph/label/popular with default limit."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/label/popular')

        assert response.status_code == 200
        mock_rag.chunk_entity_relation_graph.get_popular_labels.assert_called_once_with(300)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_popular_labels_custom_limit(self, test_app, mock_rag):
        """Test GET /graph/label/popular with custom limit."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/label/popular?limit=100')

        assert response.status_code == 200
        mock_rag.chunk_entity_relation_graph.get_popular_labels.assert_called_once_with(100)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_popular_labels_invalid_limit(self, test_app):
        """Test GET /graph/label/popular with invalid limit returns 422."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/label/popular?limit=0')

        assert response.status_code == 422

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_search_labels(self, test_app, mock_rag):
        """Test GET /graph/label/search endpoint."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/label/search?q=Tes')

        assert response.status_code == 200
        assert response.json() == ['Tesla', 'Texas']
        mock_rag.chunk_entity_relation_graph.search_labels.assert_called_once_with('Tes', 50)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_search_labels_custom_limit(self, test_app, mock_rag):
        """Test GET /graph/label/search with custom limit."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/label/search?q=Tes&limit=10')

        assert response.status_code == 200
        mock_rag.chunk_entity_relation_graph.search_labels.assert_called_once_with('Tes', 10)

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_search_labels_missing_query(self, test_app):
        """Test GET /graph/label/search without query returns 422."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/label/search')

        assert response.status_code == 422


class TestKnowledgeGraphEndpoint:
    """Test GET /graphs endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_knowledge_graph_basic(self, test_app, mock_rag):
        """Test GET /graphs with basic parameters."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graphs?label=Tesla')

        assert response.status_code == 200
        data = response.json()
        assert 'nodes' in data
        assert 'edges' in data
        mock_rag.get_knowledge_graph.assert_called_once_with(
            node_label='Tesla',
            max_depth=3,
            max_nodes=1000,
            min_degree=0,
            include_orphans=False,
        )

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_knowledge_graph_all_params(self, test_app, mock_rag):
        """Test GET /graphs with all custom parameters."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graphs?label=*&max_depth=5&max_nodes=500&min_degree=2&include_orphans=true')

        assert response.status_code == 200
        mock_rag.get_knowledge_graph.assert_called_once_with(
            node_label='*',
            max_depth=5,
            max_nodes=500,
            min_degree=2,
            include_orphans=True,
        )

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_get_knowledge_graph_missing_label(self, test_app):
        """Test GET /graphs without label returns 422."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graphs')

        assert response.status_code == 422


class TestEntityExistsEndpoint:
    """Test GET /graph/entity/exists endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_entity_exists_true(self, test_app, mock_rag):
        """Test entity exists returns true."""
        mock_rag.chunk_entity_relation_graph.has_node = AsyncMock(return_value=True)

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/entity/exists?name=Tesla')

        assert response.status_code == 200
        assert response.json() == {'exists': True}

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_entity_exists_false(self, test_app, mock_rag):
        """Test entity exists returns false for non-existent entity."""
        mock_rag.chunk_entity_relation_graph.has_node = AsyncMock(return_value=False)

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/entity/exists?name=NonExistent')

        assert response.status_code == 200
        assert response.json() == {'exists': False}

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_entity_exists_missing_name(self, test_app):
        """Test entity exists without name returns 422."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.get('/graph/entity/exists')

        assert response.status_code == 422


class TestEntityEditEndpoint:
    """Test POST /graph/entity/edit endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_update_entity_success(self, test_app):
        """Test successful entity update."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/entity/edit',
                json={
                    'entity_name': 'Tesla',
                    'updated_data': {'description': 'Updated description'},
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert data['message'] == 'Entity updated successfully'
        assert 'data' in data
        assert 'operation_summary' in data

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_update_entity_with_rename(self, test_app, mock_rag):
        """Test entity update with rename enabled."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/entity/edit',
                json={
                    'entity_name': 'Elon Msk',
                    'updated_data': {'entity_name': 'Elon Musk'},
                    'allow_rename': True,
                    'allow_merge': True,
                },
            )

        assert response.status_code == 200
        mock_rag.aedit_entity.assert_called_once_with(
            entity_name='Elon Msk',
            updated_data={'entity_name': 'Elon Musk'},
            allow_rename=True,
            allow_merge=True,
        )

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_update_entity_value_error(self, test_app, mock_rag):
        """Test entity update with ValueError returns 400."""
        mock_rag.aedit_entity = AsyncMock(side_effect=ValueError('Entity not found'))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/entity/edit',
                json={
                    'entity_name': 'NonExistent',
                    'updated_data': {'description': 'test'},
                },
            )

        assert response.status_code == 400
        assert 'Entity not found' in response.json()['detail']


class TestRelationEditEndpoint:
    """Test POST /graph/relation/edit endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_update_relation_success(self, test_app):
        """Test successful relation update."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/relation/edit',
                json={
                    'source_id': 'Elon Musk',
                    'target_id': 'Tesla',
                    'updated_data': {'description': 'Updated relationship'},
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert data['message'] == 'Relation updated successfully'

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_update_relation_value_error(self, test_app, mock_rag):
        """Test relation update with ValueError returns 400."""
        mock_rag.aedit_relation = AsyncMock(side_effect=ValueError('Relation not found'))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/relation/edit',
                json={
                    'source_id': 'A',
                    'target_id': 'B',
                    'updated_data': {'description': 'test'},
                },
            )

        assert response.status_code == 400
        assert 'Relation not found' in response.json()['detail']

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_update_relation_unexpected_error_returns_500(self, test_app, mock_rag):
        """Test relation update with unexpected error returns 500."""
        mock_rag.aedit_relation = AsyncMock(side_effect=RuntimeError('unexpected failure'))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/relation/edit',
                json={
                    'source_id': 'A',
                    'target_id': 'B',
                    'updated_data': {'description': 'test'},
                },
            )

        assert response.status_code == 500
        assert response.json()['detail'] == 'Internal server error while updating relation'


class TestEntityCreateEndpoint:
    """Test POST /graph/entity/create endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_create_entity_success(self, test_app):
        """Test successful entity creation."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/entity/create',
                json={
                    'entity_name': 'SpaceX',
                    'entity_data': {
                        'description': 'Space company',
                        'entity_type': 'ORGANIZATION',
                    },
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'SpaceX' in data['message']

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_create_entity_duplicate(self, test_app, mock_rag):
        """Test creating duplicate entity returns 400."""
        mock_rag.acreate_entity = AsyncMock(side_effect=ValueError('Entity already exists'))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/entity/create',
                json={
                    'entity_name': 'Tesla',
                    'entity_data': {'description': 'Duplicate'},
                },
            )

        assert response.status_code == 400
        assert 'Entity already exists' in response.json()['detail']


class TestRelationCreateEndpoint:
    """Test POST /graph/relation/create endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_create_relation_success(self, test_app):
        """Test successful relation creation."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/relation/create',
                json={
                    'source_entity': 'Elon Musk',
                    'target_entity': 'SpaceX',
                    'relation_data': {
                        'description': 'Founder relationship',
                        'keywords': 'founder, owner',
                    },
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'Elon Musk' in data['message']
        assert 'SpaceX' in data['message']

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_create_relation_missing_entity(self, test_app, mock_rag):
        """Test creating relation with missing entity returns 400."""
        mock_rag.acreate_relation = AsyncMock(side_effect=ValueError('Source entity does not exist'))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/relation/create',
                json={
                    'source_entity': 'NonExistent',
                    'target_entity': 'Tesla',
                    'relation_data': {'description': 'test'},
                },
            )

        assert response.status_code == 400
        assert 'Source entity does not exist' in response.json()['detail']


class TestEntityMergeEndpoint:
    """Test POST /graph/entities/merge endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_merge_entities_success(self, test_app):
        """Test successful entity merge."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/entities/merge',
                json={
                    'entities_to_change': ['Elon Msk', 'Ellon Musk'],
                    'entity_to_change_into': 'Elon Musk',
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert '2 entities' in data['message']
        assert 'Elon Musk' in data['message']

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_merge_entities_target_not_found(self, test_app, mock_rag):
        """Test merge with non-existent target entity returns 400."""
        mock_rag.amerge_entities = AsyncMock(side_effect=ValueError('Target entity does not exist'))

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/entities/merge',
                json={
                    'entities_to_change': ['A', 'B'],
                    'entity_to_change_into': 'NonExistent',
                },
            )

        assert response.status_code == 400
        assert 'Target entity does not exist' in response.json()['detail']


class TestOrphanConnectionEndpoint:
    """Test POST /graph/orphans/connect endpoint."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_connect_orphans_success(self, test_app):
        """Test successful orphan connection."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post(
                '/graph/orphans/connect',
                json={
                    'max_candidates': 5,
                    'similarity_threshold': 0.7,
                    'confidence_threshold': 0.8,
                    'cross_connect': True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'success'
        assert 'Connected 3 out of 10 orphan entities' in data['message']

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_connect_orphans_default_params(self, test_app, mock_rag):
        """Test orphan connection with default parameters."""
        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post('/graph/orphans/connect', json={})

        assert response.status_code == 200
        mock_rag.aconnect_orphan_entities.assert_called_once_with(
            max_candidates=3,
            similarity_threshold=None,
            confidence_threshold=None,
            cross_connect=None,
        )

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_connect_orphans_malformed_result_returns_500(self, test_app, mock_rag):
        """Test malformed orphan connection response returns 500."""
        mock_rag.aconnect_orphan_entities = AsyncMock(return_value={'connections': []})

        async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
            response = await client.post('/graph/orphans/connect', json={})

        assert response.status_code == 500
        assert response.json()['detail'] == 'Internal server error while connecting orphan entities'


class TestOrphanBackgroundControlEndpoints:
    """Tests for /graph/orphans background control endpoints."""

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_orphan_status_exposes_structured_queue_state(self, test_app):
        status = {
            'busy': True,
            'job_name': 'Connecting orphan entities',
            'job_start': '2024-01-15T10:30:00',
            'total_orphans': 4,
            'processed_orphans': 2,
            'connections_made': 1,
            'pending_request': {'max_candidates': 5, 'max_degree': 1, 'requested_at': '2024-01-15T10:31:00'},
            'active_request': {'max_candidates': 3, 'max_degree': 0, 'requested_at': '2024-01-15T10:30:00'},
            'cancellation_requested': False,
            'latest_message': 'running',
            'history_messages': ['a', 'b'],
        }

        with patch(
            'yar.kg.shared_storage.get_namespace_data',
            new=AsyncMock(return_value=status),
        ):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
                response = await client.get('/graph/orphans/status')

        assert response.status_code == 200
        payload = response.json()
        assert payload['pending_request']['max_candidates'] == 5
        assert payload['active_request']['max_degree'] == 0

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_orphan_start_preclaims_job_atomically(self, test_app, mock_rag):
        status = {
            'busy': False,
            'job_name': '',
            'job_start': None,
            'total_orphans': 0,
            'processed_orphans': 0,
            'connections_made': 0,
            'pending_request': None,
            'active_request': None,
            'cancellation_requested': False,
            'latest_message': '',
            'history_messages': [],
        }
        status_lock = asyncio.Lock()
        mock_rag.aprocess_orphan_connections_background = AsyncMock(return_value={'status': 'ok'})

        with (
            patch('yar.kg.shared_storage.get_namespace_data', new=AsyncMock(return_value=status)),
            patch('yar.kg.shared_storage.get_namespace_lock', return_value=status_lock),
        ):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
                response = await client.post('/graph/orphans/start?max_candidates=4&max_degree=2')

        assert response.status_code == 200
        assert response.json()['status'] == 'started'
        assert status['busy'] is True
        assert _structured_orphan_request(status['active_request'])['max_candidates'] == 4
        mock_rag.aprocess_orphan_connections_background.assert_awaited_once_with(
            max_candidates=4,
            max_degree=2,
            preclaimed=True,
        )

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_orphan_start_queues_structured_request_when_busy(self, test_app, mock_rag):
        status = {
            'busy': True,
            'pending_request': None,
            'active_request': {'max_candidates': 3, 'max_degree': 0, 'requested_at': '2024-01-15T10:30:00'},
            'history_messages': [],
            'latest_message': '',
        }
        status_lock = asyncio.Lock()
        mock_rag.aprocess_orphan_connections_background = AsyncMock()

        with (
            patch('yar.kg.shared_storage.get_namespace_data', new=AsyncMock(return_value=status)),
            patch('yar.kg.shared_storage.get_namespace_lock', return_value=status_lock),
        ):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
                response = await client.post('/graph/orphans/start?max_candidates=5&max_degree=1')

        assert response.status_code == 200
        payload = response.json()
        assert payload['status'] == 'already_running'
        assert payload['queued_request']['max_candidates'] == 5
        pending_request = _structured_orphan_request(status['pending_request'])
        assert pending_request['max_degree'] == 1
        mock_rag.aprocess_orphan_connections_background.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_orphan_cancel_honors_preclaimed_active_request(self, test_app):
        status = {
            'busy': False,
            'active_request': {'max_candidates': 3, 'max_degree': 0, 'requested_at': '2024-01-15T10:30:00'},
            'cancellation_requested': False,
            'history_messages': [],
        }
        status_lock = asyncio.Lock()

        with (
            patch('yar.kg.shared_storage.get_namespace_data', new=AsyncMock(return_value=status)),
            patch('yar.kg.shared_storage.get_namespace_lock', return_value=status_lock),
        ):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
                response = await client.post('/graph/orphans/cancel')

        assert response.status_code == 200
        assert response.json()['status'] == 'cancellation_requested'
        assert status['cancellation_requested'] is True

    @pytest.mark.asyncio
    @pytest.mark.offline
    async def test_orphan_cancel_not_running_without_active_request(self, test_app):
        status = {
            'busy': False,
            'active_request': None,
            'cancellation_requested': False,
            'history_messages': [],
        }
        status_lock = asyncio.Lock()

        with (
            patch('yar.kg.shared_storage.get_namespace_data', new=AsyncMock(return_value=status)),
            patch('yar.kg.shared_storage.get_namespace_lock', return_value=status_lock),
        ):
            async with AsyncClient(transport=ASGITransport(app=test_app), base_url='http://test') as client:
                response = await client.post('/graph/orphans/cancel')

        assert response.status_code == 200
        assert response.json()['status'] == 'not_running'
