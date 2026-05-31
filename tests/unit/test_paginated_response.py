from domain.models import PaginatedResponse


def test_paginated_response_pages_rounds_up():
    resp = PaginatedResponse(items=[], total=21, page=2, limit=20)
    assert resp.pages == 2


def test_paginated_response_single_page():
    resp = PaginatedResponse(items=["a"], total=1, page=1, limit=20)
    assert resp.pages == 1


def test_paginated_response_zero_total():
    resp = PaginatedResponse(items=[], total=0, page=1, limit=20)
    assert resp.pages == 1
