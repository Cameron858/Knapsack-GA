import pytest
from knapsack import Item


@pytest.fixture
def test_items() -> list[Item]:
    """Fixture providing a small list of test items."""
    return [
        Item(name="item1", value=10, weight=5),
        Item(name="item2", value=5, weight=3),
        Item(name="item3", value=15, weight=9),
        Item(name="item4", value=7, weight=4),
        Item(name="item5", value=6, weight=2),
    ]
