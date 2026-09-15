import pandas as pd
import pytest


@pytest.fixture
def raw_reviews():
    rows = []
    for user_index, user_id in enumerate(("u1", "u2", "u3")):
        for time, item_id, score, text in (
            (1, "a", 5, "apple tea"),
            (2, "b", 4, "berry tea"),
            (3, "c", 2 if user_index == 0 else 5, "cocoa tea"),
        ):
            rows.append(
                {
                    "UserId": user_id,
                    "ProductId": item_id,
                    "Score": score,
                    "Time": time,
                    "Summary": text,
                    "Text": text,
                }
            )
    # Same review event under another ASIN: must not survive preparation.
    rows.append({"UserId": "u1", "ProductId": "alias-a", "Score": 5, "Time": 1, "Summary": "apple tea", "Text": "apple tea"})
    return pd.DataFrame(rows)


@pytest.fixture
def train_frame():
    return pd.DataFrame(
        [
            ("u1", "a", 5, 1, "apple", "apple tea"),
            ("u1", "b", 4, 2, "berry", "berry tea"),
            ("u2", "a", 5, 1, "apple", "apple tea"),
            ("u2", "c", 5, 2, "cocoa", "cocoa tea"),
            ("u3", "b", 4, 1, "berry", "berry tea"),
            ("u3", "c", 5, 2, "cocoa", "cocoa tea"),
            ("u4", "d", 5, 1, "date", "date tea"),
        ],
        columns=["user_id", "item_id", "rating", "timestamp", "Summary", "Text"],
    )

