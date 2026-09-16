from foodrec.data import DataConfig, ReviewDatasetBuilder, per_user_temporal_split


def test_preparation_removes_cross_asin_copy_and_iterates_core(raw_reviews):
    builder = ReviewDatasetBuilder(DataConfig(min_user_interactions=2, min_item_interactions=2))
    prepared = builder.prepare(raw_reviews)
    assert "alias-a" not in set(prepared.item_id)
    assert not prepared.duplicated(["user_id", "item_id"]).any()
    assert prepared.groupby("user_id").size().min() >= 2
    assert prepared.groupby("item_id").size().min() >= 2
    assert builder.report is not None
    assert builder.report.after_cross_asin_dedup < builder.report.after_required_values


def test_per_user_temporal_split_does_not_mix_timestamp_groups(raw_reviews):
    builder = ReviewDatasetBuilder(DataConfig(min_user_interactions=2, min_item_interactions=2))
    prepared = builder.prepare(raw_reviews)
    split = per_user_temporal_split(prepared)
    for user_id, train_group in split.train.groupby("user_id"):
        assert train_group.timestamp.max() < split.validation.query("user_id == @user_id").timestamp.min()
        assert split.validation.query("user_id == @user_id").timestamp.max() < split.test.query("user_id == @user_id").timestamp.min()

