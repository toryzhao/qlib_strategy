def test_statistical_package_exists():
    """Test that statistical strategy package can be imported"""
    import strategies.statistical
    assert strategies.statistical is not None
