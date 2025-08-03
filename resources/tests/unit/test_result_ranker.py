import pytest
from jarvis.services.ranking import ResultRanker
from jarvis.models.document import SearchResult
from pathlib import Path

@pytest.fixture
def result_ranker():
    return ResultRanker()

def test_rank_results_semantic_only(result_ranker):
    results = [
        SearchResult("vault1", Path("path1"), 0.8),
        SearchResult("vault1", Path("path2"), 0.9),
        SearchResult("vault1", Path("path3"), 0.7),
    ]
    ranked = result_ranker.rank_results(results)
    assert [r.similarity_score for r in ranked] == [0.9, 0.8, 0.7]

def test_rank_results_keyword_only(result_ranker):
    results = [
        {"path": "pathB", "name": "Note B", "type": "keyword"},
        {"path": "pathA", "name": "Note A", "type": "keyword"},
        {"path": "pathC", "name": "Note C", "type": "keyword"},
    ]
    ranked = result_ranker.rank_results(results)
    assert [r["path"] for r in ranked] == ["pathA", "pathB", "pathC"]

def test_rank_results_mixed(result_ranker):
    semantic_results = [
        SearchResult("vault1", Path("path1"), 0.8),
        SearchResult("vault1", Path("path2"), 0.9),
    ]
    keyword_results = [
        {"path": "pathA", "name": "Note A", "type": "keyword"},
        {"path": "pathB", "name": "Note B", "type": "keyword"},
    ]
    all_results = semantic_results + keyword_results
    ranked = result_ranker.rank_results(all_results)
    # Expect semantic results first (by score), then keyword (by path)
    assert ranked[0].similarity_score == 0.9
    assert ranked[1].similarity_score == 0.8
    assert ranked[2]["path"] == "pathA"
    assert ranked[3]["path"] == "pathB"

def test_deduplicate_results_no_duplicates(result_ranker):
    results = [
        SearchResult("vault1", Path("path1"), 0.8),
        {"path": "path2", "vault_name": "vault1", "type": "keyword"},
    ]
    deduplicated = result_ranker.deduplicate_results(results)
    assert len(deduplicated) == 2

def test_deduplicate_results_semantic_over_keyword(result_ranker):
    semantic = SearchResult("vault1", Path("path1"), 0.9)
    keyword = {"path": "path1", "vault_name": "vault1", "type": "keyword"}
    results = [keyword, semantic]
    deduplicated = result_ranker.deduplicate_results(results)
    assert len(deduplicated) == 1
    assert isinstance(deduplicated[0], SearchResult) # Semantic should be kept
    assert deduplicated[0].similarity_score == 0.9

def test_deduplicate_results_semantic_only_duplicates(result_ranker):
    semantic1 = SearchResult("vault1", Path("path1"), 0.9)
    semantic2 = SearchResult("vault1", Path("path1"), 0.8) # Lower score, should be removed
    results = [semantic1, semantic2]
    deduplicated = result_ranker.deduplicate_results(results)
    assert len(deduplicated) == 1
    assert deduplicated[0].similarity_score == 0.9

def test_merge_and_rank(result_ranker):
    semantic_results = [
        SearchResult("vault1", Path("noteA.md"), 0.9),
        SearchResult("vault1", Path("noteC.md"), 0.7),
    ]
    keyword_results = [
        {"path": "noteB.md", "vault_name": "vault1", "name": "Note B", "type": "keyword"},
        {"path": "noteA.md", "vault_name": "vault1", "name": "Note A", "type": "keyword"}, # Duplicate of semantic
    ]
    merged_ranked = result_ranker.merge_and_rank(semantic_results, keyword_results)
    assert len(merged_ranked) == 3
    assert merged_ranked[0].path == Path("noteA.md") # Semantic, highest score
    assert merged_ranked[1].path == Path("noteC.md") # Semantic
    assert merged_ranked[2]["path"] == "noteB.md" # Keyword
