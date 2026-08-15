from concurrent.futures import ThreadPoolExecutor
from src.utils.repositories import load_repositories, save_repository, delete_repository
from src.retrieval.retriever import search
from src.indexing.indexer import index_codebase


def test_repository_isolation():
    # Search only inside repository A
    results_a = search(
        query="user model",
        repository="sample_repo",
        k=5,
        threshold=0
    )

    # Every result must belong to sample_repo
    for result in results_a["results"]:
        assert result["metadata"]["repository"] == "sample_repo"

    print("Repository isolation test passed.")

def test_repository_name_collision():

    result = search(
        query="user model",
        repository="sample_repo",
        k=10,
        threshold=0
    )

    for item in result["results"]:
        assert item["metadata"]["repository"] == "sample_repo"

    print("Repository name collision test passed.")

def test_concurrent_indexing():

    repositories = [
        "sample_repo",
        "sample_repo2"
    ]

    # Index both repositories in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(index_codebase, repo)
            for repo in repositories
        ]

        # Wait for both indexing tasks to finish
        for future in futures:
            future.result()

    # Check repository A
    result_a = search(
        query="user model",
        repository="sample_repo",
        k=5,
        threshold=0
    )

    for result in result_a["results"]:
        assert result["metadata"]["repository"] == "sample_repo"

    # Check repository B
    result_b = search(
        query="user model",
        repository="sample_repo2",
        k=5,
        threshold=0
    )

    for result in result_b["results"]:
        assert result["metadata"]["repository"] == "sample_repo2"

    print("Concurrent indexing test passed.")

def test_concurrent_repository_writes():

    repository_names = [
        "concurrent_repo_1",
        "concurrent_repo_2",
        "concurrent_repo_3",
        "concurrent_repo_4"
    ]

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(save_repository, repo)
            for repo in repository_names
        ]

        for future in futures:
            future.result()

    repositories = load_repositories()

    for repo in repository_names:
        assert repo in repositories

    for repo in repository_names:
        delete_repository(repo)
        

    print("Concurrent repository write test passed.")

def test_nonexistent_repository():
    result = search(
        query="user model",
        repository="does_not_exist",
        k=5,
        threshold=0
    )

    assert result["results"] == []

    print("Non-existent repository test passed.")

def test_paraphrased_query():
    result = search(
        query="How is the user data model defined?",
        repository="sample_repo",
        k=5,
        threshold=0
    )

    assert len(result["results"]) > 0

    for item in result["results"]:
        assert item["metadata"]["repository"] == "sample_repo"

    print("Paraphrased query test passed.")

def test_cross_repository_contamination():
    result = search(
        query="user model",
        repository="sample_repo",
        k=10,
        threshold=0
    )

    for item in result["results"]:
        assert item["metadata"]["repository"] == "sample_repo"

    print("Cross-repository contamination test passed.")

def test_deletion_isolation():
    # Make sure both repositories are indexed
    index_codebase("sample_repo")
    index_codebase("sample_repo2")

    # Delete repository A
    delete_repository("sample_repo")

    # Repository B should still be searchable
    result = search(
        query="user model",
        repository="sample_repo2",
        k=10,
        threshold=0
    )

    # B's chunks must still exist
    assert len(result["results"]) > 0

    # No result from repository A should exist
    for item in result["results"]:
        assert item["metadata"]["repository"] == "sample_repo2"

    print("Deletion isolation test passed.")

def test_reindexing_isolation():
    # Index both repositories
    index_codebase("sample_repo")
    index_codebase("sample_repo2")

    # Re-index repository A
    index_codebase("sample_repo")

    # Search repository B
    result_b = search(
        query="user model",
        repository="sample_repo2",
        k=10,
        threshold=0
    )

    # B must still contain only its own chunks
    assert len(result_b["results"]) > 0

    for item in result_b["results"]:
        assert item["metadata"]["repository"] == "sample_repo2"

    print("Re-indexing isolation test passed.")

def test_ambiguous_query():
    result = search(
        query="data",
        repository="sample_repo",
        k=5,
        threshold=0
    )

    # Search should return only chunks from requested repository
    for item in result["results"]:
        assert item["metadata"]["repository"] == "sample_repo"

    print("Ambiguous query test passed.")

if __name__ == "__main__":
    test_repository_isolation()
    test_repository_name_collision()
    test_concurrent_indexing()
    test_concurrent_repository_writes()
    test_nonexistent_repository()
    test_paraphrased_query()
    test_cross_repository_contamination()
    test_deletion_isolation()
    test_reindexing_isolation()
    test_ambiguous_query()