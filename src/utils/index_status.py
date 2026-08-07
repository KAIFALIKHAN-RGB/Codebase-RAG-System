# Stores current indexing status for each repository
index_status = {}


def set_index_status(repo_name, status):
    index_status[repo_name] = status


def get_index_status(repo_name):
    return index_status.get(repo_name, "not_found")