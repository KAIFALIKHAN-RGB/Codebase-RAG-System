from threading import Event, Thread

from src.storage import chroma_store
from src.storage.storage_lock import storage_access


def test_reset_callback_registration_is_idempotent(monkeypatch):
    callback = lambda: None
    monkeypatch.setattr(chroma_store, "_reset_callbacks", [])

    chroma_store.register_reset_callback(callback)
    chroma_store.register_reset_callback(callback)

    assert chroma_store._reset_callbacks == [callback]


def test_reset_waits_for_active_storage_operation(monkeypatch):
    from src.retrieval import retriever

    previous_client = chroma_store.client
    previous_collection = chroma_store.collection
    previous_generation = chroma_store.get_storage_generation()
    previous_retriever_client = retriever.client
    previous_retriever_collection = retriever.collection
    entered = Event()
    release = Event()
    reset_started = Event()
    reset_finished = Event()
    monkeypatch.setattr(chroma_store.os.path, "exists", lambda path: True)
    monkeypatch.setattr(chroma_store.shutil, "rmtree", lambda path: None)

    def hold_storage():
        with storage_access():
            entered.set()
            release.wait(timeout=2)

    def reset():
        reset_started.set()
        chroma_store.reset_database()
        reset_finished.set()

    holder = Thread(target=hold_storage)
    resetter = Thread(target=reset)
    holder.start()
    assert entered.wait(timeout=2)
    resetter.start()
    assert reset_started.wait(timeout=2)
    assert not reset_finished.wait(timeout=0.1)

    release.set()
    holder.join(timeout=2)
    resetter.join(timeout=2)

    assert reset_finished.is_set()
    chroma_store.client = previous_client
    chroma_store.collection = previous_collection
    retriever.client = previous_retriever_client
    retriever.collection = previous_retriever_collection
    monkeypatch.setattr(chroma_store, "_storage_generation", previous_generation)
