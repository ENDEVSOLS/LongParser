"""Small external Motor boundary double for isolated persistence workflows.

It supports only the Mongo operations used by the tested application. It is not
a MongoDB compatibility implementation and does not replace real-store tests.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace


def matches(document, query):
    for key, expected in query.items():
        if key == "$or":
            if not any(matches(document, item) for item in expected):
                return False
            continue
        actual = document.get(key)
        if isinstance(expected, dict):
            for operation, value in expected.items():
                if operation == "$in" and actual not in value:
                    return False
                if operation == "$lte" and (actual is None or actual > value):
                    return False
                if operation == "$ne" and actual == value:
                    return False
        elif actual != expected:
            return False
    return True


def projected(document, projection):
    result = deepcopy(document)
    if projection:
        includes = {key for key, enabled in projection.items() if enabled}
        if includes:
            result = {key: value for key, value in result.items() if key in includes}
        else:
            for key, enabled in projection.items():
                if not enabled:
                    result.pop(key, None)
    return result


class Cursor:
    def __init__(self, documents):
        self.documents = documents

    def sort(self, key, direction):
        self.documents.sort(key=lambda document: document.get(key), reverse=direction < 0)
        return self

    def skip(self, count):
        self.documents = self.documents[count:]
        return self

    def limit(self, count):
        self.documents = self.documents[:count]
        return self

    async def to_list(self, length):
        return deepcopy(self.documents[:length])

    def __aiter__(self):
        self.iterator = iter(self.documents)
        return self

    async def __anext__(self):
        try:
            return next(self.iterator)
        except StopIteration as error:
            raise StopAsyncIteration from error


class Collection:
    def __init__(self):
        self.documents = []
        self.indexes = []
        self.calls = []

    async def create_index(self, keys, **kwargs):
        self.indexes.append((keys, kwargs))

    async def insert_one(self, document):
        self.documents.append(deepcopy(document))
        return SimpleNamespace(inserted_id=len(self.documents))

    async def find_one(self, query, projection=None):
        self.calls.append(("find_one", deepcopy(query)))
        return next((projected(document, projection) for document in self.documents if matches(document, query)), None)

    def find(self, query, projection=None):
        self.calls.append(("find", deepcopy(query)))
        return Cursor([projected(document, projection) for document in self.documents if matches(document, query)])

    async def count_documents(self, query):
        return sum(matches(document, query) for document in self.documents)

    def _update(self, query, updates, many=False, upsert=False):
        self.calls.append(("update", deepcopy(query), deepcopy(updates)))
        selected = [document for document in self.documents if matches(document, query)]
        if not many:
            selected = selected[:1]
        if not selected and upsert:
            document = {key: value for key, value in query.items() if not isinstance(value, dict)}
            self.documents.append(document)
            selected = [document]
        for document in selected:
            document.update(deepcopy(updates.get("$set", {})))
            for key, amount in updates.get("$inc", {}).items():
                document[key] = document.get(key, 0) + amount
            for key in updates.get("$unset", {}):
                document.pop(key, None)
        return SimpleNamespace(modified_count=len(selected), matched_count=len(selected))

    async def update_one(self, query, updates, upsert=False):
        return self._update(query, updates, upsert=upsert)

    async def update_many(self, query, updates):
        return self._update(query, updates, many=True)

    async def find_one_and_update(self, query, updates, return_document=True, projection=None, sort=None):
        if sort:
            for key, direction in reversed(sort):
                self.documents.sort(key=lambda record: record.get(key), reverse=direction < 0)
        result = self._update(query, updates)
        if not result.matched_count:
            return None
        # The application's optimistic version filter has changed after update.
        query = {key: value for key, value in query.items() if key != "version"}
        return await self.find_one(query, projection)

    async def delete_many(self, query):
        previous = len(self.documents)
        self.documents = [document for document in self.documents if not matches(document, query)]
        return SimpleNamespace(deleted_count=previous - len(self.documents))

    async def delete_one(self, query):
        for document in self.documents:
            if matches(document, query):
                self.documents.remove(document)
                return SimpleNamespace(deleted_count=1)
        return SimpleNamespace(deleted_count=0)

    def aggregate(self, pipeline):
        query = pipeline[0]["$match"]
        counts = {}
        for document in self.documents:
            if matches(document, query):
                status = document.get("review_status")
                counts[status] = counts.get(status, 0) + 1
        return Cursor([{"_id": status, "count": count} for status, count in counts.items()])


class MemoryClient:
    def __init__(self, *args, **kwargs):
        self.collections = {}
        self.closed = False

    def __getitem__(self, key):
        return self

    def collection(self, key):
        return self.collections.setdefault(key, Collection())

    def close(self):
        self.closed = True


class MemoryDatabase:
    def __init__(self, client):
        self.client = client

    def __getitem__(self, key):
        return self.client.collection(key)


class MotorClient(MemoryClient):
    def __getitem__(self, key):
        return MemoryDatabase(self)
