import contextlib, shelve

@contextlib.contextmanager
def open_db(path: str = 'app.db'):
    db = shelve.open(path, writeback=True)
    try:
        yield db
    finally:
        db.close()
