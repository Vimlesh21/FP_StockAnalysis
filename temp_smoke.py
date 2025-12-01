import importlib
modules = ['fastapi','uvicorn','sqlalchemy','psycopg','psycopg_binary','numpy','pandas','sklearn','joblib']
for m in modules:
    try:
        mod = importlib.import_module(m)
        ver = getattr(mod, '__version__', None) or getattr(mod, 'VERSION', None) or 'unknown'
        print(f'{m}: OK version={ver}')
    except Exception as e:
        print(f'{m}: ERROR: {e.__class__.__name__}: {e}')
