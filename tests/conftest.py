import sys
import types

# stub heavy external modules so main can be imported without full dependencies
mods = {
    'mammoth': {},
    'numpy': {'array': lambda x: x},
    'openai': {},
    'pdfplumber': {},
    'PyPDF2': {'PdfReader': lambda *a, **k: types.SimpleNamespace(pages=[])},
    'docx': {'Document': lambda *a, **k: None},
    'dotenv': {'load_dotenv': lambda: None},
    'certifi': {'where': lambda: ''},
    'itsdangerous': {
        'URLSafeTimedSerializer': lambda *a, **k: types.SimpleNamespace(dumps=lambda obj: 'token', loads=lambda t: {}),
        'BadSignature': Exception,
    },
    'multipart': {
        '__version__': '0',
    },
}
for name, attrs in mods.items():
    if name not in sys.modules:
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

if 'multipart' in sys.modules and 'multipart.multipart' not in sys.modules:
    sub = types.ModuleType('multipart.multipart')
    sub.parse_options_header = lambda *a, **k: None
    sys.modules['multipart.multipart'] = sub

# minimal httpx stub for Starlette's TestClient
if 'httpx' not in sys.modules:
    httpx = types.ModuleType('httpx')

    class Request:
        def __init__(self, method, url):
            from urllib.parse import urlsplit
            parts = urlsplit(url)
            self.method = method
            self.url = types.SimpleNamespace(
                scheme=parts.scheme or 'http',
                netloc=parts.netloc.encode(),
                path=parts.path,
                raw_path=parts.path.encode(),
                query=parts.query.encode(),
            )
            self.headers = {}
            self._content = b''

        def read(self):
            return self._content

    class ByteStream:
        def __init__(self, data):
            self._data = data

        def read(self):
            return self._data

    class Response:
        def __init__(self, status_code=200, headers=None, stream=None, request=None):
            self.status_code = status_code
            self.headers = headers or []
            self.request = request
            data = stream.read() if hasattr(stream, 'read') else b''
            self.text = data.decode()
            self.content = data

    class BaseTransport:
        pass

    class Client:
        def __init__(self, *args, **kwargs):
            self._transport = kwargs.get('transport')
            self.base_url = kwargs.get('base_url', '')

        def request(self, method, url, **kwargs):
            req = Request(method, self.base_url + url)
            return self._transport.handle_request(req)

        def get(self, url, **kwargs):
            return self.request('GET', url, **kwargs)

        def post(self, url, **kwargs):
            return self.request('POST', url, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

    httpx.Request = Request
    httpx.Response = Response
    httpx.ByteStream = ByteStream
    httpx.Client = Client
    httpx.BaseTransport = BaseTransport
    httpx._client = types.SimpleNamespace(
        USE_CLIENT_DEFAULT=object(),
        CookieTypes=object,
        UseClientDefault=object,
        TimeoutTypes=object,
    )
    httpx._types = types.SimpleNamespace(
        URLTypes=object,
        HeaderTypes=object,
        QueryParamTypes=object,
        CookieTypes=object,
        RequestContent=object,
        RequestFiles=object,
        AuthTypes=object,
    )

    sys.modules['httpx'] = httpx

# stub passlib with minimal CryptContext
passlib_context = types.ModuleType('passlib.context')
passlib_context.CryptContext = lambda **kw: types.SimpleNamespace(hash=lambda p: p, verify=lambda p, h: p == h)
sys.modules['passlib.context'] = passlib_context
passlib_hash = types.ModuleType('passlib.hash')
passlib_hash.bcrypt = types.SimpleNamespace(hash=lambda p: p, verify=lambda p, h: p == h)
sys.modules['passlib.hash'] = passlib_hash

# stub Mongo and related modules used by db/mongo_utils
pymongo = types.ModuleType('pymongo')
pymongo.MongoClient = lambda *a, **k: None
pymongo.errors = types.SimpleNamespace(PyMongoError=Exception)
sys.modules['pymongo'] = pymongo
bson = types.ModuleType('bson')
bson.ObjectId = str
sys.modules['bson'] = bson

# stub custom helper modules used by main
stub_mongo_utils = types.ModuleType('mongo_utils')
stub_mongo_utils.update_resume = lambda *a, **kw: None
stub_mongo_utils.delete_resume_by_id = lambda *a, **kw: None
stub_mongo_utils.chat_find_one = lambda *a, **kw: None
stub_mongo_utils.chat_upsert = lambda *a, **kw: None
stub_mongo_utils.resumes_all = lambda: []
stub_mongo_utils.resumes_by_ids = lambda ids: []
stub_mongo_utils.resumes_collection = None
stub_mongo_utils.add_project_history = lambda *a, **kw: None
class DummyColl:
    def __getattr__(self, name):
        return lambda *a, **k: None

class DummyDB(dict):
    def __getitem__(self, key):
        return DummyColl()

stub_mongo_utils.db = DummyDB()
stub_mongo_utils._users = {}
stub_mongo_utils.ENV = 'test'
stub_mongo_utils._guard = lambda op: False
sys.modules['mongo_utils'] = stub_mongo_utils

# minimal template stub so FastAPI can be imported without jinja2
if 'fastapi.templating' not in sys.modules:
    templating = types.ModuleType('fastapi.templating')
    class Jinja2Templates:
        def __init__(self, directory=None):
            self.directory = directory
        def TemplateResponse(self, name, ctx, status_code=200):
            return types.SimpleNamespace(status_code=status_code)
    templating.Jinja2Templates = Jinja2Templates
    sys.modules['fastapi.templating'] = templating

if 'fastapi.testclient' not in sys.modules:
    testclient = types.ModuleType('fastapi.testclient')
    class TestClient:
        def __init__(self, app):
            self.app = app
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def get(self, path, **kw):
            import main, types as _t
            if path == '/login':
                req = _t.SimpleNamespace(url=_t.SimpleNamespace(path=path), cookies={})
                return main.login_form(req)
            return _t.SimpleNamespace(status_code=404)
        def post(self, path, data=None, **kw):
            import main, asyncio, types as _t
            if path == '/login':
                req = _t.SimpleNamespace(url=_t.SimpleNamespace(path=path), cookies={})
                return asyncio.run(main.login_post(req, username=data.get('username'), password=data.get('password')))
            return _t.SimpleNamespace(status_code=404)
    testclient.TestClient = TestClient
    sys.modules['fastapi.testclient'] = testclient

stub_pinecone_utils = types.ModuleType('pinecone_utils')
stub_pinecone_utils.add_resume_to_pinecone = lambda *a, **kw: None
stub_pinecone_utils.embed_text = lambda *a, **kw: []
stub_pinecone_utils.index = None
stub_pinecone_utils.search_best_resumes = lambda *a, **kw: []
sys.modules['pinecone_utils'] = stub_pinecone_utils
